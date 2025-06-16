#!/usr/bin/env python
"""
MATB EEG → feature table generator
================================

• **Per‑channel features**: absolute/log/relative band power, ratios, broadband
  summaries per EEG channel (or grand average if `--average_channels`).
• **Sliding‑window epoching**: default 5 s windows with 50 % overlap (step = 2.5 s).
• **Data‑directory constant**: edit `DEFAULT_DATA_DIR` once; run the script
  without CLI flags. `--data_dir` still overrides when needed.
• **Graceful XDF import**: if `mne.io.read_raw_xdf` is missing, the script now
  prints a clear message telling you to `pip install --upgrade mne pyxdf` and
  exits instead of crashing.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd

###############################################################################
# -------------------------- static paths ----------------------------------- #
###############################################################################

# <<<  CHANGE THIS PATH TO YOUR *.xdf FOLDER  >>>
DEFAULT_DATA_DIR = Path(r"C:/Users/yourname/Documents/xdf_recordings")

###############################################################################
# -------------------------- optional XDF reader ---------------------------- #
###############################################################################

# New: try to import read_raw_xdf once, give a helpful message otherwise
try:
    from mne.io import read_raw_xdf  # noqa: F401
except (ImportError, AttributeError):
    read_raw_xdf = None  # type: ignore[misc]

###############################################################################
# -------------------------- configuration ---------------------------------- #
###############################################################################

@dataclass
class Config:
    data_dir: Path
    out_csv: Path = Path("features_matb.csv")
    fs_new: int = 256                     # Hz after resampling
    epoch_length_sec: float = 5.0         # window size (s)
    step_size_sec: float = 2.5            # hop size (s) → 50 % overlap
    l_freq: float = 1.0                   # band‑pass low edge (Hz)
    h_freq: float = 40.0                  # band‑pass high edge (Hz)
    average_channels: bool = False        # True → grand‑average PSD
    # canonical band edges
    band_defs: Dict[str, Tuple[int, int]] | None = None

    def __post_init__(self):
        if self.band_defs is None:
            self.band_defs = {
                "delta": (1, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta":  (13, 30),
                "gamma": (30, 40),
            }

###############################################################################
# -------------------------- helpers ---------------------------------------- #
###############################################################################

def load_eeg_stream(xdf_path: Path, cfg: Config) -> mne.io.Raw:
    """Load EEG **and marker** streams, then resample, band‑pass & rereference.

    ⚠️ IMPORTANT: Do **not** filter out non‑EEG streams here; the XDF marker
    stream contains the task start/stop annotations we rely on. If you pass
    `include=["EEG"]` you get a perfectly fine signal but **zero annotations** –
    which explains why no epochs (and thus no CSV) were produced.
    """
    if read_raw_xdf is None:
        msg = (
            "mne.io.read_raw_xdf is not available in this environment."
            "▶  Install / upgrade with:"
            "    pip install --upgrade mne pyxdf"
            "▶  Then re-run the script."
        )
        raise ImportError(msg)

    # --- read ALL streams so annotations come along ---
    raw, _ = read_raw_xdf(xdf_path, preload=True)  # type: ignore[arg-type]

    # find first EEG stream (there may be AUX / EXG streams too)
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    raw.pick(eeg_picks)

    raw.resample(cfg.fs_new)
    raw.filter(cfg.l_freq, cfg.h_freq, fir_design="firwin", verbose="error")
    raw.set_eeg_reference("average", verbose="error")
    return raw


def slice_sliding_windows(raw: mne.io.Raw, start_samp: int, stop_samp: int, *, cfg: Config) -> List[mne.io.Epochs]:
    """Return Epoch objects in a sliding‑window fashion (win, step in *samples*)."""
    win = int(cfg.epoch_length_sec * cfg.fs_new)
    step = int(cfg.step_size_sec * cfg.fs_new)

    epochs: List[mne.io.Epochs] = []
    for beg in range(start_samp, stop_samp - win + 1, step):
        end = beg + win
        # Crop copy to avoid touching Raw
        ep_raw = raw.copy().crop(beg / cfg.fs_new, (end - 1) / cfg.fs_new)
        # Dummy event so we can wrap into an EpochsArray
        events = np.array([[0, 0, 1]])
        ep = mne.EpochsArray(
            ep_raw.get_data()[np.newaxis, ...], info=ep_raw.info,
            events=events, event_id={"dummy": 1}, verbose="error"
        )
        epochs.append(ep)
    return epochs


def epoch_between(raw: mne.io.Raw, event_id: dict, label_start: str, label_end: str, *, cfg: Config):
    """Yield `(Epochs, label)` for each start→end pair."""
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="error")
    start_code, end_code = event_id[label_start], event_id[label_end]

    boundaries: List[Tuple[int, int]] = []
    current: int | None = None
    for samp, _, code in events:
        if code == start_code:
            current = samp
        elif code == end_code and current is not None:
            boundaries.append((current, samp))
            current = None

    label = "easy" if "easy" in label_start else "hard"
    for beg, end in boundaries:
        for ep in slice_sliding_windows(raw, beg, end, cfg=cfg):
            yield ep, label


def compute_psd_features(epoch: mne.io.Epochs, *, cfg: Config) -> Dict[str, float]:
    """Return a flat dict of PSD‑based features for *one* epoch."""
    data = epoch.get_data()[0]  # (n_ch, n_times)
    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=cfg.fs_new, n_fft=cfg.fs_new, verbose="error"
    )

    if cfg.average_channels:
        psd = psd.mean(axis=0, keepdims=True)
        ch_names = ["avg"]
    else:
        ch_names = epoch.ch_names

    band_idx = {
        band: np.where((freqs >= lo) & (freqs < hi))[0]
        for band, (lo, hi) in cfg.band_defs.items()
    }
    eps = 1e-30

    feats: Dict[str, float] = {}
    for ch_i, ch_name in enumerate(ch_names):
        ch_psd = psd[ch_i]
        total_power = ch_psd[(freqs >= cfg.l_freq) & (freqs <= cfg.h_freq)].sum()

        # canonical bands
        band_power: Dict[str, float] = {}
        for band, idx in band_idx.items():
            p = ch_psd[idx].sum()
            band_power[band] = p
            base = f"{ch_name}_{band}"
            feats[f"{base}_pow"] = p
            feats[f"{base}_logpow"] = np.log10(p + eps)
            feats[f"{base}_relpow"] = p / (total_power + eps)

        # workload ratios
        theta, alpha, beta, gamma = (
            band_power["theta"],
            band_power["alpha"],
            band_power["beta"],
            band_power["gamma"],
        )
        feats[f"{ch_name}_theta_over_alpha"] = theta / (alpha + eps)
        feats[f"{ch_name}_beta_over_alpha"] = beta / (alpha + eps)
        feats[f"{ch_name}_gamma_over_theta_alpha"] = gamma / (theta + alpha + eps)

        # broadband summaries
        freqs_band = freqs[(freqs >= cfg.l_freq) & (freqs <= cfg.h_freq)]
        feats[f"{ch_name}_avg_power"] = total_power / len(freqs_band)
        feats[f"{ch_name}_spectral_centroid"] = (
            (freqs_band * ch_psd[(freqs >= cfg.l_freq) & (freqs <= cfg.h_freq)]).sum()
            / (total_power + eps)
        )
        feats[f"{ch_name}_peak_freq"] = freqs[np.argmax(ch_psd)]

    return feats

###############################################################################
# -------------------------- drivers ---------------------------------------- #
###############################################################################

def process_file(xdf_path: Path, cfg: Config) -> pd.DataFrame:
    try:
        raw = load_eeg_stream(xdf_path, cfg)
    except ImportError as e:
        print(f"[ERROR] {e}")
        return pd.DataFrame()

    annots = raw.annotations
    event_id = {d: i + 1 for i, d in enumerate(sorted(set(annots.description)))}
    required = {"start_easy_MATB", "end_easy_MATB", "start_hard_MATB", "end_hard_MATB"}
    if not required.issubset(event_id):
        print(f"[WARN] Missing annotations in {xdf_path.name} – skipping.")