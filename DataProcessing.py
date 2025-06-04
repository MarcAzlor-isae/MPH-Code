"""
Batch EEG Epoching – Alpha-only Feature Table
============================================
Identical pipeline to your original script, but the final CSV keeps **only**

* alpha-band absolute power (column `alpha`)
* recording file name (`file_name`)
* epoch index (`epoch_idx`)
* condition label (`difficulty`)
"""

# ---------------- USER SETTINGS -------------------------------------------
from pathlib import Path
DATA_DIR            = Path(r"c:/Users/marca/Documents/MAE/Research Project/3. Code/ml/3rd_session_recordings")
OUTPUT_FEATURE_CSV  = Path("features_matb_alpha.csv")
fs_new              = 256          # Hz
epoch_length_sec    = 15.0         # s
l_freq, h_freq      = 1, 40        # Hz band-pass
# -------------------------------------------------------------------------

import numpy as np
import pandas as pd
from pyxdf import match_streaminfos, resolve_streams
from mnelab.io.xdf import read_raw_xdf
import mne
from mne.channels import make_standard_montage

drop_channels = [
    "Trig1", "EX1", "EX2", "EX3", "EX4", "EX5", "EX6", "EX7", "EX8",
    *[f"AUX{i}" for i in range(1, 17)]
]

# ---------------- HELPER: load & pre-process ------------------------------

def load_eeg_stream(xdf_path: Path):
    stream_id = match_streaminfos(resolve_streams(str(xdf_path)), [{"type": "EEG"}])[0]
    return read_raw_xdf(str(xdf_path), stream_ids=[stream_id], fs_new=fs_new)

def preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw.drop_channels([ch for ch in drop_channels if ch in raw.ch_names])
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    raw.rename_channels(lambda x: x.strip("."))
    std_montage = make_standard_montage("standard_1005")
    try:
        raw.set_montage(std_montage, verbose="ERROR")
    except ValueError:
        raw.set_montage(std_montage, on_missing="ignore", verbose="ERROR")
    return raw

# ---------------- EPOCHING ------------------------------------------------

def epoch_between(raw, start_id, end_id, difficulty):
    events, _ = mne.events_from_annotations(raw, verbose="ERROR")
    mask = np.isin(events[:, 2], [start_id, end_id])
    rel_events = events[mask][np.argsort(events[mask][:, 0])]

    epoch_len = int(epoch_length_sec * raw.info["sfreq"])
    out = []
    for i in range(0, len(rel_events) - 1, 2):
        if rel_events[i, 2] != start_id or rel_events[i+1, 2] != end_id:
            continue
        seg_start, seg_end = rel_events[i, 0], rel_events[i+1, 0]
        n_epochs = (seg_end - seg_start) // epoch_len
        for j in range(n_epochs):
            e_start = seg_start + j * epoch_len
            dummy = np.array([[e_start, 0, 999]])
            ep = mne.Epochs(raw, dummy, {"Custom": 999}, tmin=0,
                            tmax=epoch_length_sec, baseline=None,
                            preload=True, verbose=False)
            ep.metadata = pd.DataFrame({"difficulty": [difficulty]})
            out.append(ep)
    return out

# ---------------- ALPHA-ONLY FEATURE --------------------------------------

_alpha_limits = (8, 13)   # Hz
_eps = 1e-30

def compute_alpha(epoch: mne.Epochs) -> float:
    data = epoch.get_data().squeeze(0)                 # [n_ch, n_t]
    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=fs_new, fmin=l_freq, fmax=h_freq,
        n_fft=fs_new, verbose="ERROR")
    mean_psd = psd.mean(axis=0)                        # avg across channels
    mask = (freqs >= _alpha_limits[0]) & (freqs < _alpha_limits[1])
    return mean_psd[mask].mean() + _eps               # absolute α power

# ---------------- PROCESS ONE FILE ----------------------------------------

def process_file(xdf_path: Path) -> pd.DataFrame:
    print(f"\n→ {xdf_path.name}")
    raw = preprocess_raw(load_eeg_stream(xdf_path))
    _, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    req = ["start_easy_MATB", "end_easy_MATB",
           "start_hard_MATB", "end_hard_MATB"]
    if not all(r in event_id for r in req):
        print("  ! Missing annotations, skipping")
        return pd.DataFrame()

    epochs = []
    epochs += epoch_between(raw, event_id["start_easy_MATB"],
                            event_id["end_easy_MATB"], "easy")
    epochs += epoch_between(raw, event_id["start_hard_MATB"],
                            event_id["end_hard_MATB"], "hard")

    records = []
    for idx, ep in enumerate(epochs):
        records.append({
            "alpha":      compute_alpha(ep),
            "file_name":  xdf_path.name,
            "epoch_idx":  idx,
            "difficulty": ep.metadata["difficulty"].iloc[0],
        })
    print(f"  ✔ {len(records)} epochs")
    return pd.DataFrame.from_records(records)

# ---------------- MAIN ----------------------------------------------------

def main():
    dfs = [process_file(p) for p in sorted(DATA_DIR.glob("*.xdf"))]
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        print("No features extracted.")
        return
    all_feat = pd.concat(dfs, ignore_index=True)
    all_feat.to_csv(OUTPUT_FEATURE_CSV, index=False)
    print(f"\n✅ Saved {len(all_feat)} rows → {OUTPUT_FEATURE_CSV}")

if __name__ == "__main__":
    main()
