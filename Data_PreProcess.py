"""

Batch EEG Epoching & Feature Extraction for MATB XDF files

---------------------------------------------------------

This script iterates over every ``.xdf`` recording found in ``DATA_DIR`` and

performs the following processing pipeline for **both** the *easy* and *hard*

MATB blocks that are annotated in each file:



1.  **Load & Pre‑process**

    * Detect the EEG stream in the XDF container and load it with a target

      sampling rate of ``fs_new``.

    * Drop auxiliary/trigger channels, band‑pass filter (``l_freq`` – ``h_freq``),

      re‑reference to average, and apply the *standard_1005* montage.



2.  **Epoching**

    * Identify the annotation *pairs* ``start_easy_MATB`` → ``end_easy_MATB`` and

      ``start_hard_MATB`` → ``end_hard_MATB``.

    * Chop the data between each pair into consecutive non‑overlapping epochs of

      length ``epoch_length_sec`` (default **4 s**).



3.  **Feature computation** (per epoch)

    * Welch PSD in the 1‑40 Hz range (``n_fft = fs_new`` → 1‑Hz resolution).

    * **Band‑limited power**: δ 1‑4 Hz, θ 4‑8 Hz, α 8‑13 Hz, β 13‑30 Hz, γ 30‑40 Hz.

    * Natural‑scale **and** ``log10``‑scaled band power.

    * Average broadband power, spectral centroid, and peak frequency.



4.  **Export**

    * All per‑epoch features for every file are concatenated into a single

      ``pandas`` DataFrame and saved to ``features_matb.csv``.

    * Each epoch row is tagged with its ``file_name`` and ``difficulty``

      ("easy"/"hard") for downstream ML‑ready usage.



Usage

-----

Simply point ``DATA_DIR`` to your folder with the ``.xdf`` recordings and run::



    python batch_epoch_features.py



Dependencies: mne~=1.7, mnelab, pyxdf, numpy, pandas

"""



from pathlib import Path

import numpy as np

import pandas as pd

from pyxdf import match_streaminfos, resolve_streams

from mnelab.io.xdf import read_raw_xdf

import mne

from mne.channels import make_standard_montage



# ---------------------------  USER‑CONFIGURABLE  --------------------------- #

DATA_DIR            = Path("c:/Users/marca/Documents/MAE/Research Project/3. Code/ml//xdf_recordings_all")   # folder containing *.xdf files

OUTPUT_FEATURE_CSV  = Path("features_matb_test.csv")

fs_new              = 256                        # resample rate (Hz)

epoch_length_sec    = 4.0                        # epoch duration (s)

l_freq, h_freq      = 1, 40                      # band‑pass limits (Hz)

# -------------------------------------------------------------------------- #



drop_channels = [

    "Trig1", "EX1", "EX2", "EX3", "EX4", "EX5", "EX6", "EX7", "EX8", *

    [f"AUX{i}" for i in range(1, 17)]

]



# ---------------------------  HELPER FUNCTIONS  --------------------------- #



def load_eeg_stream(xdf_path: Path):

    """Return an MNE Raw object containing only the EEG stream."""

    stream_id = match_streaminfos(

        resolve_streams(str(xdf_path)), [{"type": "EEG"}]

    )[0]

    return read_raw_xdf(str(xdf_path), stream_ids=[stream_id], fs_new=fs_new)





def preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:

    """Standard pre‑processing chain (drop, filter, reref, montage)."""

    raw.drop_channels([ch for ch in drop_channels if ch in raw.ch_names])

    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")

    raw.set_eeg_reference("average", verbose="ERROR")

    raw.rename_channels(lambda x: x.strip("."))



    std_montage = make_standard_montage("standard_1005")

    try:

        raw.set_montage(std_montage, verbose="ERROR")

    except ValueError:

        # fallback for BioSemi‑style caps (A1…B32 etc.)

        if any(ch[0] in {"A", "B", "C", "D"} and ch[1:].isdigit() for ch in raw.ch_names):

            n_eeg = len(raw.ch_names)

            bs_name = "biosemi64" if n_eeg <= 64 else "biosemi128" if n_eeg <= 128 else "biosemi256"

            try:

                raw.set_montage(make_standard_montage(bs_name), on_missing="ignore", verbose="ERROR")

            except Exception:

                raw.set_montage(std_montage, on_missing="ignore", verbose="ERROR")

        else:

            raw.set_montage(std_montage, on_missing="ignore", verbose="ERROR")

    return raw





def epoch_between(raw: mne.io.BaseRaw, start_id: int, end_id: int, difficulty: str):

    """Return list of consecutive epochs between *start* and *end* events."""

    events, _ = mne.events_from_annotations(raw, verbose="ERROR")

    mask = np.isin(events[:, 2], [start_id, end_id])

    rel_events = events[mask]

    rel_events = rel_events[np.argsort(rel_events[:, 0])]



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

            ep = mne.Epochs(raw, dummy, {"Custom": 999}, tmin=0, tmax=epoch_length_sec,

                            baseline=None, preload=True, verbose=False)

            ep.metadata = pd.DataFrame({"difficulty": [difficulty]})

            out.append(ep)

    return out





# ---------------------------  FEATURE ENGINEERING  ------------------------ #



_band_defs = {

    "delta": (1, 4),

    "theta": (4, 8),

    "alpha": (8, 13),

    "beta":  (13, 30),

    "gamma": (30, 40),

}

_eps = 1e-30  # avoid log10(0)





def compute_psd_features(epoch: mne.Epochs):

    """Compute power features for a single Epoch object."""

    data = epoch.get_data()  # shape (1, n_channels, n_times)

    psds, freqs = mne.time_frequency.psd_array_welch(

        data.squeeze(0), sfreq=fs_new, fmin=l_freq, fmax=h_freq,

        n_fft=fs_new, verbose="ERROR"

    )  # psds → [n_channels, n_freqs]



    mean_psd = psds.mean(axis=0)                # avg across channels

    total_power = mean_psd.mean()               # broadband power

    centroid = np.nan if mean_psd.sum() == 0 else (mean_psd * freqs).sum() / mean_psd.sum()

    peak_freq = freqs[np.argmax(mean_psd)]



    feats = {

        "avg_power": total_power,

        "spectral_centroid": centroid,

        "peak_freq": peak_freq,

    }



    # band‑limited power (absolute)

    for band, (f_low, f_high) in _band_defs.items():

        mask = (freqs >= f_low) & (freqs < f_high)

        band_pow = mean_psd[mask].mean()

        feats[f"pow_{band}"] = band_pow

        feats[f"log_pow_{band}"] = np.log10(band_pow + _eps)



    return feats





# ---------------------------  FILE‑LEVEL DRIVER  -------------------------- #



def process_file(xdf_path: Path) -> pd.DataFrame:

    print(f"\n→ Processing {xdf_path.name} …")

    raw = load_eeg_stream(xdf_path)

    preprocess_raw(raw)

    _, event_id = mne.events_from_annotations(raw, verbose="ERROR")



    required = ["start_easy_MATB", "end_easy_MATB", "start_hard_MATB", "end_hard_MATB"]

    if not all(k in event_id for k in required):

        print("  ! Missing required annotation labels → skipping file")

        return pd.DataFrame()



    epochs = []

    epochs += epoch_between(raw, event_id["start_easy_MATB"], event_id["end_easy_MATB"], "easy")

    epochs += epoch_between(raw, event_id["start_hard_MATB"], event_id["end_hard_MATB"], "hard")



    recs = []

    for idx, ep in enumerate(epochs):

        feat = compute_psd_features(ep)

        feat.update({

            "file_name": xdf_path.name,

            "epoch_idx": idx,

            "difficulty": ep.metadata["difficulty"].iloc[0],

        })

        recs.append(feat)



    print(f"  ✔ {len(recs)} epochs processed → features")

    return pd.DataFrame.from_records(recs)





# ---------------------------  MAIN  --------------------------------------- #



def main():

    dfs = []

    for xdf_path in sorted(DATA_DIR.glob("*.xdf")):

        print(xdf_path)

        df = process_file(xdf_path)

        if not df.empty:

            dfs.append(df)



    if not dfs:

        print("No features extracted – check your data & labels.")

        return



    all_feat = pd.concat(dfs, ignore_index=True)

    all_feat.to_csv(OUTPUT_FEATURE_CSV, index=False)

    print(f"\n✅ Saved {len(all_feat)} rows to {OUTPUT_FEATURE_CSV}")





if __name__ == "__main__":

    main()

