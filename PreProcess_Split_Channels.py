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
      length ``epoch_length_sec`` (default **4 s**).

3.  **Feature computation** (per epoch)
    * Welch PSD in the 1‑40 Hz range with proper windowing and overlap.
    * **Band‑limited power**: δ 1‑4 Hz, θ 4‑8 Hz, α 8‑13 Hz, β 13‑30 Hz, γ 30‑40 Hz.
    * Natural‑scale **and** ``log10``‑scaled band power.
    * Relative band power (normalized by total power).
    * Average broadband power, spectral centroid, and peak frequency.
    * Per-channel features for spatial analysis.

4.  **Export**
    * All per‑epoch features for every file are concatenated into a single
      ``pandas`` DataFrame and saved to ``features_matb.csv``.
    * Each epoch row is tagged with its ``file_name`` and ``difficulty``
      ("easy"/"hard") for downstream ML‑ready usage.

Usage
-----
Simply point ``DATA_DIR`` to your folder with the ``.xdf`` recordings and run::

    python batch_epoch_features.py

Dependencies: mne~=1.7, mnelab, pyxdf, numpy, pandas, scipy
"""

from pathlib import Path
import numpy as np
import pandas as pd
from pyxdf import match_streaminfos, resolve_streams
from mnelab.io.xdf import read_raw_xdf
import mne
from mne.channels import make_standard_montage
from scipy import signal
import warnings

# ---------------------------  USER‑CONFIGURABLE  --------------------------- #
DATA_DIR            = Path("c:/Users/marca/Documents/MAE/Research Project/3. Code/ml//3rd_session_recordings")
OUTPUT_FEATURE_CSV  = Path("features_matb_improved_30s.csv")
fs_new              = 256                        # resample rate (Hz)
epoch_length_sec    = 30.0                        # epoch duration (s)
l_freq, h_freq      = 1, 40                      # band‑pass limits (Hz)
# -------------------------------------------------------------------------- #

drop_channels = [
    "Trig1", "EX1", "EX2", "EX3", "EX4", "EX5", "EX6", "EX7", "EX8",
    *[f"AUX{i}" for i in range(1, 17)]
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
    # Drop channels
    raw.drop_channels([ch for ch in drop_channels if ch in raw.ch_names])
    
    # Filter - using a more conservative approach
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR", method='fir')
    
    # Set reference
    raw.set_eeg_reference("average", verbose="ERROR")
    
    # Clean channel names
    raw.rename_channels(lambda x: x.strip("."))

    # Set montage
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
            # Create proper event array for epoching
            dummy_events = np.array([[e_start, 0, 999]])
            
            try:
                ep = mne.Epochs(raw, dummy_events, {"Custom": 999}, 
                              tmin=0, tmax=epoch_length_sec,
                              baseline=None, preload=True, verbose=False)
                ep.metadata = pd.DataFrame({"difficulty": [difficulty]})
                out.append(ep)
            except Exception as e:
                print(f"  Warning: Could not create epoch {j} for {difficulty} condition: {e}")
                continue
    
    return out

# ---------------------------  FEATURE ENGINEERING  ------------------------ #

_band_defs = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 40),
}
_eps = 1e-12  # avoid log10(0)

def compute_psd_features(epoch: mne.Epochs):
    """Compute power features for a single Epoch object."""
    try:
        data = epoch.get_data()  # shape (1, n_channels, n_times)
        if data.shape[0] == 0:
            return {}
        
        data = data.squeeze(0)  # Remove epoch dimension: (n_channels, n_times)
        sfreq = epoch.info['sfreq']
        
        # Compute PSD using Welch's method with proper parameters
        # Use 50% overlap and appropriate windowing
        nperseg = min(int(sfreq * 2), data.shape[1])  # 2-second windows or full length
        noverlap = nperseg // 2
        
        # Compute PSD for each channel
        psds_all_channels = []
        for ch_idx in range(data.shape[0]):
            freqs, psd = signal.welch(data[ch_idx], fs=sfreq, 
                                    nperseg=nperseg, noverlap=noverlap,
                                    window='hann', detrend='constant')
            # Only keep frequencies within our band of interest
            freq_mask = (freqs >= l_freq) & (freqs <= h_freq)
            psds_all_channels.append(psd[freq_mask])
        
        psds = np.array(psds_all_channels)  # shape: (n_channels, n_freqs)
        freqs = freqs[freq_mask]
        
        # Average PSD across channels for global features
        mean_psd = psds.mean(axis=0)
        
        # Compute broadband features
        total_power = mean_psd.sum()  # Sum across frequencies (more appropriate than mean)
        
        # Spectral centroid (weighted average frequency)
        if total_power > 0:
            centroid = np.sum(mean_psd * freqs) / total_power
        else:
            centroid = np.nan
        
        # Peak frequency
        peak_freq = freqs[np.argmax(mean_psd)]
        
        # Initialize feature dictionary
        feats = {
            "total_power": total_power,
            "spectral_centroid": centroid,
            "peak_freq": peak_freq,
        }
        
        # Compute band-limited power features
        for band, (f_low, f_high) in _band_defs.items():
            # Find frequency indices for this band
            band_mask = (freqs >= f_low) & (freqs < f_high)
            
            if np.any(band_mask):
                # Absolute band power (sum across frequencies in band)
                band_power_abs = mean_psd[band_mask].sum()
                
                # Relative band power (normalized by total power)
                band_power_rel = band_power_abs / total_power if total_power > 0 else 0
                
                # Log-transformed power
                band_power_log = np.log10(band_power_abs + _eps)
                
                feats[f"pow_{band}"] = band_power_abs
                feats[f"pow_{band}_rel"] = band_power_rel
                feats[f"log_pow_{band}"] = band_power_log
                
                # Per-channel band power (for spatial analysis)
                for ch_idx, ch_name in enumerate(epoch.ch_names):
                    ch_band_power = psds[ch_idx][band_mask].sum()
                    feats[f"pow_{band}_{ch_name}"] = ch_band_power
            else:
                # If no frequencies in band, set to zero/NaN
                feats[f"pow_{band}"] = 0
                feats[f"pow_{band}_rel"] = 0
                feats[f"log_pow_{band}"] = np.log10(_eps)
        
        # Additional spectral features
        # Spectral edge frequency (95% of power)
        cumsum_psd = np.cumsum(mean_psd)
        edge_95 = freqs[np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0][0]] if len(cumsum_psd) > 0 else np.nan
        feats["spectral_edge_95"] = edge_95
        
        # Alpha/beta ratio (commonly used in EEG analysis)
        alpha_power = feats.get("pow_alpha", 0)
        beta_power = feats.get("pow_beta", 0)
        feats["alpha_beta_ratio"] = alpha_power / (beta_power + _eps)
        
        # Theta/beta ratio (associated with attention)
        theta_power = feats.get("pow_theta", 0)
        feats["theta_beta_ratio"] = theta_power / (beta_power + _eps)
        
        return feats
        
    except Exception as e:
        print(f"  Error computing PSD features: {e}")
        return {}

# ---------------------------  FILE‑LEVEL DRIVER  -------------------------- #

def process_file(xdf_path: Path) -> pd.DataFrame:
    """Process a single XDF file and return DataFrame with features."""
    print(f"\n→ Processing {xdf_path.name} …")
    
    try:
        raw = load_eeg_stream(xdf_path)
        raw = preprocess_raw(raw)
        
        # Get events and check for required annotations
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        
        required = ["start_easy_MATB", "end_easy_MATB", "start_hard_MATB", "end_hard_MATB"]
        missing = [k for k in required if k not in event_id]
        
        if missing:
            print(f"  ! Missing required annotations: {missing} → skipping file")
            return pd.DataFrame()
        
        # Create epochs for both conditions
        epochs = []
        easy_epochs = epoch_between(raw, event_id["start_easy_MATB"], event_id["end_easy_MATB"], "easy")
        hard_epochs = epoch_between(raw, event_id["start_hard_MATB"], event_id["end_hard_MATB"], "hard")
        
        epochs.extend(easy_epochs)
        epochs.extend(hard_epochs)
        
        if not epochs:
            print("  ! No valid epochs found → skipping file")
            return pd.DataFrame()
        
        # Extract features from each epoch
        records = []
        for idx, ep in enumerate(epochs):
            feat = compute_psd_features(ep)
            
            if feat:  # Only add if features were successfully computed
                feat.update({
                    "file_name": xdf_path.name,
                    "epoch_idx": idx,
                    "difficulty": ep.metadata["difficulty"].iloc[0],
                    "n_channels": len(ep.ch_names),
                    "sfreq": ep.info['sfreq']
                })
                records.append(feat)
        
        print(f"  ✔ {len(records)} epochs processed → features extracted")
        return pd.DataFrame.from_records(records)
        
    except Exception as e:
        print(f"  ! Error processing {xdf_path.name}: {e}")
        return pd.DataFrame()

# ---------------------------  MAIN  --------------------------------------- #

def main():
    """Main processing loop."""
    print("Starting EEG feature extraction pipeline...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output file: {OUTPUT_FEATURE_CSV}")
    
    # Find all XDF files
    xdf_files = list(DATA_DIR.glob("*.xdf"))
    print(f"Found {len(xdf_files)} XDF files")
    
    if not xdf_files:
        print("No XDF files found in the specified directory!")
        return
    
    # Process each file
    dfs = []
    for xdf_path in sorted(xdf_files):
        df = process_file(xdf_path)
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        print("No features extracted – check your data & labels.")
        return
    
    # Combine all features
    all_feat = pd.concat(dfs, ignore_index=True)
    
    # Save results
    all_feat.to_csv(OUTPUT_FEATURE_CSV, index=False)
    print(f"\n✅ Saved {len(all_feat)} rows to {OUTPUT_FEATURE_CSV}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Total epochs: {len(all_feat)}")
    print(f"  Files processed: {all_feat['file_name'].nunique()}")
    print(f"  Easy epochs: {sum(all_feat['difficulty'] == 'easy')}")
    print(f"  Hard epochs: {sum(all_feat['difficulty'] == 'hard')}")
    print(f"  Feature columns: {len(all_feat.columns)}")

if __name__ == "__main__":
    main()