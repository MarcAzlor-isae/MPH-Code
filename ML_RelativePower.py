#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML_combined_pipeline.py

Enhanced ML pipeline that combines features from both scripts:
- Proper cross-validation without data leakage
- Feature selection and scaling done within each fold
- Multiple models with hyperparameter tuning
- Feature importance extraction for tree-based models
- Comprehensive evaluation metrics
- ENHANCED: Support for relative power features and per-channel ratios
- ADDED: Confusion matrix printing for all models
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ---- CONFIG ----
FEATURE_CSV   = "features_matb_15s_relative.csv"
TARGET_COLUMN = "difficulty"

# Feature type configuration
FEATURE_MODE = "power"  # Options: "power", "ratios", "both"

# ENHANCED: Feature detail configuration
POWER_FEATURE_MODE = "absolute_only"  # Options: "absolute_only", "relative_only", "both_abs_rel", "log_relative_only"

# Power band configuration (used when FEATURE_MODE is "power" or "both")
BANDS         = ["alpha"] 

# Ratio configurations (used when FEATURE_MODE is "ratios" or "both")
RATIO_PAIRS   = [
    #("theta", "alpha"),    # Theta/Alpha ratio (common in attention studies)
    ("alpha", "beta"),     # Alpha/Beta ratio (relaxation vs activation)
    #("theta", "beta"),     # Theta/Beta ratio (ADHD marker)
    #("delta", "theta"),    # Delta/Theta ratio
    #("beta", "gamma"),     # Beta/Gamma ratio (high frequency activity)
]
CHANNELS_BY_BAND = {
    "alpha": ["A27", "B32", "A29", "A25", "B31", "A26", "A21", "B25"],
}

RANDOM_STATE  = 42
NUM_FOLDS     = 5
N_SELECT      = 64  # Number of features to select (if > total features, all are used)

# Model configurations with hyperparameter grids
MODELS = { 
    "lr": {
        "name": "Logistic Regression",
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "param_grid": {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__penalty": ["l2", "l1"],
            "clf__solver": ["liblinear"]
        }
    },
    "lda": {
        "name": "Linear Discriminant Analysis",
        "estimator": LinearDiscriminantAnalysis(),
        "param_grid": {
            "clf__solver": ["svd", "lsqr"]
        }
    },
    "svm": {
        "name": "SVM (RBF kernel)",
        "estimator": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "param_grid": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto", 0.001, 0.01]
        }
    },
    "rf": {
        "name": "Random Forest",
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": [2, 5, 10]
        }
    },
    "gb": {
        "name": "Gradient Boosting",
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__max_depth": [3, 6, 9]
        }
    }
}

# ENHANCED: Feature mode configurations
POWER_FEATURE_MODES = {
    "absolute_only": {
        "include_relative": False, 
        "include_log_relative": False,
        "exclude_absolute": False
    },
    "relative_only": {
        "include_relative": True, 
        "include_log_relative": False,
        "exclude_absolute": True
    },
    "both_abs_rel": {
        "include_relative": True, 
        "include_log_relative": True,
        "exclude_absolute": False
    },
    "log_relative_only": {
        "include_relative": False, 
        "include_log_relative": True,
        "exclude_absolute": True
    }
}

# ---- END CONFIG ----


def generate_band_power_features(bands, mode="both_abs_rel"):
    """
    Return the list of power-band feature names requested, but only for the
    channels defined in CHANNELS_BY_BAND for each band.

    If a band is not in CHANNELS_BY_BAND we fall back to all 64 channels.
    """
    mode_cfg = POWER_FEATURE_MODES.get(mode, POWER_FEATURE_MODES["both_abs_rel"])
    features = []

    for band in bands:
        # pick the channel list for this band, or default to the full 64
        chan_list = CHANNELS_BY_BAND.get(band.lower())
        if chan_list is None:                         # fall back → A1…B32
            chan_list = [f"{p}{i}" for p in ("A", "B") for i in range(1, 33)]

        for ch in chan_list:
            # absolute
            if not mode_cfg["exclude_absolute"]:
                features.append(f"pow_{band}_{ch}")
            # relative %
            if mode_cfg["include_relative"]:
                features.append(f"pow_{band}_rel_{ch}")
            # log-relative
            if mode_cfg["include_log_relative"]:
                features.append(f"log_pow_{band}_rel_{ch}")

    return features



def generate_ratio_features(ratio_pairs, include_per_channel_ratios=True):
    """
    Generate ratio feature names for specified band pairs across all 64 channels.
    ENHANCED: Now includes common per-channel EEG ratios.
    
    Args:
        ratio_pairs: List of tuples for custom ratios
        include_per_channel_ratios: Whether to include alpha/beta and theta/beta ratios per channel
    
    Returns:
        List of feature names
    """
    features = []
    
    # Custom ratio features (computed from per-channel absolute power)
    for numerator, denominator in ratio_pairs:
        for prefix in ("A", "B"):
            for i in range(1, 33):
                features.append(f"ratio_{numerator}_{denominator}_{prefix}{i}")
    
    # Common EEG ratios per channel (from preprocessing script)
    if include_per_channel_ratios:
        common_ratios = [("alpha", "beta"), ("theta", "beta")]
        for num_band, den_band in common_ratios:
            for prefix in ("A", "B"):
                for i in range(1, 33):
                    features.append(f"{num_band}_{den_band}_ratio_{prefix}{i}")
    
    return features


def compute_ratio_features(df, ratio_pairs):
    """
    Compute ratio features from power band features and add them to the dataframe.
    Returns modified dataframe with ratio features added.
    """
    df_with_ratios = df.copy()
    
    for numerator, denominator in ratio_pairs:
        print(f"Computing {numerator}/{denominator} ratios...")
        
        # Generate channel names for both bands
        num_channels = []
        den_channels = []
        
        for prefix in ("A", "B"):
            for i in range(1, 33):
                num_channels.append(f"pow_{numerator}_{prefix}{i}")
                den_channels.append(f"pow_{denominator}_{prefix}{i}")
        
        # Compute ratios for each channel
        for num_ch, den_ch in zip(num_channels, den_channels):
            if num_ch in df.columns and den_ch in df.columns:
                ratio_name = f"ratio_{numerator}_{denominator}_{num_ch.split('_')[-1]}"
                
                # Compute ratio with small epsilon to avoid division by zero
                epsilon = 1e-10
                df_with_ratios[ratio_name] = (
                    df[num_ch] / (df[den_ch] + epsilon)
                )
            else:
                missing = []
                if num_ch not in df.columns:
                    missing.append(num_ch)
                if den_ch not in df.columns:
                    missing.append(den_ch)
                print(f"   Warning: Missing columns for ratio computation: {missing}")
    
    return df_with_ratios


def get_required_bands_for_ratios(ratio_pairs):
    """
    Get all unique bands required for computing the specified ratios.
    ENHANCED: Also includes bands needed for common ratios.
    """
    required_bands = set()
    for numerator, denominator in ratio_pairs:
        required_bands.add(numerator)
        required_bands.add(denominator)
    
    # Add bands for common ratios (alpha/beta, theta/beta)
    required_bands.update(["alpha", "beta", "theta"])
    
    return list(required_bands)


def load_and_prepare_data(csv_path, selected_features, feature_mode="both", ratio_pairs=None):
    """
    Load CSV and prepare data for ML pipeline.
    
    Args:
        csv_path: Path to the CSV file
        selected_features: List of feature names to use
        feature_mode: "power", "ratios", or "both"
        ratio_pairs: List of tuples for ratio computation
    
    Returns:
        filtered dataframe and list of available features
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Cannot find feature file: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Check target column
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"CSV must have a '{TARGET_COLUMN}' column.")

    # Compute ratio features if needed
    if feature_mode in ["ratios", "both"] and ratio_pairs:
        print("Computing custom ratio features...")
        df = compute_ratio_features(df, ratio_pairs)
        print(f"Added custom ratio features for {len(ratio_pairs)} band pairs")

    # Filter features that actually exist in the dataset
    available_features = [f for f in selected_features if f in df.columns]
    missing_features = [f for f in selected_features if f not in df.columns]
    
    if missing_features:
        print(f"⚠️  Warning: {len(missing_features)} features missing from CSV")
        if len(missing_features) <= 5:
            print(f"   Missing: {missing_features}")
        else:
            print(f"   First 5 missing: {missing_features[:5]}")

    if len(available_features) == 0:
        raise RuntimeError("No requested features found in the CSV.")

    # Keep only available features + target
    keep_cols = available_features + [TARGET_COLUMN]
    filtered_df = df[keep_cols].copy()

    return filtered_df, available_features


def create_pipeline(estimator, n_features):
    """
    Create a pipeline with imputation, scaling, feature selection, and classification.
    Feature selection is only applied if n_features > N_SELECT.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    
    # Add feature selection if we have more features than N_SELECT
    if n_features > N_SELECT:
        steps.append(("selector", SelectKBest(f_classif, k=N_SELECT)))
    
    steps.append(("clf", estimator))
    
    return Pipeline(steps)


def print_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """
    Print a nicely formatted confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Title for the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{title}:")
    print("-" * len(title))
    
    # Print header
    print(f"{'':>12}", end="")
    for class_name in class_names:
        print(f"{class_name:>12}", end="")
    print()
    
    # Print matrix rows
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>12}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>12}", end="")
        print()
    
    # Print percentages
    print(f"\nRow Percentages (% of actual class):")
    print(f"{'':>12}", end="")
    for class_name in class_names:
        print(f"{class_name:>12}", end="")
    print()
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>12}", end="")
        row_sum = cm[i, :].sum()
        for j in range(len(class_names)):
            if row_sum > 0:
                percentage = (cm[i, j] / row_sum) * 100
                print(f"{percentage:>11.1f}%", end="")
            else:
                print(f"{'0.0%':>12}", end="")
        print()


def evaluate_model(model_key, model_config, X, y, cv_folds, class_names):
    """
    Evaluate a single model using cross-validation with proper data handling.
    Returns best score, best parameters, and fitted model.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_config['name']} ({model_key})")
    print(f"{'='*60}")
    
    # Create pipeline
    pipeline = create_pipeline(model_config['estimator'], X.shape[1])
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=model_config['param_grid'],
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    print("Running GridSearchCV...")
    grid_search.fit(X, y)
    
    # Results
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print(f"Best CV Accuracy: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    
    # Get cross-validation predictions for confusion matrix
    print("Generating cross-validation predictions for confusion matrix...")
    y_pred_cv = cross_val_predict(best_model, X, y, cv=cv_folds)
    
    # Print confusion matrix for cross-validation results
    print_confusion_matrix(
        y, y_pred_cv, class_names, 
        f"Cross-Validation Confusion Matrix - {model_config['name']}"
    )
    
    return best_score, best_params, best_model


def extract_feature_importances(model, model_key, feature_names):
    """
    Extract and display feature importances for tree-based models.
    """
    if model_key not in ["rf", "gb"]:
        return
    
    print(f"\nFeature Importances for {model_key.upper()}:")
    print("-" * 40)
    
    # Get the classifier from the pipeline
    clf = model.named_steps["clf"]
    
    # Handle feature selection case
    if "selector" in model.named_steps:
        # Get selected feature indices
        selector = model.named_steps["selector"]
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        importances = clf.feature_importances_
    else:
        selected_features = feature_names
        importances = clf.feature_importances_
    
    # Sort by importance
    feature_importance_pairs = sorted(
        zip(selected_features, importances), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Display top 10 features
    print("Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance_pairs[:10], 1):
        print(f"{i:2d}. {feature:<30} {importance:.4f}")


def main():
    print("ENHANCED EEG-based Difficulty Classification Pipeline")
    print("=" * 55)
    
    # Determine what features to generate based on FEATURE_MODE
    requested_features = []
    
    # 1. Generate power band features if needed
    if FEATURE_MODE in ["power", "both"]:
        power_features = generate_band_power_features(BANDS, mode=POWER_FEATURE_MODE)
        requested_features.extend(power_features)
        print(f"Power bands: {BANDS}")
        print(f"Power feature mode: {POWER_FEATURE_MODE}")
        print(f"Generated {len(power_features)} power band features")
    
    # 2. Generate ratio features if needed
    if FEATURE_MODE in ["ratios", "both"]:
        # For ratios, we need the underlying power bands too
        if FEATURE_MODE == "ratios":
            # Get all bands needed for the ratios
            required_bands = get_required_bands_for_ratios(RATIO_PAIRS)
            power_features = generate_band_power_features(required_bands, mode=POWER_FEATURE_MODE)
            requested_features.extend(power_features)
            print(f"Required bands for ratios: {required_bands}")
            print(f"Generated {len(power_features)} power band features for ratio computation")
        
        ratio_features = generate_ratio_features(RATIO_PAIRS, include_per_channel_ratios=True)
        requested_features.extend(ratio_features)
        print(f"Custom ratio computation: {RATIO_PAIRS}")
        print(f"Including per-channel alpha/beta and theta/beta ratios: Yes")
        print(f"Generated {len(ratio_features)} ratio features")
    
    print(f"Total requested features: {len(requested_features)}")
    
    # 3. Load and prepare data
    data, available_features = load_and_prepare_data(
        FEATURE_CSV, 
        requested_features,
        feature_mode=FEATURE_MODE,
        ratio_pairs=RATIO_PAIRS if FEATURE_MODE in ["ratios", "both"] else None
    )
    print(f"\nDataset loaded: {len(data)} samples, {len(available_features)} features")
    
    # 4. Prepare X and y
    X = data[available_features].values
    y = data[TARGET_COLUMN].values
    
    # 5. Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    
    print(f"Target distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # 6. ENHANCED: Feature type breakdown
    if FEATURE_MODE in ["ratios", "both"]:
        power_feats = [f for f in available_features if f.startswith("pow_")]
        ratio_feats = [f for f in available_features if f.startswith("ratio_") or "_ratio_" in f]
        
        # Further breakdown of power features
        abs_power_feats = [f for f in power_feats if "_rel_" not in f and not f.startswith("log_")]
        rel_power_feats = [f for f in power_feats if "_rel_" in f and not f.startswith("log_")]
        log_rel_power_feats = [f for f in power_feats if f.startswith("log_pow_") and "_rel_" in f]
        
        print(f"\nENHANCED Feature breakdown:")
        print(f"  Absolute power features: {len(abs_power_feats)}")
        print(f"  Relative power features: {len(rel_power_feats)}")
        print(f"  Log relative power features: {len(log_rel_power_feats)}")
        print(f"  Ratio features: {len(ratio_feats)}")
        print(f"  Total: {sum([len(abs_power_feats), len(rel_power_feats), len(log_rel_power_feats), len(ratio_feats)])}")
    
    # 7. Set up cross-validation
    cv_folds = StratifiedKFold(
        n_splits=NUM_FOLDS, 
        shuffle=True, 
        random_state=RANDOM_STATE
    )
    
    # 8. Evaluate all models
    results = {}
    best_overall_score = 0
    best_overall_model = None
    best_overall_key = None
    
    for model_key, model_config in MODELS.items():
        try:
            score, params, model = evaluate_model(
                model_key, model_config, X, y_encoded, cv_folds, class_names
            )
            
            results[model_key] = {
                'score': score,
                'params': params,
                'model': model
            }
            
            # Track best model
            if score > best_overall_score:
                best_overall_score = score
                best_overall_model = model
                best_overall_key = model_key
            
            # Extract feature importances for tree-based models
            extract_feature_importances(model, model_key, available_features)
            
        except Exception as e:
            print(f"Error evaluating {model_key}: {str(e)}")
            continue
    
    # 9. Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print(f"Configuration:")
    print(f"  Feature Mode: {FEATURE_MODE}")
    print(f"  Power Feature Mode: {POWER_FEATURE_MODE}")
    if FEATURE_MODE in ["power", "both"]:
        print(f"  Bands: {BANDS}")
    if FEATURE_MODE in ["ratios", "both"]:
        print(f"  Custom Ratio Pairs: {RATIO_PAIRS}")
        print(f"  Per-channel Ratios: alpha/beta, theta/beta")
    print(f"  Total Features: {len(available_features)}")
    print(f"  Feature Selection: {'Yes' if len(available_features) > N_SELECT else 'No'} (N_SELECT={N_SELECT})")
    
    print(f"\n{'Model':<20} {'CV Accuracy':<12}")
    print("-" * 35)
    
    for model_key, result in results.items():
        model_name = MODELS[model_key]['name']
        score = result['score']
        print(f"{model_name:<20} {score:.4f}")
    
    if best_overall_model is not None:
        print(f"\nBest Model: {MODELS[best_overall_key]['name']}")
        print(f"Best CV Accuracy: {best_overall_score:.4f}")
        
        # Fit best model on full data for final evaluation
        print("\nFitting best model on full dataset...")
        best_overall_model.fit(X, y_encoded)
        
        # Make predictions for classification report and confusion matrix
        y_pred = best_overall_model.predict(X)
        
        print("\nClassification Report on Full Dataset:")
        print(classification_report(
            y_encoded, y_pred, 
            target_names=class_names
        ))
        
        # Print confusion matrix for best model on full dataset
        print_confusion_matrix(
            y_encoded, y_pred, class_names,
            f"Full Dataset Confusion Matrix - {MODELS[best_overall_key]['name']}"
        )
    
    print("\nENHANCED Pipeline completed successfully!")


if __name__ == "__main__":
    main()