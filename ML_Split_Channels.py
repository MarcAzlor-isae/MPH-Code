"""
Enhanced EEG MATB Task Difficulty Classifier - Alpha Power Features Only (64 Channels)
====================================================================================
ML classifier for easy/hard MATB task classification using alpha power features from A1-A32, B1-B32.
Supports both epoch-level and file-level (participant-level) classification.
Includes feature selection and comprehensive analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input data
FEATURE_CSV = "features_matb_improved_30s.csv"  # Output from enhanced feature extraction script
TARGET_COLUMN = "difficulty"  # 'easy' or 'hard'

# Generate alpha power feature names for 64 channels: A1-A32, B1-B32
def generate_alpha_power_features():
    """Generate all alpha power feature names for 64 channels (A1-A32, B1-B32)."""
    features = []
    
    # Generate electrode names A1-A32, B1-B32 (total 64 channels)
    for letter in ['A', 'B']:
        for number in range(1, 33):  # 1 to 32 for each letter (total 64)
            electrode = f"{letter}{number}"
            feature_name = f"pow_delta_{electrode}"  # Fixed: alpha instead of theta
            features.append(feature_name)
    
    print(f"Generated {len(features)} alpha power features:")
    print(f"A channels: A1-A32 ({32} channels)")
    print(f"B channels: B1-B32 ({32} channels)")
    print(f"Total: {len(features)} channels")
    
    return features

# Alpha power features - all 64 electrodes (A1-A32, B1-B32)
ALPHA_POWER_FEATURES = generate_alpha_power_features()

# Only one feature set to test
FEATURE_SETS = {
    "Alpha Power Features (64 channels)": ALPHA_POWER_FEATURES
}

# Model settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
N_FEATURE_SELECTION = 32  # Select top 32 most discriminative channels

# ============================================================================
# DATA LOADING AND EXPLORATION
# ============================================================================

def load_and_explore_data(csv_path):
    """Load feature data and perform basic exploration."""
    print("📊 Loading and exploring data...")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    
    # Check which alpha power features are available
    available_alpha_features = [f for f in ALPHA_POWER_FEATURES if f in df.columns]
    missing_alpha_features = [f for f in ALPHA_POWER_FEATURES if f not in df.columns]
    
    print(f"\nAlpha Power Features (64 channels):")
    print(f"  Expected: {len(ALPHA_POWER_FEATURES)} features (A1-A32, B1-B32)")
    print(f"  Available: {len(available_alpha_features)} features")
    
    if missing_alpha_features:
        print(f"  Missing: {len(missing_alpha_features)} features")
        print(f"  First few missing: {missing_alpha_features[:10]}...")
        
        # Show which channels are missing
        missing_electrodes = [f.replace('pow_alpha_', '') for f in missing_alpha_features]
        print(f"  Missing electrodes: {', '.join(missing_electrodes[:20])}...")
    else:
        print(f"  ✅ All 64 alpha power features found!")
    
    # Show sample of available alpha features
    if available_alpha_features:
        sample_electrodes = [f.replace('pow_alpha_', '') for f in available_alpha_features[:10]]
        print(f"  Sample electrodes: {', '.join(sample_electrodes)}...")
    
    print(f"\nDataset preview:")
    if available_alpha_features:
        preview_cols = ['file_name', TARGET_COLUMN] + available_alpha_features[:5]
        print(df[preview_cols].head())
    else:
        print(df.head())
    
    # Basic statistics
    print(f"\nClass distribution:")
    print(df[TARGET_COLUMN].value_counts())
    print(f"\nClass proportions:")
    print(df[TARGET_COLUMN].value_counts(normalize=True))
    
    # Files and epochs info
    print(f"\nDataset info:")
    print(f"  Number of files: {df['file_name'].nunique()}")
    print(f"  Total epochs: {len(df)}")
    
    epochs_per_file = df.groupby('file_name').size()
    print(f"  Epochs per file - Mean: {epochs_per_file.mean():.1f}, Range: {epochs_per_file.min()}-{epochs_per_file.max()}")
    
    epochs_per_condition = df.groupby(['file_name', 'difficulty']).size()
    print(f"  Epochs per condition per file - Mean: {epochs_per_condition.mean():.1f}")
    
    available_features = {"Alpha Power Features (64 channels)": available_alpha_features}
    return df, available_features

def analyze_alpha_feature_statistics(df, alpha_features, top_n=20):
    """Analyze statistical differences for alpha power features across 64 channels."""
    from scipy.stats import ttest_ind, mannwhitneyu
    
    print(f"\n📈 Statistical Analysis of 64-Channel Alpha Power")
    print("=" * 60)
    
    easy_data = df[df[TARGET_COLUMN] == 'easy']
    hard_data = df[df[TARGET_COLUMN] == 'hard']
    
    significant_features = []
    
    print(f"Analyzing {len(alpha_features)} alpha power features (64 channels)...")
    
    for feature in alpha_features:
        if feature not in df.columns:
            continue
            
        easy_values = easy_data[feature].dropna()
        hard_values = hard_data[feature].dropna()
        
        if len(easy_values) == 0 or len(hard_values) == 0:
            continue
        
        # Mann-Whitney U test (non-parametric) - more robust for EEG data
        try:
            u_stat, u_pval = mannwhitneyu(easy_values, hard_values, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(easy_values) - 1) * easy_values.var() + 
                                 (len(hard_values) - 1) * hard_values.var()) / 
                                (len(easy_values) + len(hard_values) - 2))
            cohens_d = (hard_values.mean() - easy_values.mean()) / pooled_std if pooled_std > 0 else 0
            
            if u_pval < 0.05:
                significant_features.append((feature, u_pval, abs(cohens_d)))
                
        except Exception as e:
            continue
    
    # Sort by effect size
    significant_features.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n🎯 Most discriminative channels (p < 0.05):")
    print(f"Found {len(significant_features)} significant channels out of {len(alpha_features)} tested")
    
    for i, (feature, pval, effect_size) in enumerate(significant_features[:top_n]):
        electrode = feature.replace('pow_alpha_', '')
        easy_mean = easy_data[feature].mean()
        hard_mean = hard_data[feature].mean()
        direction = "↑ Hard" if hard_mean > easy_mean else "↓ Hard"
        print(f"  {i+1:2d}. {electrode:>4}: p={pval:.6f}, |d|={effect_size:.3f} {direction}")
    
    # Analyze by electrode groups
    print(f"\n📊 Channel Group Analysis:")
    a_channels = [f for f in significant_features if 'A' in f[0]]
    b_channels = [f for f in significant_features if 'B' in f[0]]
    
    print(f"  A channels (A1-A32): {len(a_channels)} significant")
    print(f"  B channels (B1-B32): {len(b_channels)} significant")
    
    return [f[0] for f in significant_features]

def visualize_64_channel_topography(df, significant_features, top_n=24):
    """Create visualization of 64-channel alpha power patterns."""
    print(f"\n📈 Creating 64-channel alpha power visualization...")
    
    if not significant_features:
        print("No significant features to visualize.")
        return
    
    # Get top features
    features_to_plot = significant_features[:top_n]
    
    # Calculate effect sizes for visualization
    easy_data = df[df[TARGET_COLUMN] == 'easy']
    hard_data = df[df[TARGET_COLUMN] == 'hard']
    
    effect_data = []
    for feature in features_to_plot:
        easy_mean = easy_data[feature].mean()
        hard_mean = hard_data[feature].mean()
        effect_size = (hard_mean - easy_mean) / easy_data[feature].std() if easy_data[feature].std() > 0 else 0
        electrode = feature.replace('pow_alpha_', '')
        effect_data.append((electrode, effect_size, easy_mean, hard_mean))
    
    # Create comprehensive visualization
    electrodes, effects, easy_means, hard_means = zip(*effect_data)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Effect size plot
    ax1 = plt.subplot(2, 2, 1)
    colors = ['red' if e > 0 else 'blue' for e in effects]
    bars1 = ax1.bar(range(len(electrodes)), effects, color=colors, alpha=0.7)
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Effect Size (Hard - Easy)')
    ax1.set_title('Alpha Power: Effect Size by Channel\n(Red: Higher in Hard, Blue: Higher in Easy)')
    ax1.set_xticks(range(len(electrodes)))
    ax1.set_xticklabels(electrodes, rotation=45, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Mean values comparison
    ax2 = plt.subplot(2, 2, 2)
    x = np.arange(len(electrodes))
    width = 0.35
    ax2.bar(x - width/2, easy_means, width, label='Easy', alpha=0.7, color='lightblue')
    ax2.bar(x + width/2, hard_means, width, label='Hard', alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Alpha Power')
    ax2.set_title('Alpha Power: Mean Values by Condition')
    ax2.set_xticks(x)
    ax2.set_xticklabels(electrodes, rotation=45, fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Channel group analysis
    ax3 = plt.subplot(2, 2, 3)
    a_channels = [e for e in electrodes if e.startswith('A')]
    b_channels = [e for e in electrodes if e.startswith('B')]
    
    groups = ['A Channels', 'B Channels']
    counts = [len(a_channels), len(b_channels)]
    colors_group = ['lightgreen', 'lightcoral']
    
    ax3.bar(groups, counts, color=colors_group, alpha=0.7)
    ax3.set_ylabel('Number of Significant Channels')
    ax3.set_title('Significant Channels by Group')
    ax3.grid(True, alpha=0.3)
    
    # Add text annotations
    for i, count in enumerate(counts):
        ax3.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Distribution of effect sizes
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(effects, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Effect Size')
    ax4.set_ylabel('Number of Channels')
    ax4.set_title('Distribution of Effect Sizes')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# FEATURE SELECTION
# ============================================================================

def perform_feature_selection(X, y, feature_names, method='statistical'):
    """Perform feature selection for 64-channel data."""
    print(f"\n🎯 Feature Selection for 64 Channels ({method})")
    print("=" * 50)
    
    n_features = min(N_FEATURE_SELECTION, len(feature_names))
    
    if method == 'statistical':
        # Statistical feature selection (ANOVA F-test)
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
        scores = selector.scores_[selector.get_support()]
        
        print(f"Selected {len(selected_features)} most discriminative channels:")
        
        # Group by A/B channels
        a_selected = [(f, s) for f, s in zip(selected_features, scores) if 'A' in f]
        b_selected = [(f, s) for f, s in zip(selected_features, scores) if 'B' in f]
        
        print(f"\nA Channels ({len(a_selected)}):")
        for feature, score in sorted(a_selected, key=lambda x: x[1], reverse=True):
            electrode = feature.replace('pow_alpha_', '')
            print(f"  {electrode:>4}: F-score = {score:.3f}")
        
        print(f"\nB Channels ({len(b_selected)}):")
        for feature, score in sorted(b_selected, key=lambda x: x[1], reverse=True):
            electrode = feature.replace('pow_alpha_', '')
            print(f"  {electrode:>4}: F-score = {score:.3f}")
    
    elif method == 'rfe':
        # Recursive Feature Elimination with Logistic Regression
        estimator = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        selector = RFE(estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
        
        print(f"Selected {len(selected_features)} channels using RFE:")
        for feature in selected_features:
            electrode = feature.replace('pow_alpha_', '')
            print(f"  {electrode}")
    
    else:
        # Return all features
        X_selected = X
        selected_features = feature_names
    
    return X_selected, selected_features

# ============================================================================
# CLASSIFICATION
# ============================================================================

def alpha_power_64_channel_classification(df, alpha_features, use_feature_selection=True):
    """Perform classification using 64-channel alpha power features."""
    print(f"\n🔬 64-CHANNEL ALPHA POWER CLASSIFICATION")
    print("=" * 60)
    
    # Check available features
    available_features = [f for f in alpha_features if f in df.columns]
    missing_features = [f for f in alpha_features if f not in df.columns]
    
    if not available_features:
        print(f"❌ No alpha power features available")
        return {}
    
    if missing_features:
        print(f"⚠️  Missing {len(missing_features)} channels out of 64")
        missing_electrodes = [f.replace('pow_alpha_', '') for f in missing_features[:10]]
        print(f"   Missing electrodes: {', '.join(missing_electrodes)}...")
    
    print(f"Using {len(available_features)} out of 64 alpha power channels")
    
    # Prepare data
    X = df[available_features].values
    y = df[TARGET_COLUMN].values
    
    # Handle any NaN or infinite values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Encode labels
    y_encoded = np.where(y == 'hard', 1, 0)  # hard=1, easy=0
    
    print(f"Total samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Class distribution: Easy={np.sum(y_encoded==0)}, Hard={np.sum(y_encoded==1)}")
    
    # Check data quality
    print(f"Data quality check:")
    print(f"  Mean values range: {X.mean(axis=0).min():.6f} to {X.mean(axis=0).max():.6f}")
    print(f"  Std values range: {X.std(axis=0).min():.6f} to {X.std(axis=0).max():.6f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
        stratify=y_encoded
    )
    
    # Scale features (use RobustScaler for better handling of outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection
    if use_feature_selection and len(available_features) > N_FEATURE_SELECTION:
        X_train_selected, selected_features = perform_feature_selection(
            X_train_scaled, y_train, available_features, method='statistical'
        )
        
        # Apply same selection to test set
        selector = SelectKBest(score_func=f_classif, k=len(selected_features))
        selector.fit(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        X_train_final, X_test_final = X_train_selected, X_test_selected
        final_features = selected_features
    else:
        X_train_final, X_test_final = X_train_scaled, X_test_scaled
        final_features = available_features
    
    print(f"Final feature count: {len(final_features)}")
    
    # Define models optimized for EEG data
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE, 
            max_iter=2000,
            C=1.0,
            penalty='l2'
        ),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(
            solver='svd'  # More stable for high-dimensional data
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', 
            random_state=RANDOM_STATE, 
            probability=True, 
            gamma='scale',
            C=1.0
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, 
            random_state=RANDOM_STATE,
            max_depth=10,
            min_samples_split=5
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, 
            random_state=RANDOM_STATE,
            learning_rate=0.1,
            max_depth=6
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        try:
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_final, y_train, 
                cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                scoring='accuracy'
            )
            print(f"CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            
            # Train and test
            model.fit(X_train_final, y_train)
            y_pred = model.predict(X_test_final)
            y_pred_proba = model.predict_proba(X_test_final)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            test_acc = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            print(f"Test Accuracy: {test_acc:.3f}")
            if auc_score:
                print(f"AUC Score: {auc_score:.3f}")
            
            results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'test_accuracy': test_acc,
                'auc_score': auc_score,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'scaler': scaler,
                'features': final_features,
                'y_test': y_test
            }
            
        except Exception as e:
            print(f"Error with {name}: {e}")
            continue
    
    if not results:
        return {}
    
    # Best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"Test Accuracy: {best_result['test_accuracy']:.3f}")
    if best_result['auc_score']:
        print(f"AUC Score: {best_result['auc_score']:.3f}")
    
    # Detailed results for best model
    print(f"\nClassification Report ({best_model_name}):")
    print(classification_report(y_test, best_result['y_pred'], 
                              target_names=['Easy', 'Hard']))
    
    return results

def analyze_64_channel_feature_importance(results, top_n=20):
    """Analyze feature importance for 64-channel alpha power."""
    print(f"\n🔍 64-CHANNEL FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    importance_summary = {}
    
    for model_name, result in results.items():
        model = result['model']
        features = result['features']
        
        print(f"\n--- {model_name} ---")
        
        if hasattr(model, 'coef_') and len(model.coef_[0]) == len(features):
            # Linear models - coefficients
            coefs = model.coef_[0]
            feature_importance = list(zip(features, coefs, np.abs(coefs)))
            feature_importance.sort(key=lambda x: x[2], reverse=True)
            
            print("Top discriminative channels (coefficients):")
            
            # Show top channels
            for i, (feature, coef, abs_coef) in enumerate(feature_importance[:top_n]):
                electrode = feature.replace('pow_alpha_', '')
                direction = "→ Hard" if coef > 0 else "→ Easy"
                print(f"  {i+1:2d}. {electrode:>4}: {coef:>8.4f} {direction}")
            
            # Group analysis
            a_channels = [(f, c, ac) for f, c, ac in feature_importance if 'A' in f]
            b_channels = [(f, c, ac) for f, c, ac in feature_importance if 'B' in f]
            
            print(f"\nChannel group summary:")
            print(f"  A channels in top {top_n}: {len([f for f, c, ac in feature_importance[:top_n] if 'A' in f])}")
            print(f"  B channels in top {top_n}: {len([f for f, c, ac in feature_importance[:top_n] if 'B' in f])}")
                
            importance_summary[model_name] = feature_importance
            
        elif hasattr(model, 'feature_importances_') and len(model.feature_importances_) == len(features):
            # Tree-based models - feature importance
            importances = model.feature_importances_
            feature_importance = list(zip(features, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("Top discriminative channels (importance):")
            for i, (feature, importance) in enumerate(feature_importance[:top_n]):
                electrode = feature.replace('pow_alpha_', '')
                print(f"  {i+1:2d}. {electrode:>4}: {importance:>8.4f}")
            
            # Group analysis
            print(f"\nChannel group summary:")
            print(f"  A channels in top {top_n}: {len([f for f, i in feature_importance[:top_n] if 'A' in f])}")
            print(f"  B channels in top {top_n}: {len([f for f, i in feature_importance[:top_n] if 'B' in f])}")
                
            importance_summary[model_name] = feature_importance
    
    return importance_summary

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("🧠 EEG 64-Channel Alpha Power Classification")
    print("📡 Channels: A1-A32, B1-B32 (Total: 64)")
    print("=" * 70)
    
    # Check if feature file exists
    if not Path(FEATURE_CSV).exists():
        print(f"❌ Error: Feature file '{FEATURE_CSV}' not found!")
        print("Please run the enhanced feature extraction script first.")
        return
    
    # Load and explore data
    df, available_features = load_and_explore_data(FEATURE_CSV)
    
    # Get available alpha features
    alpha_features = available_features["Alpha Power Features (64 channels)"]
    
    if not alpha_features:
        print("❌ No alpha power features found in the dataset!")
        print("Expected features: pow_alpha_A1, pow_alpha_A2, ..., pow_alpha_B32")
        return
    
    print(f"\n✅ Found {len(alpha_features)} out of 64 expected alpha power channels")
    
    # Statistical analysis of alpha features
    significant_alpha_features = analyze_alpha_feature_statistics(df, alpha_features)
    
    # Visualize 64-channel patterns
    if significant_alpha_features:
        visualize_64_channel_topography(df, significant_alpha_features)
    
    # Perform classification with 64-channel alpha features
    results = alpha_power_64_channel_classification(df, alpha_features, use_feature_selection=True)
    
    # Analyze feature importance
    if results:
        print(f"\n🎯 64-channel alpha power feature importance analysis")
        importance_analysis = analyze_64_channel_feature_importance(results)
    
    print("\n✅ 64-channel alpha power classification analysis complete!")
    
    if results:
        best_model = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        best_acc = results[best_model]['test_accuracy']
        best_auc = results[best_model].get('auc_score', 0.0)
        
        print("\n💡 Key Results:")
        print(f"  • Best model: {best_model}")
        print(f"  • Best accuracy: {best_acc:.3f}")
        if best_auc:
            print(f"  • Best AUC: {best_auc:.3f}")
        print(f"  • Channels used: {len(results[best_model]['features'])}")
        print(f"  • Significant channels: {len(significant_alpha_features)}")
        print(f"  • Total channels analyzed: {len(alpha_features)}")
    
    print("\n📈 64-Channel Alpha Power Analysis Summary:")
    print("  • Using full 64-channel coverage (A1-A32, B1-B32)")
    print("  • Alpha power patterns across all electrode positions")
    print("  • Spatial distribution analysis for cognitive load detection")
    print("  • Feature selection identifies most discriminative channels")

if __name__ == "__main__":
    main()