"""
EEG MATB Task Difficulty Classifier
===================================
Simple ML classifier for easy/hard MATB task classification using single EEG features.
Supports both epoch-level and file-level (participant-level) classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input data
FEATURE_CSV = "features_matb_15s_3rd_session.csv"  # Output from enhanced feature extraction script
FEATURE_COLUMNS = ["avg_power","spectral_centroid","peak_freq","pow_delta","log_pow_delta","pow_theta","log_pow_theta","pow_alpha","log_pow_alpha","pow_beta","log_pow_beta","pow_gamma","log_pow_gamma"]
TARGET_COLUMN = "difficulty"  # 'easy' or 'hard'

# Model settings
TEST_SIZE = 0.2
RANDOM_STATE = 1
CV_FOLDS = 5

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
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Basic statistics
    print(f"\nClass distribution:")
    print(df[TARGET_COLUMN].value_counts())
    print(f"\nClass proportions:")
    print(df[TARGET_COLUMN].value_counts(normalize=True))
    
    # Feature statistics by class
    print(f"\nFeature statistics by difficulty:")
    for feature in FEATURE_COLUMNS:
        if feature in df.columns:
            print(f"\n{feature}:")
            feature_stats = df.groupby(TARGET_COLUMN)[feature].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(6)
            print(feature_stats)
    
    # Files per participant
    print(f"\nNumber of files: {df['file_name'].nunique()}")
    print(f"Epochs per file:")
    epochs_per_file = df.groupby('file_name').size()
    print(f"  Mean: {epochs_per_file.mean():.1f}")
    print(f"  Min: {epochs_per_file.min()}")
    print(f"  Max: {epochs_per_file.max()}")
    
    return df

def visualize_feature_distribution(df):
    """Create visualizations of feature distributions."""
    print("\n📈 Creating feature distribution plots...")
    
    n_features = len(FEATURE_COLUMNS)
    n_cols = min(2, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(6 * n_cols, 5 * n_rows * 2))
    if n_features == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('EEG Feature Distributions by Task Difficulty', fontsize=16)
    
    for i, feature in enumerate(FEATURE_COLUMNS):
        if feature not in df.columns:
            continue
            
        row = (i // n_cols) * 2
        col = i % n_cols
        
        # Histogram
        for difficulty in ['easy', 'hard']:
            data = df[df[TARGET_COLUMN] == difficulty][feature]
            axes[row, col].hist(data, alpha=0.7, label=difficulty, bins=30)
        axes[row, col].set_xlabel(f'{feature.title().replace("_", "/")} Power')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].set_title(f'{feature.title().replace("_", "/")} Distribution')
        axes[row, col].legend()
        
        # Box plot
        df.boxplot(column=feature, by=TARGET_COLUMN, ax=axes[row + 1, col])
        axes[row + 1, col].set_title(f'{feature.title().replace("_", "/")} Box Plot')
        axes[row + 1, col].set_xlabel('Task Difficulty')
    
    # Hide empty subplots
    total_plots = n_rows * 2 * n_cols
    used_plots = len(FEATURE_COLUMNS) * 2
    for i in range(used_plots, total_plots):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# EPOCH-LEVEL CLASSIFICATION
# ============================================================================

def epoch_level_classification(df):
    """Perform epoch-level classification (each epoch is a sample)."""
    print("\n🔬 EPOCH-LEVEL CLASSIFICATION")
    print("=" * 50)
    
    # Prepare data
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    # Check for missing features
    missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        print(f"Available columns: {list(df.columns)}")
        return {}, None, None
    
    # Encode labels
    y_encoded = np.where(y == 'hard', 1, 0)  # hard=1, easy=0
    
    print(f"Total samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y_encoded)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
        stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'SVM (RBF)': SVC(kernel='rbf', random_state=RANDOM_STATE, probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE))
        print(f"CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Train and test
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
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
            'scaler': scaler
        }
    
    # Best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"Test Accuracy: {best_result['test_accuracy']:.3f}")
    
    # Detailed results for best model
    print(f"\nDetailed Classification Report ({best_model_name}):")
    print(classification_report(y_test, best_result['y_pred'], 
                              target_names=['Easy', 'Hard']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_result['y_pred'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Easy', 'Hard'], yticklabels=['Easy', 'Hard'])
    plt.title(f'Confusion Matrix - {best_model_name} (Epoch Level)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return results, X_test_scaled, y_test

# ============================================================================
# FILE-LEVEL CLASSIFICATION
# ============================================================================

def file_level_classification(df):
    """Perform file-level classification (each file/participant is a sample)."""
    print("\n📁 FILE-LEVEL CLASSIFICATION")
    print("=" * 50)
    
    # Aggregate features by file and difficulty
    # For each file, we'll compute mean feature values for easy and hard epochs
    file_features = []
    
    for file_name in df['file_name'].unique():
        file_data = df[df['file_name'] == file_name]
        
        for difficulty in ['easy', 'hard']:
            diff_data = file_data[file_data[TARGET_COLUMN] == difficulty]
            if len(diff_data) > 0:
                # Aggregate features (mean across epochs)
                feature_dict = {
                    'file_name': file_name,
                    'difficulty': difficulty,
                    'n_epochs': len(diff_data)
                }
                
                # Add mean and std for each feature
                for feature in FEATURE_COLUMNS:
                    if feature in diff_data.columns:
                        feature_dict[f'{feature}_mean'] = diff_data[feature].mean()
                        feature_dict[f'{feature}_std'] = diff_data[feature].std()
                
                file_features.append(feature_dict)
    
    file_df = pd.DataFrame(file_features)
    print(f"File-level dataset shape: {file_df.shape}")
    print(f"Files per condition:")
    print(file_df['difficulty'].value_counts())
    
    # Prepare data - use mean features
    feature_cols = [f'{f}_mean' for f in FEATURE_COLUMNS]
    missing_cols = [col for col in feature_cols if col not in file_df.columns]
    if missing_cols:
        print(f"❌ Missing feature columns: {missing_cols}")
        return {}, file_df
    
    X = file_df[feature_cols].values
    y = file_df['difficulty'].values
    y_encoded = np.where(y == 'hard', 1, 0)
    
    # Check if we have enough samples for train/test split
    if len(X) < 10:
        print("⚠️  Warning: Very few file-level samples. Using cross-validation only.")
        return perform_cv_only_classification(X, y_encoded, file_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
        stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models (simpler for small datasets)
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Cross-validation
        if len(np.unique(y_encoded)) > 1:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                       cv=min(CV_FOLDS, len(X_train)//2))
            print(f"CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Train and test
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        test_acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {test_acc:.3f}")
        
        results[name] = {
            'model': model,
            'test_accuracy': test_acc,
            'y_pred': y_pred,
            'scaler': scaler
        }
    
    return results, file_df

def perform_cv_only_classification(X, y_encoded, file_df):
    """Perform classification using only cross-validation for small datasets."""
    print("Using cross-validation only due to small sample size...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis()
    }
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        cv_scores = cross_val_score(model, X_scaled, y_encoded, 
                                   cv=min(3, len(X)//2))  # Smaller CV folds
        print(f"CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    
    return {}, file_df

# ============================================================================
# FEATURE IMPORTANCE AND INTERPRETATION
# ============================================================================

def analyze_feature_importance(results, feature_names):
    """Analyze and visualize feature importance/coefficients."""
    print(f"\n🔍 FEATURE ANALYSIS")
    print("=" * 50)
    
    for name, result in results.items():
        model = result['model']
        print(f"\n--- {name} ---")
        
        if hasattr(model, 'coef_'):
            coefs = model.coef_[0] if len(model.coef_[0]) > 1 else [model.coef_[0][0]]
            print("Feature Coefficients:")
            for feature, coef in zip(feature_names, coefs):
                direction = "↑ Higher" if coef > 0 else "↓ Lower"
                print(f"  {feature:>20}: {coef:>8.6f} ({direction} → Higher difficulty)")
                
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            print("Feature Importances:")
            # Sort features by importance
            importance_pairs = list(zip(feature_names, importances))
            importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            for feature, importance in importance_pairs:
                print(f"  {feature:>20}: {importance:>8.6f}")
                
        elif hasattr(model, 'scalings_'):  # LDA
            scalings = model.scalings_[:, 0] if model.scalings_.shape[1] > 0 else model.scalings_
            print("LDA Scalings (discriminant weights):")
            for feature, scaling in zip(feature_names, scalings):
                direction = "↑ Higher" if scaling > 0 else "↓ Lower"
                print(f"  {feature:>20}: {scaling:>8.6f} ({direction} → Higher difficulty)")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("🧠 EEG MATB Task Difficulty Classification")
    print("=" * 60)
    
    # Check if feature file exists
    if not Path(FEATURE_CSV).exists():
        print(f"❌ Error: Feature file '{FEATURE_CSV}' not found!")
        print("Please run the enhanced feature extraction script first.")
        print("Available features will be: theta, alpha, theta_alpha_ratio")
        return
    
    print(f"Using features: {FEATURE_COLUMNS}")
    
    # Load and explore data
    df = load_and_explore_data(FEATURE_CSV)
    
    # Visualize distributions
    visualize_feature_distribution(df)
    
    # Epoch-level classification
    epoch_results, X_test, y_test = epoch_level_classification(df)
    
    # File-level classification
    file_results, file_df = file_level_classification(df)
    
    # Feature importance analysis
    if epoch_results:
        analyze_feature_importance(epoch_results, FEATURE_COLUMNS)
    
    print("\n✅ Classification analysis complete!")
    print("\n💡 Next steps:")
    print("1. Try different feature combinations by changing FEATURE_COLUMNS:")
    print("   - Single features: ['alpha'], ['theta'], ['theta_alpha_ratio']")
    print("   - Multiple features: ['alpha', 'theta'], ['alpha', 'theta_alpha_ratio']")
    print("   - All features: ['alpha', 'theta', 'theta_alpha_ratio']")
    print("2. Experiment with different preprocessing parameters")
    print("3. Consider participant-specific normalization")
    print("4. Add more frequency bands (beta, gamma) or connectivity features")

if __name__ == "__main__":
    main()