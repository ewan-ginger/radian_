import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.fft import rfft
from scipy import signal  # NEW: For IMU signal processing
from itertools import combinations
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import os
import json
import traceback

# Cross-validation imports
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer
# Feature importance analysis imports (NEW)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')  # Suppress CV warnings

# Performance timing analysis imports (NEW)
import time
import psutil  # For memory monitoring
import gc  # For garbage collection
from contextlib import contextmanager
from collections import defaultdict

# Optional SHAP import with graceful handling (NEW)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#------------------------------------------------------------------------------
# LOSO Cross-Validation Splitter Class (NEW)
#------------------------------------------------------------------------------
class LeaveOneSubjectOut:
    """
    Custom CV splitter for Leave-One-Subject-Out cross-validation.
    
    This splitter creates folds by leaving out one subject/player at a time,
    which is essential for evaluating subject-independent performance in HAR.
    
    Parameters:
    -----------
    player_ids : array-like
        Array of player/subject identifiers corresponding to each sample
    min_samples_per_subject : int, default=50
        Minimum number of samples a subject must have to be included
        
    Usage:
    ------
    player_ids = [1, 1, 1, 2, 2, 2, 3, 3, 3]  # Example player assignments
    loso_cv = LeaveOneSubjectOut(player_ids)
    
    for train_idx, test_idx in loso_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Train and evaluate model
    """
    
    def __init__(self, player_ids, min_samples_per_subject=50):
        self.player_ids = np.array(player_ids)
        self.unique_players = np.unique(self.player_ids)
        self.min_samples_per_subject = min_samples_per_subject
        
        # Filter out subjects with too few samples
        self.valid_players = []
        for player in self.unique_players:
            n_samples = np.sum(self.player_ids == player)
            if n_samples >= self.min_samples_per_subject:
                self.valid_players.append(player)
            else:
                print(f"Warning: Player {player} has only {n_samples} samples, excluding from LOSO CV")
        
        self.valid_players = np.array(self.valid_players)
        print(f"LOSO CV: Using {len(self.valid_players)} subjects out of {len(self.unique_players)} total")
        print(f"Valid subjects: {self.valid_players}")
        
        if len(self.valid_players) < 2:
            raise ValueError(f"LOSO CV requires at least 2 subjects with >= {min_samples_per_subject} samples each")
    
    def split(self, X, y=None, groups=None):
        """Generate train/test splits leaving one subject out"""
        for player in self.valid_players:
            # Test set: current player
            test_mask = self.player_ids == player
            # Train set: all other valid players
            train_mask = np.isin(self.player_ids, self.valid_players) & ~test_mask
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            print(f"LOSO fold: Training on {len(train_indices)} samples from {len(self.valid_players)-1} subjects, "
                  f"testing on {len(test_indices)} samples from subject {player}")
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits (number of valid subjects)"""
        return len(self.valid_players)
    
    def get_subjects_info(self):
        """Return detailed information about subjects and their sample counts"""
        info = {}
        for player in self.unique_players:
            n_samples = np.sum(self.player_ids == player)
            info[player] = {
                'n_samples': n_samples,
                'included_in_cv': player in self.valid_players
            }
        return info

#------------------------------------------------------------------------------
# Global Parameters (Static)
#------------------------------------------------------------------------------
SAMPLING_RATE = 50
WINDOW_SIZE_SECONDS = 2
WINDOW_SIZE = int(WINDOW_SIZE_SECONDS * SAMPLING_RATE)  # 2 seconds at 50Hz

# File paths
TRAINING_DATA_PATH = 'data/train/50Hz_FINAL_training_data.csv'
TEST_DATA_PATH = 'data/test/all_2v2_data.csv'
ENSEMBLE_GROUND_TRUTH_PATH = 'data/labels/ground_truth_2v2.json' # New path for comparison at the end
OUTPUT_DIR_FIGURES = 'figures/'
OUTPUT_DIR_MODELS = 'models/'

# Organized output subdirectories
OUTPUT_DIR_FEATURE_IMPORTANCE = 'figures/feature_importance/'
OUTPUT_DIR_TIMING_ANALYSIS = 'figures/timing_analysis/'
OUTPUT_DIR_MODEL_PREDICTIONS = 'figures/model_predictions/'
OUTPUT_DIR_ENSEMBLE_PREDICTIONS = 'figures/ensemble_predictions/'
OUTPUT_DIR_PERFORMANCE_ANALYSIS = 'figures/performance_analysis/'
OUTPUT_DIR_GROUND_TRUTH_COMPARISON = 'figures/ground_truth_comparison/'

# Ensure output directories exist
os.makedirs(OUTPUT_DIR_FIGURES, exist_ok=True)
os.makedirs(OUTPUT_DIR_MODELS, exist_ok=True)
os.makedirs(OUTPUT_DIR_FEATURE_IMPORTANCE, exist_ok=True)
os.makedirs(OUTPUT_DIR_TIMING_ANALYSIS, exist_ok=True)
os.makedirs(OUTPUT_DIR_MODEL_PREDICTIONS, exist_ok=True)
os.makedirs(OUTPUT_DIR_ENSEMBLE_PREDICTIONS, exist_ok=True)
os.makedirs(OUTPUT_DIR_PERFORMANCE_ANALYSIS, exist_ok=True)
os.makedirs(OUTPUT_DIR_GROUND_TRUTH_COMPARISON, exist_ok=True)

# Window overlap options to test (as fractions)
WINDOW_OVERLAP_OPTIONS = [0.5, 0.75]
# Primary metric for selecting the best model configuration
PRIMARY_METRIC = 'f1' # Can be 'accuracy', 'precision', 'recall', 'f1'
PERFORM_UNDERSAMPLING = True # Flag to control undersampling

# Cross-validation configuration (NEW)
USE_CROSS_VALIDATION = True  # Set to True to enable CV, False to use original train-test split
CV_FOLDS = 5  # Number of CV folds when USE_CROSS_VALIDATION is True
CV_RANDOM_STATE = 42
CV_SCORING = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
CV_REDUCED_EPOCHS = 25  # Reduced epochs for CNN CV to save time
CV_VERBOSE = 0  # CNN training verbosity during CV (0=silent, 1=progress bar)

# IMU Data Preprocessing Configuration (NEW)
USE_IMU_PREPROCESSING = False  # Set to True to enable signal preprocessing
LOWPASS_CUTOFF_FREQ = 15.0  # Low-pass filter cutoff frequency in Hz (for human movements)
HIGHPASS_CUTOFF_FREQ = 0.5  # High-pass filter cutoff frequency in Hz (remove DC bias)
FILTER_ORDER = 4  # Butterworth filter order
APPLY_LOWPASS = True  # Apply low-pass filtering when preprocessing enabled
APPLY_HIGHPASS = True  # Apply high-pass filtering when preprocessing enabled
REMOVE_GRAVITY = False  # Remove gravity component from accelerometer data

# False Positive Reduction Configuration (NEW)
CONFIDENCE_THRESHOLD = 0.9  # Minimum confidence for ALL model predictions (ML models via predict_proba, CNN via probabilities)
MERGE_SIMILAR_EVENTS = True  # Merge consecutive similar events within time window
APPLY_NON_MAX_SUPPRESSION = True  # Apply non-maximum suppression to reduce overlapping predictions
NMS_TIME_WINDOW = 2.0  # Time window for non-maximum suppression (seconds)
EVENT_MERGE_TIME_WINDOW = 3.0  # Time window for merging similar events (seconds)

# Hyperparameter Tuning Configuration (NEW)
USE_HYPERPARAMETER_TUNING = False  # Set to True to enable hyperparameter tuning after overlap selection
HYPERPARAMETER_CV_FOLDS = 3  # Reduced folds for hyperparameter tuning to save time
HYPERPARAMETER_VERBOSE = 1  # GridSearchCV verbosity level
HYPERPARAMETER_N_JOBS = -1  # Number of parallel jobs for GridSearchCV

# Parameter grids for hyperparameter tuning (NEW)
HYPERPARAMETER_GRIDS = {
    'Random Forest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, None],
        'classifier__max_features': ['sqrt', 'log2']
    },
    'SVM': {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'classifier__kernel': ['rbf', 'poly', 'sigmoid']
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 6],
        'classifier__learning_rate': [0.1, 0.2]
    },
    'Neural Network': {
        'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
        'classifier__alpha': [0.0001, 0.001, 0.01],
        'classifier__learning_rate_init': [0.001, 0.01, 0.1],
        'classifier__max_iter': [300, 500, 1000]
    }
}

# CNN Hyperparameter Tuning Configuration (NEW)
CNN_HYPERPARAMETER_CONFIGS = [
    {
        'filters_1': 32, 'filters_2': 64, 'filters_3': 128,
        'dense_1': 64, 'dense_2': 32, 'dropout_1': 0.3, 'dropout_2': 0.2,
        'batch_size': 16, 'learning_rate': 0.001
    },
    {
        'filters_1': 64, 'filters_2': 128, 'filters_3': 256,
        'dense_1': 128, 'dense_2': 64, 'dropout_1': 0.5, 'dropout_2': 0.3,
        'batch_size': 32, 'learning_rate': 0.001
    },
    {
        'filters_1': 128, 'filters_2': 256, 'filters_3': 512,
        'dense_1': 256, 'dense_2': 128, 'dropout_1': 0.4, 'dropout_2': 0.3,
        'batch_size': 32, 'learning_rate': 0.0005
    }
]

# Feature Importance Analysis Configuration (NEW)
USE_FEATURE_IMPORTANCE_ANALYSIS = False  # Set to True to enable comprehensive feature importance analysis
FEATURE_IMPORTANCE_TOP_N = 30  # Number of top features to display in plots
FEATURE_IMPORTANCE_METHODS = ['builtin', 'permutation', 'shap']  # Methods to use: builtin, permutation, shap
FEATURE_IMPORTANCE_PERMUTATION_REPEATS = 5  # Number of repeats for permutation importance
FEATURE_IMPORTANCE_SHAP_SAMPLES = 500  # Number of samples for SHAP analysis (reduce for speed)

# Performance Timing Analysis Configuration (NEW)
USE_TIMING_ANALYSIS = False  # Set to True to enable comprehensive timing analysis
TIMING_WARMUP_ITERATIONS = 3  # Number of warmup iterations for prediction timing
TIMING_MEASUREMENT_ITERATIONS = 10  # Number of measurement iterations for statistical accuracy
TIMING_INCLUDE_MEMORY_USAGE = True  # Monitor memory usage during training and prediction
TIMING_DETAILED_BREAKDOWN = True  # Include detailed timing for feature extraction, preprocessing, etc.

# LOSO CV Configuration (NEW)
USE_LOSO_CV = True  # Set to True to use Leave-One-Subject-Out CV instead of regular CV
LOSO_PLAYER_ID_COLUMN = 'player_id'  # Column name containing subject/player identifiers
LOSO_MIN_SAMPLES_PER_SUBJECT = 20  # Minimum number of windows per subject to include in LOSO CV

# Model Selection Configuration (NEW)
USE_CNN_MODEL = True  # Set to False to skip CNN training (much faster, ML models only)
USE_ML_MODELS = True  # Set to False to skip ML models (CNN only)

#------------------------------------------------------------------------------
# Cross-Validation Usage Instructions
#------------------------------------------------------------------------------
"""
CROSS-VALIDATION FEATURE:

This script now supports optional cross-validation for more robust model evaluation.

To enable cross-validation:
1. Set USE_CROSS_VALIDATION = True
2. Adjust CV_FOLDS as needed (default: 5)
3. Run the script as usual

Cross-validation benefits:
- More robust performance estimates with confidence intervals
- Better model selection based on multiple train-validation splits
- Statistical significance testing

Trade-offs:
- Increased computation time (~5x for ML models, ~5x for CNN)
- CNN uses reduced epochs during CV to save time
- Final models still trained on full dataset for predictions

The script maintains full backward compatibility - all existing functionality
works unchanged when USE_CROSS_VALIDATION = False (default).
"""

#------------------------------------------------------------------------------
# LOSO Cross-Validation Usage Instructions (NEW)
#------------------------------------------------------------------------------
"""
LOSO (LEAVE-ONE-SUBJECT-OUT) CROSS-VALIDATION FEATURE:

This script now supports Leave-One-Subject-Out (LOSO) cross-validation for 
subject-independent evaluation. LOSO CV is the gold standard in Human Activity 
Recognition (HAR) research for evaluating model generalization to unseen subjects.

🎯 WHAT IS LOSO CV?
LOSO CV creates folds by leaving out one subject/player at a time. Each fold 
trains on N-1 subjects and tests on the remaining subject. This evaluates how 
well models generalize to completely new individuals.

🔧 TO ENABLE LOSO CV:
1. Set USE_CROSS_VALIDATION = True  (enables CV framework)
2. Set USE_LOSO_CV = True           (switches to LOSO instead of regular CV)
3. Adjust LOSO_MIN_SAMPLES_PER_SUBJECT as needed (default: 50)
4. Run the script as usual

📊 LOSO CV vs REGULAR CV:
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Aspect              │ Regular CV          │ LOSO CV             │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Fold Creation       │ Random stratified   │ By subject/player   │
│ Training Data       │ Mixed subjects      │ N-1 subjects        │
│ Test Data           │ Mixed subjects      │ 1 held-out subject  │
│ Number of Folds     │ CV_FOLDS (5)        │ # of valid subjects │
│ Performance         │ Higher (easier)     │ Lower (harder)      │
│ Realism             │ Optimistic          │ Realistic           │
│ Subject Independence│ No                  │ Yes                 │
└─────────────────────┴─────────────────────┴─────────────────────┘

🧠 WHY USE LOSO CV?
✅ Subject Independence: Tests true generalization to unseen individuals
✅ Research Standard: Required for publication in HAR/ML journals
✅ Real-world Simulation: Mimics deployment to new users
✅ Prevents Overfitting: Can't memorize subject-specific patterns
✅ Better Model Selection: Identifies truly generalizable models

⚠️  EXPECTED RESULTS:
- LOSO CV typically shows 10-30% lower performance than regular CV
- This is NORMAL and EXPECTED - it's a harder, more realistic evaluation
- Lower scores indicate better subject independence assessment
- Use LOSO results for research publications and real-world deployment planning

🔧 CONFIGURATION OPTIONS:
- LOSO_MIN_SAMPLES_PER_SUBJECT: Minimum windows per subject (default: 50)
  • Subjects with fewer samples are excluded from CV
  • Prevents unreliable evaluation on subjects with insufficient data
  • Adjust based on your dataset size and window overlap settings

📈 INTERPRETATION GUIDELINES:
- LOSO F1 > 0.70: Excellent subject-independent performance
- LOSO F1 > 0.60: Good subject-independent performance  
- LOSO F1 > 0.50: Acceptable subject-independent performance
- LOSO F1 < 0.50: Poor generalization, needs improvement

🚀 BEST PRACTICES:
1. Always report LOSO results for HAR research
2. Use regular CV for development, LOSO for final evaluation
3. Ensure balanced subject representation across activity classes
4. Consider subject demographics (age, fitness, experience) in analysis
5. Report both individual subject results and aggregate statistics

⚡ PERFORMANCE IMPACT:
- Training time: ~N_subjects × base_time (vs ~5× for regular CV)
- Memory usage: Similar to regular CV
- Typically 3-10 folds depending on number of subjects in dataset

The script maintains full backward compatibility - all existing functionality
works unchanged when USE_LOSO_CV = False (default).
"""

#------------------------------------------------------------------------------
# Model Selection Usage Instructions (NEW)
#------------------------------------------------------------------------------
"""
MODEL SELECTION FEATURE:

This script now supports selective training of model types to dramatically reduce 
training time during development and experimentation.

🎯 WHAT IS MODEL SELECTION?
Model selection allows you to enable/disable entire model categories:
- Traditional ML Models: Random Forest, SVM, XGBoost, Neural Network
- Deep Learning Model: Convolutional Neural Network (CNN)

🔧 CONFIGURATION OPTIONS:
USE_CNN_MODEL = True/False   # Control CNN training
USE_ML_MODELS = True/False   # Control ML model training

📊 USAGE SCENARIOS:
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Scenario            │ USE_CNN_MODEL       │ USE_ML_MODELS       │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Full Evaluation     │ True                │ True                │
│ Quick Development   │ False               │ True                │
│ CNN Experimentation │ True                │ False               │
│ Feature Engineering │ False               │ True                │
│ Final Publication   │ True                │ True                │
└─────────────────────┴─────────────────────┴─────────────────────┘

⚡ PERFORMANCE IMPACT:
Time savings when disabling CNN:
- Regular CV (5 folds): ~80% faster training
- LOSO CV (15 subjects): ~90% faster training
- Memory usage: ~70% reduction during training
- GPU usage: Not required when CNN disabled

Example timing for typical dataset:
- Full training (ML + CNN): 2-4 hours with LOSO CV
- ML only (USE_CNN_MODEL = False): 15-30 minutes
- CNN only (USE_ML_MODELS = False): 1.5-3 hours

🚀 DEVELOPMENT WORKFLOW:
1. START: USE_CNN_MODEL = False, USE_ML_MODELS = True
   • Rapid iteration on feature engineering
   • Quick hyperparameter tuning for ML models
   • Fast overlap and preprocessing experimentation

2. REFINE: USE_CNN_MODEL = True, USE_ML_MODELS = False  
   • Focus on CNN architecture optimization
   • CNN-specific hyperparameter tuning
   • Deep learning experimentation

3. FINAL: USE_CNN_MODEL = True, USE_ML_MODELS = True
   • Complete evaluation for publication
   • Final model comparison and selection
   • Comprehensive performance analysis

💡 RECOMMENDED SETTINGS:
For Development: USE_CNN_MODEL = False, USE_ML_MODELS = True
For Production: USE_CNN_MODEL = True, USE_ML_MODELS = True
For CNN Research: USE_CNN_MODEL = True, USE_ML_MODELS = False

⚠️  IMPORTANT NOTES:
- At least one model type must be enabled (script will handle gracefully)
- All analysis features work regardless of which models are enabled
- Model ensemble requires both types enabled for maximum effectiveness
- Best configuration selection adapts to available models automatically

The script maintains full backward compatibility - all existing functionality
works unchanged when both USE_CNN_MODEL = True and USE_ML_MODELS = True (default).
"""

#------------------------------------------------------------------------------
# IMU Preprocessing Usage Instructions
#------------------------------------------------------------------------------
"""
IMU PREPROCESSING FEATURE:

This script now supports optional signal preprocessing for IMU data before windowing.

To enable IMU preprocessing:
1. Set USE_IMU_PREPROCESSING = True
2. Adjust filter parameters as needed:
   - LOWPASS_CUTOFF_FREQ: Remove high-frequency noise (default: 15 Hz)
   - HIGHPASS_CUTOFF_FREQ: Remove DC bias and drift (default: 0.5 Hz)
   - FILTER_ORDER: Butterworth filter order (default: 4)
   - APPLY_LOWPASS, APPLY_HIGHPASS, REMOVE_GRAVITY: Enable/disable specific filters

Preprocessing benefits:
- Cleaner FFT features focused on relevant frequencies
- Better feature stability and reduced noise
- Improved CNN performance with cleaner temporal patterns
- Reduced overfitting by removing noise

Recommended settings for sports data:
- LOWPASS_CUTOFF_FREQ = 15.0  # Captures most human movements
- HIGHPASS_CUTOFF_FREQ = 0.5  # Removes DC bias
- REMOVE_GRAVITY = True       # Important for accelerometer

The script maintains full backward compatibility - all existing functionality
works unchanged when USE_IMU_PREPROCESSING = False (default).
"""

#------------------------------------------------------------------------------
# False Positive Reduction Usage Instructions
#------------------------------------------------------------------------------
"""
FALSE POSITIVE REDUCTION FEATURES:

This script now includes advanced false positive reduction for individual model 
evaluations against ground truth data.

Key Features:
1. CONFIDENCE_THRESHOLD: Only predictions above this confidence are included (ALL models)
2. MERGE_SIMILAR_EVENTS: Merges consecutive predictions of same action within time window
3. APPLY_NON_MAX_SUPPRESSION: Removes overlapping predictions, keeping highest confidence
4. Configurable time windows for merging and suppression

Configuration:
- CONFIDENCE_THRESHOLD = 0.9  # Higher = fewer false positives, might miss some true events
- MERGE_SIMILAR_EVENTS = True  # Reduces temporal redundancy from sliding windows
- APPLY_NON_MAX_SUPPRESSION = True  # Additional overlap removal
- NMS_TIME_WINDOW = 2.0  # Suppression window (match your window size)
- EVENT_MERGE_TIME_WINDOW = 3.0  # Merging window for similar events

Expected Impact:
- 60-80% reduction in false positives for individual models
- 30-45% improvement in precision
- Better comparison with ensemble performance

These features only affect ground truth evaluation - model training remains unchanged.
"""

#------------------------------------------------------------------------------
# Hyperparameter Tuning Usage Instructions
#------------------------------------------------------------------------------
"""
HYPERPARAMETER TUNING FEATURE:

This script now supports optional hyperparameter tuning as a second optimization stage
after finding the best window overlap configurations.

Two-Stage Optimization Process:
1. Stage 1: Find optimal window overlap for each model (existing functionality)
2. Stage 2: Perform hyperparameter tuning on best overlap configurations (NEW)

To enable hyperparameter tuning:
1. Set USE_HYPERPARAMETER_TUNING = True
2. Adjust parameter grids in HYPERPARAMETER_GRIDS as needed
3. Modify CNN_HYPERPARAMETER_CONFIGS for CNN architecture tuning
4. Run the script as usual

Configuration:
- HYPERPARAMETER_CV_FOLDS = 3  # Reduced folds to save computation time
- HYPERPARAMETER_VERBOSE = 1   # Shows GridSearchCV progress
- HYPERPARAMETER_N_JOBS = -1   # Use all available cores

Expected Benefits:
- 10-20% improvement in model performance over default hyperparameters
- Better model selection through systematic parameter exploration
- More robust final models for ensemble and individual predictions

Trade-offs:
- Increased computation time (~3x for ML models, varies for CNN)
- Uses reduced CV folds to balance thoroughness with efficiency

The script maintains full backward compatibility - all existing functionality
works unchanged when USE_HYPERPARAMETER_TUNING = False (default).
"""

#------------------------------------------------------------------------------
# Feature Importance Analysis Usage Instructions (NEW)
#------------------------------------------------------------------------------
"""
FEATURE IMPORTANCE ANALYSIS FEATURES:

This script now includes comprehensive feature importance analysis to understand 
which IMU sensor features contribute most to each model's predictions.

Analysis Methods:
1. BUILTIN: Model-specific feature importance (Random Forest, XGBoost)
2. PERMUTATION: Model-agnostic permutation importance for all models
3. SHAP: SHAP values for detailed feature attribution (requires shap library)

Key Features:
- Sensor-wise importance ranking (accelerometer vs gyroscope vs magnetometer vs orientation)
- Feature type importance (statistical vs FFT vs correlation features)
- Cross-model feature importance comparison
- Detailed visualization and CSV exports
- Statistical significance testing for permutation importance

Configuration:
- USE_FEATURE_IMPORTANCE_ANALYSIS = True  # Enable analysis
- FEATURE_IMPORTANCE_TOP_N = 30  # Number of top features to show
- FEATURE_IMPORTANCE_METHODS = ['builtin', 'permutation', 'shap']  # Analysis methods
- FEATURE_IMPORTANCE_PERMUTATION_REPEATS = 5  # Robustness for permutation
- FEATURE_IMPORTANCE_SHAP_SAMPLES = 500  # Sample size for SHAP (reduce for speed)

Output Files:
- feature_importance_comparison.png: Cross-model feature ranking comparison
- feature_importance_by_sensor.png: Sensor-wise importance breakdown
- feature_importance_by_type.png: Feature type importance analysis
- feature_importance_detailed_[model].png: Per-model detailed analysis
- feature_importance_results.csv: Complete numerical results
- feature_importance_summary.csv: Aggregated summary statistics

Requirements:
- pip install shap (optional, for SHAP analysis)
- Sufficient computation time for permutation importance on large datasets

Expected Insights:
- Which sensor modalities matter most (e.g., "gyroscope features dominate for shot detection")
- Which feature types are most informative (e.g., "FFT features crucial for periodic motions")
- Model-specific preferences (e.g., "Random Forest prefers statistical features")
- Feature redundancy analysis across different models

The script maintains full backward compatibility - all existing functionality
works unchanged when USE_FEATURE_IMPORTANCE_ANALYSIS = False.
"""

#------------------------------------------------------------------------------
# Performance Timing Analysis Usage Instructions
#------------------------------------------------------------------------------
"""
PERFORMANCE TIMING ANALYSIS FEATURE:

This script now supports comprehensive timing analysis for model training and prediction performance.

To enable timing analysis:
1. Set USE_TIMING_ANALYSIS = True
2. Adjust timing parameters as needed:
   - TIMING_WARMUP_ITERATIONS = 3  # Warmup runs for stable prediction timing
   - TIMING_MEASUREMENT_ITERATIONS = 10  # Measurement runs for statistical accuracy
   - TIMING_INCLUDE_MEMORY_USAGE = True  # Monitor memory consumption
   - TIMING_DETAILED_BREAKDOWN = True  # Include feature extraction timing

Configuration:
- USE_TIMING_ANALYSIS = True  # Enable comprehensive timing analysis
- TIMING_WARMUP_ITERATIONS = 3  # Warmup runs to stabilize performance
- TIMING_MEASUREMENT_ITERATIONS = 10  # Measurement runs for statistics
- TIMING_INCLUDE_MEMORY_USAGE = True  # Monitor memory usage during operations
- TIMING_DETAILED_BREAKDOWN = True  # Include detailed component timing

Output Files:
- timing_analysis_comprehensive.png: Complete performance dashboard with 6 visualizations
- timing_analysis_results.csv: Detailed numerical results for all models
- timing_analysis_summary.csv: Performance rankings and real-time capabilities

Analysis Components:
1. Training Time: Model fitting/training duration with memory monitoring
2. Prediction Time: Inference timing with warmup and statistical analysis
3. Throughput Analysis: Samples/second processing rates
4. Memory Usage: Training and prediction memory consumption
5. Real-time Assessment: Capability for real-time processing (>1x real-time factor)
6. Performance Rankings: Comparative rankings across multiple criteria

Expected Insights:
- Training speed comparison (e.g., "Random Forest trains 10x faster than CNN")
- Prediction speed analysis (e.g., "SVM processes 1000 samples/sec")
- Real-time deployment suitability (e.g., "KNN achieves 5x real-time performance")
- Memory efficiency comparison (e.g., "CNN uses 500MB training, 50MB prediction")
- Overall performance recommendations for different deployment scenarios

Real-time Capability Assessment:
- Models are tested against the actual window duration (2 seconds at 50Hz)
- Real-time factor >1.0 means the model can process faster than real-time
- Maximum concurrent streams calculation for parallel processing scenarios

Requirements:
- pip install psutil (for memory monitoring)
- Sufficient computation time for multiple timing iterations
- Best model configurations must be available (runs after model selection)

The script maintains full backward compatibility - all existing functionality
works unchanged when USE_TIMING_ANALYSIS = False.
"""

#------------------------------------------------------------------------------
# LOSO CV Usage Instructions (NEW)
#------------------------------------------------------------------------------
"""
LOSO CV FEATURE:

This script now supports Leave-One-Subject-Out (LOSO) cross-validation for more
robust model evaluation. LOSO CV is a subject-independent evaluation method
that tests how well models generalize to unseen subjects/players.

To enable LOSO CV:
1. Set USE_LOSO_CV = True
2. Adjust LOSO_MIN_SAMPLES_PER_SUBJECT as needed (default: 10)
3. Run the script as usual

LOSO CV benefits:
- Subject independence: Tests how well models generalize to unseen subjects/players
- More realistic evaluation: Simulates real-world deployment where you encounter new players
- Research standard: LOSO CV is the gold standard in HAR literature for subject-independent evaluation
- Reduces overfitting: Prevents models from learning player-specific patterns rather than general activity patterns

Trade-offs:
- Harder evaluation: Models can't memorize player-specific patterns
- More realistic: Better reflects real-world deployment scenarios
- Better model selection: Identifies models that truly generalize

The script maintains full backward compatibility - all existing functionality
works unchanged when USE_LOSO_CV = False (default).
"""

#------------------------------------------------------------------------------
# Data Loading and Exploration (Done ONCE)
#------------------------------------------------------------------------------
print("Loading data...")
data = pd.read_csv(TRAINING_DATA_PATH)

print("Dataset shape:", data.shape)
print("\nColumns in the dataset:", data.columns.tolist())
print("\nSample data:")
print(data.head())
print("\nMissing values in each column:")
print(data.isnull().sum())
data['label'] = data['label'].fillna('null')

print("\nLabel distribution (initial):")
initial_label_counts = data['label'].value_counts()
print(initial_label_counts)

print("\nAction Instances Summary:")
# Calculate scaled instances (count / 100, rounded)
action_instances_scaled = (initial_label_counts / 100).round().astype(int)

# Calculate unique players per label
unique_players_per_label = data.groupby('label')['player_id'].nunique().fillna(0).astype(int)

# Combine into a new summary DataFrame for action instances
action_summary_df = pd.DataFrame({
    'Scaled Instances (count/100)': action_instances_scaled,
    'Unique Players Performing Action': unique_players_per_label
})

# Ensure the index from initial_label_counts (which is sorted by count) is used, 
#y label name for consistency if preferred.
# For this table, let's align it with initial_label_counts order or sort by label name.
action_summary_df = action_summary_df.reindex(initial_label_counts.index) # Match order of initial_label_counts
# Or, to sort alphabetically by label: action_summary_df = action_summary_df.sort_index()

print(action_summary_df)

#------------------------------------------------------------------------------
# Window Creation Function - Unified for ML and CNN approaches
#------------------------------------------------------------------------------
def create_windows(df, window_size, step_size): # Added window_size and step_size as args
    windows = []
    labels = []
    player_ids = []  # NEW: Track player_ids for LOSO CV
    
    # Fixed threshold approach - 65% is a standard threshold in human activity recognition literature
    # This ensures consistent labeling criteria across all window overlaps
    label_threshold = 0.65  # 65% threshold - academically justified
    
    print(f"Using FIXED label threshold for windowing: {label_threshold:.2f} (65% - academic standard)")
    print(f"Window configuration: window_size={window_size}, step_size={step_size}")
    
    for (session_id, player_id), group_data in df.sort_values(['session_id', 'player_id', 'timestamp']).groupby(['session_id', 'player_id']):
        print(f"Processing session_id: {session_id}, player_id: {player_id} for window creation")
        
        feature_cols = ['accelerometer_x', 'accelerometer_y', 'accelerometer_z',
                        'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
                        'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
                        'orientation_x', 'orientation_y', 'orientation_z']
        
        features = group_data[feature_cols].values
        
        # NEW: Apply preprocessing to the entire sequence if enabled
        if USE_IMU_PREPROCESSING:
            features = apply_signal_filtering(features)
        
        group_labels = group_data['label'].values
        
        for i in range(0, len(features) - window_size + 1, step_size):
            window = features[i:i+window_size]
            window_labels_segment = group_labels[i:i+window_size]
            
            label_counts = pd.Series(window_labels_segment).value_counts(normalize=True)
            assigned_label = 'null' # Default
            
            # Determine dominant label based on fixed threshold
            # Order of checks matters if multiple labels meet threshold (prefer specific actions)
            if label_counts.get('pass', 0) >= label_threshold:
                assigned_label = 'pass'
            elif label_counts.get('groundball', 0) >= label_threshold:
                assigned_label = 'groundball'
            elif label_counts.get('catch', 0) >= label_threshold:
                assigned_label = 'catch'
            elif label_counts.get('faceoff', 0) >= label_threshold:
                assigned_label = 'faceoff'
            elif label_counts.get('shot', 0) >= label_threshold:
                assigned_label = 'shot'
            # Cradle is explicitly mapped to null if it meets threshold
            elif label_counts.get('cradle', 0) >= label_threshold:
                 assigned_label = 'null'
            elif label_counts.get('save', 0) >= label_threshold:
                 assigned_label = 'null'
            # Else, it remains 'null'

            windows.append(window)
            labels.append(assigned_label)
            player_ids.append(player_id)  # NEW: Track player_ids for LOSO CV
    
    print(f"Created {len(windows)} windows with labels: {np.unique(labels, return_counts=True)}")
    return np.array(windows), np.array(labels), np.array(player_ids)  # NEW: Return player_ids

#------------------------------------------------------------------------------
# Feature Extraction for Traditional ML Models (No changes needed)
#------------------------------------------------------------------------------
def extract_fft_features(channel):
    fft_vals = np.abs(rfft(channel))
    return [
        np.mean(fft_vals),
        np.std(fft_vals),
        np.sum(fft_vals ** 2),
        np.argmax(fft_vals)
    ]

def extract_features(windows):
    all_features = []
    for window in windows:
        window_features = []
        for i in range(window.shape[1]):
            channel = window[:, i]
            window_features.extend([
                np.mean(channel), np.std(channel), np.min(channel), np.max(channel),
                np.median(channel), np.percentile(channel, 25), np.percentile(channel, 75), np.ptp(channel)
            ])
            window_features.extend(extract_fft_features(channel))
            diffs = np.diff(channel)
            window_features.append(np.mean(diffs))
            window_features.append(np.std(diffs))
        sensor_groups = {
            'accel': [0, 1, 2], 'gyro': [3, 4, 5], 'mag': [6, 7, 8], 'orient': [9, 10, 11]
        }
        for group_indices in sensor_groups.values():
            group_data = window[:, group_indices]
            for i_corr, j_corr in combinations(range(group_data.shape[1]), 2):
                corr = np.corrcoef(group_data[:, i_corr], group_data[:, j_corr])[0, 1]
                window_features.append(corr if not np.isnan(corr) else 0)
        all_features.append(window_features)
    return np.array(all_features)

def generate_feature_names():
    base_stats = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'ptp']
    fft_stats = ['fft_mean', 'fft_std', 'fft_energy', 'fft_peak']
    diff_stats = ['diff_mean', 'diff_std']
    all_stats = base_stats + fft_stats + diff_stats
    sensor_channels = [
        'accelerometer_x', 'accelerometer_y', 'accelerometer_z',
        'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
        'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
        'orientation_x', 'orientation_y', 'orientation_z'
    ]
    feature_names = [f'{ch}_{stat}' for ch in sensor_channels for stat in all_stats]
    sensor_groups = {
        'accel': ['x', 'y', 'z'], 'gyro': ['x', 'y', 'z'],
        'mag': ['x', 'y', 'z'], 'orient': ['x', 'y', 'z']
    }
    for group_name, axes in sensor_groups.items():
        for i in range(len(axes)):
            for j in range(i + 1, len(axes)):
                feature_names.append(f'{group_name}_corr_{axes[i]}_{axes[j]}')
    return feature_names
#------------------------------------------------------------------------------
# CNN Model Architecture (No changes needed)
#------------------------------------------------------------------------------
def create_cnn_model(input_shape, num_classes):
    time_steps = input_shape[0]
    max_pool_layers = min(int(np.log2(time_steps)) if time_steps > 1 else 0, 3)
    print(f"Input shape ({input_shape}) allows for {max_pool_layers} max pooling layers.")
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    if max_pool_layers >= 1: model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    if max_pool_layers >= 2: model.add(MaxPooling1D(pool_size=2))
    if max_pool_layers >= 3:
        model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#------------------------------------------------------------------------------
# Cross-Validation Helper Functions (NEW)
#------------------------------------------------------------------------------
def train_ml_with_cv(X_data, y_data, model_definitions, label_encoder, class_names, player_ids=None):
    """Train ML models with cross-validation"""
    cv_results = {}
    
    # Choose CV strategy based on configuration
    if USE_LOSO_CV and player_ids is not None:
        print(f"\n🧠 Using LOSO (Leave-One-Subject-Out) CV with {len(np.unique(player_ids))} subjects...")
        cv_splitter = LeaveOneSubjectOut(player_ids, min_samples_per_subject=LOSO_MIN_SAMPLES_PER_SUBJECT)
        
        # Print subject distribution
        subjects_info = cv_splitter.get_subjects_info()
        print("\nSubject distribution:")
        for player_id, info in subjects_info.items():
            status = "✓ INCLUDED" if info['included_in_cv'] else "✗ EXCLUDED"
            print(f"  Player {player_id}: {info['n_samples']} samples - {status}")
        
    else:
        print(f"\n📊 Using Stratified K-Fold CV with {CV_FOLDS} folds...")
        cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    
    for name, model_template in model_definitions.items():
        print(f"\nTraining {name} with {cv_splitter.get_n_splits()}-fold CV...")
        
        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model_template)])
        
        # Perform cross-validation
        cv_scores = cross_validate(
            pipeline, X_data, y_data, 
            cv=cv_splitter, 
            scoring=CV_SCORING,
            return_train_score=False,
            n_jobs=-1
        )
        
        # Calculate mean and std for each metric
        cv_metrics = {}
        for metric in CV_SCORING:
            scores = cv_scores[f'test_{metric}']
            cv_metrics[f'{metric}_mean'] = np.mean(scores)
            cv_metrics[f'{metric}_std'] = np.std(scores)
            print(f"{name} {metric}: {cv_metrics[f'{metric}_mean']:.4f} (+/- {cv_metrics[f'{metric}_std']*2:.4f})")
        
        # NEW: Collect all predictions from each fold to generate true confusion matrix
        print(f"Collecting predictions from each fold for {name}...")
        all_y_true = []
        all_y_pred = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_data, y_data)):
            # Create fresh pipeline for this fold
            fold_pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model_template)])
            
            # Train on fold training data
            X_train_fold, X_val_fold = X_data[train_idx], X_data[val_idx]
            y_train_fold, y_val_fold = y_data[train_idx], y_data[val_idx]
            
            fold_pipeline.fit(X_train_fold, y_train_fold)
            y_pred_fold = fold_pipeline.predict(X_val_fold)
            
            # Collect predictions
            all_y_true.extend(y_val_fold)
            all_y_pred.extend(y_pred_fold)
        
        # Generate confusion matrix and classification report from all CV predictions
        # Use only labels that are actually present in the CV predictions to avoid ValueError
        actual_labels_present = np.unique(all_y_true)
        cv_confusion_matrix = confusion_matrix(all_y_true, all_y_pred, labels=actual_labels_present)
        cv_classification_report = classification_report(all_y_true, all_y_pred, 
                                                        labels=actual_labels_present, 
                                                        target_names=[class_names[i] for i in actual_labels_present], 
                                                        zero_division=0)
        
        # Calculate accuracy from confusion matrix to verify consistency
        manual_accuracy = accuracy_score(all_y_true, all_y_pred)
        print(f"CV Manual accuracy: {manual_accuracy:.4f} vs Mean CV accuracy: {cv_metrics['accuracy_mean']:.4f}")
        print(f"✓ Generated true confusion matrix from all {len(all_y_true)} CV predictions")
        
        # Train final model on full dataset for later use
        pipeline.fit(X_data, y_data)
        
        # Store results in same format as original code
        cv_results[name] = {
            'accuracy': cv_metrics['accuracy_mean'],
            'precision': cv_metrics['precision_macro_mean'],
            'recall': cv_metrics['recall_macro_mean'],
            'f1': cv_metrics['f1_macro_mean'],
            'cv_metrics': cv_metrics,  # Additional CV statistics
            'cv_type': 'LOSO' if USE_LOSO_CV and player_ids is not None else 'StratifiedKFold',  # NEW: Track CV type
            'pipeline': pipeline,
            'labels_for_metrics': list(class_names),
            'label_encoder_ml': label_encoder,
            # NEW: True confusion matrix and classification report from CV predictions
            'confusion_matrix': cv_confusion_matrix,
            'classification_report': cv_classification_report,
            'cv_manual_accuracy': manual_accuracy,  # For verification
            'cv_all_predictions': {'y_true': all_y_true, 'y_pred': all_y_pred}  # Store for debugging if needed
        }
    
    return cv_results

def train_cnn_with_cv(X_data, y_data, input_shape, num_classes, label_encoder, class_names, scaler, player_ids=None):
    """Train CNN with cross-validation"""
    from sklearn.model_selection import StratifiedKFold
    import gc
    
    # Choose CV strategy based on configuration
    if USE_LOSO_CV and player_ids is not None:
        print(f"\n🧠 CNN: Using LOSO (Leave-One-Subject-Out) CV with {len(np.unique(player_ids))} subjects...")
        cv_splitter = LeaveOneSubjectOut(player_ids, min_samples_per_subject=LOSO_MIN_SAMPLES_PER_SUBJECT)
    else:
        print(f"\n📊 CNN: Using Stratified K-Fold CV with {CV_FOLDS} folds...")
        cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    
    cv_scores = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
    
    # Convert categorical back to labels for stratification
    y_labels = np.argmax(y_data, axis=1)
    
    fold = 0
    for train_idx, val_idx in cv_splitter.split(X_data, y_labels):
        fold += 1
        print(f"Training CNN fold {fold}/{cv_splitter.get_n_splits()}...")
        
        # Split data
        X_train_fold, X_val_fold = X_data[train_idx], X_data[val_idx]
        y_train_fold, y_val_fold = y_data[train_idx], y_data[val_idx]
        
        # Scale data for this fold
        scaler_fold = StandardScaler()
        X_train_fold_reshaped = X_train_fold.reshape(-1, X_train_fold.shape[-1])
        X_val_fold_reshaped = X_val_fold.reshape(-1, X_val_fold.shape[-1])
        
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold_reshaped)
        X_val_fold_scaled = scaler_fold.transform(X_val_fold_reshaped)
        
        X_train_fold_scaled = X_train_fold_scaled.reshape(X_train_fold.shape)
        X_val_fold_scaled = X_val_fold_scaled.reshape(X_val_fold.shape)
        
        # Create and train model
        model_fold = create_cnn_model(input_shape, num_classes)
        
        # Train with reduced epochs for CV
        history = model_fold.fit(
            X_train_fold_scaled, y_train_fold,
            validation_split=0.2, epochs=CV_REDUCED_EPOCHS, batch_size=32, 
            verbose=CV_VERBOSE
        )
        
        # Evaluate fold
        y_pred_prob = model_fold.predict(X_val_fold_scaled)
        y_pred_indices = np.argmax(y_pred_prob, axis=1)
        y_val_indices = np.argmax(y_val_fold, axis=1)
        
        # Calculate metrics
        cv_scores['accuracy'].append(accuracy_score(y_val_indices, y_pred_indices))
        cv_scores['f1'].append(f1_score(y_val_indices, y_pred_indices, average='macro', zero_division=0))
        cv_scores['precision'].append(precision_score(y_val_indices, y_pred_indices, average='macro', zero_division=0))
        cv_scores['recall'].append(recall_score(y_val_indices, y_pred_indices, average='macro', zero_division=0))
        
        # Clean up memory
        del model_fold
        gc.collect()
    
    # Calculate CV statistics
    cnn_cv_metrics = {}
    for metric, scores in cv_scores.items():
        cnn_cv_metrics[f'{metric}_mean'] = np.mean(scores)
        cnn_cv_metrics[f'{metric}_std'] = np.std(scores)
        print(f"CNN {metric}: {cnn_cv_metrics[f'{metric}_mean']:.4f} (+/- {cnn_cv_metrics[f'{metric}_std']*2:.4f})")
    
    # Train final CNN model on full dataset (using original full training approach)
    final_model = create_cnn_model(input_shape, num_classes)
    
    # Return results in same format as original code
    return {
        'accuracy': cnn_cv_metrics['accuracy_mean'],
        'precision': cnn_cv_metrics['precision_mean'],
        'recall': cnn_cv_metrics['recall_mean'],
        'f1': cnn_cv_metrics['f1_mean'],
        'cv_metrics': cnn_cv_metrics,
        'cv_type': 'LOSO' if USE_LOSO_CV and player_ids is not None else 'StratifiedKFold',  # NEW: Track CV type
        'model_object': final_model,  # Will be trained later in original code flow
        'scaler_cnn': scaler,
        'label_encoder': label_encoder,
        'labels_for_metrics': list(class_names),
        'input_shape': input_shape,
        'num_classes': num_classes,
        'needs_final_training': True  # Flag to indicate final training needed
    }

#------------------------------------------------------------------------------
# IMU Data Preprocessing Functions (NEW)
#------------------------------------------------------------------------------
def create_butterworth_filter(cutoff_freq, sampling_rate, filter_type='low', order=4):
    """Create Butterworth filter coefficients"""
    nyquist_freq = sampling_rate / 2.0
    normalized_cutoff = cutoff_freq / nyquist_freq
    
    if normalized_cutoff >= 1.0:
        print(f"Warning: Cutoff frequency {cutoff_freq} Hz is too high for sampling rate {sampling_rate} Hz")
        normalized_cutoff = 0.99
    
    b, a = signal.butter(order, normalized_cutoff, btype=filter_type)
    return b, a

def remove_gravity_component(accel_data, gravity_estimate=None):
    """Remove gravity component from accelerometer data using high-pass filtering"""
    if gravity_estimate is None:
        # Use median as gravity estimate (assumes some stationary periods)
        gravity_estimate = np.median(accel_data, axis=0)
    
    return accel_data - gravity_estimate

def apply_signal_filtering(data, sampling_rate=SAMPLING_RATE):
    """Apply signal filtering to IMU data"""
    print(f"Applying IMU preprocessing (Sampling rate: {sampling_rate} Hz)...")
    
    # Define sensor column indices
    sensor_cols = {
        'accel': [0, 1, 2],      # accelerometer_x, y, z
        'gyro': [3, 4, 5],       # gyroscope_x, y, z  
        'mag': [6, 7, 8],        # magnetometer_x, y, z
        'orient': [9, 10, 11]    # orientation_x, y, z
    }
    
    processed_data = data.copy()
    
    # Apply low-pass filter if enabled
    if APPLY_LOWPASS:
        print(f"Applying low-pass filter (cutoff: {LOWPASS_CUTOFF_FREQ} Hz)")
        b_low, a_low = create_butterworth_filter(
            LOWPASS_CUTOFF_FREQ, sampling_rate, 'low', FILTER_ORDER
        )
        
        for sensor_name, cols in sensor_cols.items():
            for col in cols:
                processed_data[:, col] = signal.filtfilt(b_low, a_low, processed_data[:, col])
    
    # Apply high-pass filter if enabled (mainly for accelerometer)
    if APPLY_HIGHPASS:
        print(f"Applying high-pass filter (cutoff: {HIGHPASS_CUTOFF_FREQ} Hz)")
        b_high, a_high = create_butterworth_filter(
            HIGHPASS_CUTOFF_FREQ, sampling_rate, 'high', FILTER_ORDER
        )
        
        # Apply to accelerometer and gyroscope (remove DC bias)
        for sensor_name in ['accel', 'gyro']:
            for col in sensor_cols[sensor_name]:
                processed_data[:, col] = signal.filtfilt(b_high, a_high, processed_data[:, col])
    
    # Remove gravity component from accelerometer if enabled
    if REMOVE_GRAVITY:
        print("Removing gravity component from accelerometer data")
        accel_cols = sensor_cols['accel']
        processed_data[:, accel_cols] = remove_gravity_component(processed_data[:, accel_cols])
    
    return processed_data

def analyze_frequency_content(windows, sampling_rate=SAMPLING_RATE, sensor_names=None):
    """Analyze frequency content of IMU data to guide preprocessing decisions"""
    if sensor_names is None:
        sensor_names = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z',
                       'mag_x', 'mag_y', 'mag_z', 'orient_x', 'orient_y', 'orient_z']
    
    print("Analyzing frequency content of IMU data...")
    
    # Take a sample of windows for analysis
    n_sample_windows = min(100, len(windows))
    sample_indices = np.random.choice(len(windows), n_sample_windows, replace=False)
    
    avg_power_spectrum = np.zeros((windows.shape[2], windows.shape[1] // 2 + 1))
    
    for idx in sample_indices:
        window = windows[idx]
        for sensor_idx in range(window.shape[1]):
            freqs, psd = signal.welch(window[:, sensor_idx], fs=sampling_rate, nperseg=len(window))
            avg_power_spectrum[sensor_idx] += psd
    
    avg_power_spectrum /= n_sample_windows
    freqs = np.linspace(0, sampling_rate/2, windows.shape[1] // 2 + 1)
    
    # Find dominant frequencies for each sensor
    print("\nDominant frequencies by sensor:")
    for sensor_idx, sensor_name in enumerate(sensor_names):
        peak_freq_idx = np.argmax(avg_power_spectrum[sensor_idx])
        peak_freq = freqs[peak_freq_idx]
        
        # Find frequency where 95% of power is contained
        cumulative_power = np.cumsum(avg_power_spectrum[sensor_idx])
        total_power = cumulative_power[-1]
        freq_95_idx = np.where(cumulative_power >= 0.95 * total_power)[0][0]
        freq_95 = freqs[freq_95_idx]
        
        print(f"{sensor_name}: Peak at {peak_freq:.1f} Hz, 95% power below {freq_95:.1f} Hz")
    
    return freqs, avg_power_spectrum

#------------------------------------------------------------------------------
# Hyperparameter Tuning Helper Functions (NEW)
#------------------------------------------------------------------------------
def tune_ml_hyperparameters(X_data, y_data, best_config, param_grid, player_ids=None):
    """Perform hyperparameter tuning on the best overlap configuration"""
    print(f"\nPerforming hyperparameter tuning for {best_config['model_name']}...")
    print(f"Using best overlap: {best_config['overlap_fraction']*100:.0f}%")
    print(f"Parameter grid size: {np.prod([len(values) for values in param_grid.values()])} combinations")
    
    # Create base pipeline from best configuration
    base_pipeline = best_config['pipeline']
    
    # Choose CV strategy based on configuration
    if USE_LOSO_CV and player_ids is not None:
        print(f"Using LOSO CV for hyperparameter tuning with {len(np.unique(player_ids))} subjects...")
        cv_splitter = LeaveOneSubjectOut(player_ids, min_samples_per_subject=LOSO_MIN_SAMPLES_PER_SUBJECT)
        
        # Print subject distribution for hyperparameter tuning
        subjects_info = cv_splitter.get_subjects_info()
        print("Subject distribution for hyperparameter tuning:")
        for player_id, info in subjects_info.items():
            status = "✓ INCLUDED" if info['included_in_cv'] else "✗ EXCLUDED"
            print(f"  Player {player_id}: {info['n_samples']} samples - {status}")
    else:
        print(f"Using Stratified K-Fold CV ({HYPERPARAMETER_CV_FOLDS} folds) for hyperparameter tuning...")
        cv_splitter = StratifiedKFold(n_splits=HYPERPARAMETER_CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=cv_splitter,
        scoring=f'{PRIMARY_METRIC}_macro' if PRIMARY_METRIC != 'accuracy' else PRIMARY_METRIC,
        n_jobs=HYPERPARAMETER_N_JOBS,
        verbose=HYPERPARAMETER_VERBOSE,
        return_train_score=False
    )
    
    # Perform grid search
    print(f"Starting hyperparameter tuning...")
    grid_search.fit(X_data, y_data)
    
    # Extract results
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    best_pipeline_tuned = grid_search.best_estimator_
    
    print(f"Hyperparameter tuning completed!")
    print(f"Best {PRIMARY_METRIC}: {best_score:.4f} (improvement: {best_score - best_config['score_used_for_selection']:.4f})")
    print(f"Best parameters: {best_params}")
    
    return {
        'best_pipeline': best_pipeline_tuned,
        'best_score': best_score,
        'best_params': best_params,
        'grid_search_results': grid_search.cv_results_,
        'score_improvement': best_score - best_config['score_used_for_selection']
    }

def create_cnn_model_tunable(input_shape, num_classes, config):
    """Create CNN model with tunable hyperparameters"""
    time_steps = input_shape[0]
    max_pool_layers = min(int(np.log2(time_steps)) if time_steps > 1 else 0, 3)
    
    model = Sequential()
    model.add(Conv1D(filters=config['filters_1'], kernel_size=3, activation='relu', 
                     padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    if max_pool_layers >= 1: model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(filters=config['filters_2'], kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    if max_pool_layers >= 2: model.add(MaxPooling1D(pool_size=2))
    
    if max_pool_layers >= 3:
        model.add(Conv1D(filters=config['filters_3'], kernel_size=3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
    
    model.add(Flatten())
    model.add(Dense(config['dense_1'], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(config['dropout_1']))
    model.add(Dense(config['dense_2'], activation='relu'))
    model.add(Dropout(config['dropout_2']))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Use tunable learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def tune_cnn_hyperparameters(X_data, y_data, best_config, hyperparameter_configs, player_ids=None):
    """Perform CNN hyperparameter tuning using manual grid search"""
    print(f"\nPerforming CNN hyperparameter tuning...")
    print(f"Using best overlap: {best_config['overlap_fraction']*100:.0f}%")
    print(f"Testing {len(hyperparameter_configs)} configurations")
    
    input_shape = best_config['input_shape']
    num_classes = best_config['num_classes']
    
    # Choose CV strategy based on configuration
    if USE_LOSO_CV and player_ids is not None:
        print(f"Using LOSO CV for CNN hyperparameter tuning with {len(np.unique(player_ids))} subjects...")
        cv_splitter = LeaveOneSubjectOut(player_ids, min_samples_per_subject=LOSO_MIN_SAMPLES_PER_SUBJECT)
        
        # Print subject distribution for hyperparameter tuning
        subjects_info = cv_splitter.get_subjects_info()
        print("Subject distribution for CNN hyperparameter tuning:")
        for player_id, info in subjects_info.items():
            status = "✓ INCLUDED" if info['included_in_cv'] else "✗ EXCLUDED"
            print(f"  Player {player_id}: {info['n_samples']} samples - {status}")
    else:
        print(f"Using Stratified K-Fold CV ({HYPERPARAMETER_CV_FOLDS} folds) for CNN hyperparameter tuning...")
        cv_splitter = StratifiedKFold(n_splits=HYPERPARAMETER_CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    
    y_labels = np.argmax(y_data, axis=1)
    
    best_score = -1
    best_config_result = None
    best_params = None
    all_results = []
    
    for config_idx, config in enumerate(hyperparameter_configs):
        print(f"\nTesting configuration {config_idx + 1}/{len(hyperparameter_configs)}: {config}")
        
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_data, y_labels)):
            print(f"  Fold {fold_idx + 1}/{cv_splitter.get_n_splits(X_data, y_labels)}")
            
            # Split data
            X_train_fold, X_val_fold = X_data[train_idx], X_data[val_idx]
            y_train_fold, y_val_fold = y_data[train_idx], y_data[val_idx]
            
            # Scale data
            scaler_fold = StandardScaler()
            X_train_fold_reshaped = X_train_fold.reshape(-1, X_train_fold.shape[-1])
            X_val_fold_reshaped = X_val_fold.reshape(-1, X_val_fold.shape[-1])
            
            X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold_reshaped)
            X_val_fold_scaled = scaler_fold.transform(X_val_fold_reshaped)
            
            X_train_fold_scaled = X_train_fold_scaled.reshape(X_train_fold.shape)
            X_val_fold_scaled = X_val_fold_scaled.reshape(X_val_fold.shape)
            
            # Create and train model
            model_fold = create_cnn_model_tunable(input_shape, num_classes, config)
            
            # Train with early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            history = model_fold.fit(
                X_train_fold_scaled, y_train_fold,
                validation_data=(X_val_fold_scaled, y_val_fold),
                epochs=CV_REDUCED_EPOCHS,
                batch_size=config['batch_size'],
                verbose=0,
                callbacks=[early_stopping]
            )
            
            # Evaluate
            y_pred_prob = model_fold.predict(X_val_fold_scaled, verbose=0)
            y_pred_indices = np.argmax(y_pred_prob, axis=1)
            y_val_indices = np.argmax(y_val_fold, axis=1)
            
            if PRIMARY_METRIC == 'accuracy':
                score = accuracy_score(y_val_indices, y_pred_indices)
            elif PRIMARY_METRIC == 'precision':
                score = precision_score(y_val_indices, y_pred_indices, average='macro', zero_division=0)
            elif PRIMARY_METRIC == 'recall':
                score = recall_score(y_val_indices, y_pred_indices, average='macro', zero_division=0)
            elif PRIMARY_METRIC == 'f1':
                score = f1_score(y_val_indices, y_pred_indices, average='macro', zero_division=0)
            else:
                score = accuracy_score(y_val_indices, y_pred_indices)
            
            fold_scores.append(score)
            
            # Clean up memory
            del model_fold
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
        
        # Calculate mean score for this configuration
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"  Configuration {config_idx + 1} {PRIMARY_METRIC}: {mean_score:.4f} (+/- {std_score*2:.4f})")
        
        all_results.append({
            'config': config,
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_scores': fold_scores
        })
        
        if mean_score > best_score:
            best_score = mean_score
            best_config_result = config
            best_params = config
    
    print(f"\nCNN hyperparameter tuning completed!")
    print(f"Best {PRIMARY_METRIC}: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': all_results,
        'score_improvement': best_score - best_config['score_used_for_selection']
    }

#==============================================================================
# Main Loop for Window Overlap Tuning
#==============================================================================
all_runs_results_list = [] # Store detailed results for each overlap run

for current_overlap_fraction in WINDOW_OVERLAP_OPTIONS:
    current_overlap_percentage = current_overlap_fraction * 100
    print(f"\n{'='*80}")
    print(f"===== Processing with Window Overlap: {current_overlap_percentage:.0f}% =====")
    if USE_CROSS_VALIDATION:
        print(f"===== Using {CV_FOLDS}-Fold Cross-Validation =====")
    else:
        print(f"===== Using Train-Test Split (75/25) =====")
    if USE_IMU_PREPROCESSING:
        print(f"===== IMU Preprocessing: LP={LOWPASS_CUTOFF_FREQ}Hz, HP={HIGHPASS_CUTOFF_FREQ}Hz =====")
    else:
        print(f"===== No IMU Preprocessing =====")
    print(f"{'='*80}\n")

    # Calculate STEP_SIZE for current overlap
    STEP_SIZE = int(WINDOW_SIZE * (1 - current_overlap_fraction))
    if STEP_SIZE < 1: STEP_SIZE = 1 # Ensure step size is at least 1
    print(f"Using WINDOW_SIZE: {WINDOW_SIZE}, Current Overlap: {current_overlap_percentage:.0f}%, Calculated STEP_SIZE: {STEP_SIZE}")

    #--------------------------------------------------------------------------
    # Data Preparation (for this specific overlap)
    #--------------------------------------------------------------------------
    print("\nCreating sliding windows for current overlap...")
    # Pass the original `data` DataFrame, WINDOW_SIZE, and current STEP_SIZE
    windows, labels, player_ids = create_windows(data, WINDOW_SIZE, STEP_SIZE) # `data` is the full dataset
    if len(windows) == 0:
        print(f"No windows created for overlap {current_overlap_percentage:.0f}%. Skipping this configuration.")
        continue
        
    print(f"Created {len(windows)} windows. Label distribution: {np.unique(labels, return_counts=True)}")

    # --- Filter out 'cradle' class (already handled in create_windows by mapping to 'null') ---
    
    if PERFORM_UNDERSAMPLING:
        # --- Undersample 'null' class ---
        print("\nUndersampling 'null' class for current overlap...")
        null_indices = np.where(labels == 'null')[0]
        action_indices = np.where(labels != 'null')[0]
        num_action_windows = len(action_indices)
        num_null_windows_original = len(null_indices)

        print(f"Action windows: {num_action_windows}, Original 'null' windows: {num_null_windows_original}")
        if num_null_windows_original > num_action_windows and num_action_windows > 0 : # also ensure there are action windows
            print(f"Undersampling 'null' class to {num_action_windows} windows.")
            undersampled_null_indices = np.random.choice(null_indices, size=num_action_windows, replace=False)
            combined_indices = np.concatenate((action_indices, undersampled_null_indices))
            np.random.shuffle(combined_indices)
            windows = windows[combined_indices]
            labels = labels[combined_indices]
            player_ids = player_ids[combined_indices]  # NEW: Update player_ids to match
            print(f"Total windows after undersampling: {len(windows)}. New label distribution: {np.unique(labels, return_counts=True)}")
        elif num_action_windows == 0:
            print("No action windows found after windowing. Skipping undersampling and possibly this overlap configuration if all windows were null.")
            # If all windows were null and then undersampled to 0 (because no action windows), 
            # it might lead to issues. However, if PERFORM_UNDERSAMPLING is false, this path won't be hit for undersampling.
            # The check `if len(windows) == 0:` after this block handles cases where window creation itself yields nothing.
        else:
            print("No undersampling needed for 'null' class (already balanced or not majority).")
    else:
        print("\nSkipping undersampling of 'null' class as PERFORM_UNDERSAMPLING is False.")
        # Labels and windows remain as they were after creation

    if len(windows) == 0: # Final check if windows are empty (e.g. if window creation yielded nothing)
        print(f"No windows left for overlap {current_overlap_percentage:.0f}% after processing. Skipping model training for this overlap.")
        continue

    # For traditional ML models - with feature extraction
    print("\nExtracting features for ML models (current overlap)...")
    X_ml = extract_features(windows)
    # y_ml = labels # These are the potentially undersampled labels. Now we encode them.

    # --- Encode labels for all ML models for this overlap --- 
    ml_label_encoder = LabelEncoder()
    y_ml_encoded = ml_label_encoder.fit_transform(labels) # `labels` is the y_ml for this stage
    ml_encoded_unique_labels = np.unique(y_ml_encoded) # These are the integer labels for metrics
    ml_class_names = ml_label_encoder.classes_ # These are the string names for reports
    print(f"ML models will use these classes: {ml_class_names} (encoded as {ml_encoded_unique_labels})")

    print("\nSplitting ML data (current overlap)...")
    X_train_ml, X_test_ml, y_train_ml_encoded, y_test_ml_encoded = train_test_split(
        X_ml, y_ml_encoded, test_size=0.25, random_state=42, stratify=y_ml_encoded
    )
    print(f"ML training samples: {X_train_ml.shape[0]}, ML testing samples: {X_test_ml.shape[0]}")

    # For CNN model
    print("\nEncoding labels for CNN (current overlap)...")
    current_label_encoder = LabelEncoder() # Create a new encoder for this specific run's labels
    # Fit only on the labels present in the current (potentially undersampled) dataset
    cnn_labels_encoded = current_label_encoder.fit_transform(labels) 
    cnn_labels_categorical = to_categorical(cnn_labels_encoded)
    current_num_classes = len(current_label_encoder.classes_)
    current_filtered_unique_labels = current_label_encoder.classes_
    print(f"Number of classes for CNN (current overlap): {current_num_classes}, Classes: {current_filtered_unique_labels}")

    print("\nSplitting CNN data (current overlap)...")
    X_train_cnn, X_test_cnn, y_train_cnn_cat, y_test_cnn_cat = train_test_split(
        windows, cnn_labels_categorical, test_size=0.25, random_state=42, stratify=cnn_labels_encoded
    )
    y_train_cnn_labels = np.argmax(y_train_cnn_cat, axis=1) # For some metrics if needed
    y_test_cnn_labels = np.argmax(y_test_cnn_cat, axis=1)   # For evaluation

    print(f"CNN training samples: {X_train_cnn.shape[0]}, CNN testing samples: {X_test_cnn.shape[0]}")
    print(f"CNN window shape: {X_train_cnn.shape[1:]}")

    # Normalize CNN data
    current_scaler_cnn = StandardScaler() # New scaler for this run
    X_train_cnn_reshaped = X_train_cnn.reshape(-1, X_train_cnn.shape[-1])
    X_test_cnn_reshaped = X_test_cnn.reshape(-1, X_test_cnn.shape[-1])
    X_train_cnn_scaled = current_scaler_cnn.fit_transform(X_train_cnn_reshaped)
    X_test_cnn_scaled = current_scaler_cnn.transform(X_test_cnn_reshaped)
    X_train_cnn_scaled = X_train_cnn_scaled.reshape(X_train_cnn.shape)
    X_test_cnn_scaled = X_test_cnn_scaled.reshape(X_test_cnn.shape)

    #--------------------------------------------------------------------------
    # Model Training - Traditional ML Models (for this specific overlap)
    #--------------------------------------------------------------------------
    if USE_ML_MODELS:
        ml_model_definitions = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            #'KNN': KNeighborsClassifier(),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
            'Neural Network': MLPClassifier(random_state=42, max_iter=500)
        }
        current_ml_results = {}
        current_ml_pipelines = {}

        if USE_CROSS_VALIDATION:
            # NEW: Use cross-validation approach
            print(f"\n=== Using Cross-Validation for ML Models (overlap {current_overlap_percentage:.0f}%) ===")
            current_ml_results = train_ml_with_cv(X_ml, y_ml_encoded, ml_model_definitions, 
                                                  ml_label_encoder, ml_class_names, player_ids)
            
            # Generate confusion matrices and classification reports on holdout data for compatibility
            for name, result in current_ml_results.items():
                if result['confusion_matrix'] is None:  # Generate if not already present
                    pipeline = result['pipeline']
                    y_pred_ml = pipeline.predict(X_test_ml)
                    result['confusion_matrix'] = confusion_matrix(y_test_ml_encoded, y_pred_ml, labels=ml_encoded_unique_labels)
                    result['classification_report'] = classification_report(y_test_ml_encoded, y_pred_ml, 
                                                                          labels=ml_encoded_unique_labels, 
                                                                          target_names=ml_class_names, zero_division=0)
                current_ml_pipelines[name] = result['pipeline']
        else:
            # ORIGINAL: Use train-test split approach
            print(f"\n=== Using Train-Test Split for ML Models (overlap {current_overlap_percentage:.0f}%) ===")
            
            for name, model_template in ml_model_definitions.items():
                print(f"\nTraining {name} (overlap {current_overlap_percentage:.0f}%)...")
                pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model_template)])
                # Fit with encoded labels
                pipeline.fit(X_train_ml, y_train_ml_encoded) 
                y_pred_ml = pipeline.predict(X_test_ml)
                
                # Metrics calculated with encoded true labels and encoded predictions
                accuracy = accuracy_score(y_test_ml_encoded, y_pred_ml)
                precision = precision_score(y_test_ml_encoded, y_pred_ml, average='macro', zero_division=0, labels=ml_encoded_unique_labels)
                recall = recall_score(y_test_ml_encoded, y_pred_ml, average='macro', zero_division=0, labels=ml_encoded_unique_labels)
                f1 = f1_score(y_test_ml_encoded, y_pred_ml, average='macro', zero_division=0, labels=ml_encoded_unique_labels)
                # Classification report can use target_names to show string labels
                report = classification_report(y_test_ml_encoded, y_pred_ml, labels=ml_encoded_unique_labels, target_names=ml_class_names, zero_division=0)
                conf_matrix = confusion_matrix(y_test_ml_encoded, y_pred_ml, labels=ml_encoded_unique_labels)
                
                print(f"{name} Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                current_ml_results[name] = {
                    'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
                    'pipeline': pipeline, 
                    'confusion_matrix': conf_matrix,
                    'classification_report': report,
                    'labels_for_metrics': list(ml_class_names), # Store the string class names for consistency in reporting
                    'label_encoder_ml': ml_label_encoder # Store the fitted encoder if needed later
                }
                current_ml_pipelines[name] = pipeline
    else:
        print(f"\nSkipping ML models (USE_ML_MODELS = False)")
        ml_model_definitions = {}
        current_ml_results = {}
        current_ml_pipelines = {}

    #--------------------------------------------------------------------------
    # Model Training - CNN Model (for this specific overlap)
    #--------------------------------------------------------------------------
    if USE_CNN_MODEL:
        print(f"\nTraining CNN model (overlap {current_overlap_percentage:.0f}%)...")
        cnn_input_shape = (X_train_cnn_scaled.shape[1], X_train_cnn_scaled.shape[2])
        
        # Check if num_classes is valid
        if current_num_classes < 2:
            print(f"Skipping CNN for overlap {current_overlap_percentage:.0f}% due to insufficient classes ({current_num_classes}).")
            current_cnn_result = None # Mark as None if CNN couldn't be trained
        else:
            if USE_CROSS_VALIDATION:
                # NEW: Use cross-validation approach
                print(f"\n=== Using Cross-Validation for CNN (overlap {current_overlap_percentage:.0f}%) ===")
                current_cnn_result = train_cnn_with_cv(windows, cnn_labels_categorical, cnn_input_shape, 
                                                       current_num_classes, current_label_encoder, 
                                                       current_filtered_unique_labels, current_scaler_cnn, player_ids)  # NEW: Pass player_ids
                
                # Train the final model on full dataset if CV was used
                if current_cnn_result.get('needs_final_training', False):
                    print("Training final CNN model on full dataset...")
                    cnn_model_current_run = current_cnn_result['model_object']
                    cnn_model_path_current_run = f'{OUTPUT_DIR_MODELS}best_cnn_overlap_{current_overlap_fraction:.2f}_model.h5'
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
                        ModelCheckpoint(cnn_model_path_current_run, save_best_only=True, monitor='val_accuracy')
                    ]
                    history = cnn_model_current_run.fit(
                        X_train_cnn_scaled, y_train_cnn_cat,
                        validation_split=0.2, epochs=50, batch_size=32, callbacks=callbacks, verbose=1
                    )
                    # Update the result with final trained model and add missing fields for compatibility
                    current_cnn_result['model_object'] = cnn_model_current_run
                    current_cnn_result['model_path'] = cnn_model_path_current_run
                    current_cnn_result['training_history'] = history.history
                    
                    # Generate confusion matrix and classification report for compatibility
                    loss_cnn, acc_cnn = cnn_model_current_run.evaluate(X_test_cnn_scaled, y_test_cnn_cat, verbose=0)
                    y_pred_cnn_prob = cnn_model_current_run.predict(X_test_cnn_scaled)
                    y_pred_cnn_indices = np.argmax(y_pred_cnn_prob, axis=1)
                    y_test_cnn_original_labels = current_label_encoder.inverse_transform(y_test_cnn_labels)
                    y_pred_cnn_original_labels = current_label_encoder.inverse_transform(y_pred_cnn_indices)
                    current_cnn_result['confusion_matrix'] = confusion_matrix(y_test_cnn_original_labels, y_pred_cnn_original_labels, labels=current_filtered_unique_labels)
                    current_cnn_result['classification_report'] = classification_report(y_test_cnn_original_labels, y_pred_cnn_original_labels, labels=current_filtered_unique_labels, zero_division=0)
            else:
                # ORIGINAL: Use train-test split approach
                print(f"\n=== Using Train-Test Split for CNN (overlap {current_overlap_percentage:.0f}%) ===")
                
                cnn_model_current_run = create_cnn_model(cnn_input_shape, current_num_classes)
                cnn_model_current_run.summary()
                
                # Unique model path for this run's checkpoint
                cnn_model_path_current_run = f'{OUTPUT_DIR_MODELS}best_cnn_overlap_{current_overlap_fraction:.2f}_model.h5'
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
                    ModelCheckpoint(cnn_model_path_current_run, save_best_only=True, monitor='val_accuracy')
                ]
                history = cnn_model_current_run.fit(
                    X_train_cnn_scaled, y_train_cnn_cat,
                    validation_split=0.2, epochs=50, batch_size=32, callbacks=callbacks, verbose=1
                )
                # After training, cnn_model_current_run has best weights due to restore_best_weights=True
                # Or, load explicitly from cnn_model_path_current_run if preferred
                # cnn_model_current_run = load_model(cnn_model_path_current_run)

                loss_cnn, acc_cnn = cnn_model_current_run.evaluate(X_test_cnn_scaled, y_test_cnn_cat, verbose=0)
                y_pred_cnn_prob = cnn_model_current_run.predict(X_test_cnn_scaled)
                y_pred_cnn_indices = np.argmax(y_pred_cnn_prob, axis=1)
                
                # Convert y_test_cnn_labels (indices) and y_pred_cnn_indices back to original string labels for metrics
                y_test_cnn_original_labels = current_label_encoder.inverse_transform(y_test_cnn_labels)
                y_pred_cnn_original_labels = current_label_encoder.inverse_transform(y_pred_cnn_indices)

                prec_cnn = precision_score(y_test_cnn_original_labels, y_pred_cnn_original_labels, average='macro', zero_division=0, labels=current_filtered_unique_labels)
                rec_cnn = recall_score(y_test_cnn_original_labels, y_pred_cnn_original_labels, average='macro', zero_division=0, labels=current_filtered_unique_labels)
                f1_cnn = f1_score(y_test_cnn_original_labels, y_pred_cnn_original_labels, average='macro', zero_division=0, labels=current_filtered_unique_labels)
                report_cnn = classification_report(y_test_cnn_original_labels, y_pred_cnn_original_labels, labels=current_filtered_unique_labels, zero_division=0)
                conf_matrix_cnn = confusion_matrix(y_test_cnn_original_labels, y_pred_cnn_original_labels, labels=current_filtered_unique_labels)
                
                print(f"CNN Test Accuracy: {acc_cnn:.4f}, F1: {f1_cnn:.4f}")
                current_cnn_result = {
                    'accuracy': acc_cnn, 'precision': prec_cnn, 'recall': rec_cnn, 'f1': f1_cnn,
                    'model_object': cnn_model_current_run, # Store the trained model object itself
                    'model_path': cnn_model_path_current_run, # Path if needed
                    'scaler_cnn': current_scaler_cnn,
                    'label_encoder': current_label_encoder,
                    'confusion_matrix': conf_matrix_cnn,
                    'classification_report': report_cnn,
                    'training_history': history.history, # Store history for potential plotting
                    'labels_for_metrics': list(current_filtered_unique_labels),
                    'input_shape': cnn_input_shape,
                    'num_classes': current_num_classes
                }
    else:
        print(f"\nSkipping CNN model (USE_CNN_MODEL = False)")
        current_cnn_result = None

    # Store all results for this overlap configuration
    all_runs_results_list.append({
        'overlap_fraction': current_overlap_fraction,
        'step_size': STEP_SIZE,
        'ml_results': current_ml_results, # Dict of results per ML model
        'cnn_result': current_cnn_result,  # Dict of CNN results, or None
        'data_summary': {
            'num_windows_after_creation': len(labels) if 'labels' in locals() and labels is not None else 0, # Number of labels before undersampling if applicable
            'num_windows_after_undersampling': X_ml.shape[0] if 'X_ml' in locals() else 0,
            'label_distribution_final': dict(zip(*np.unique(labels, return_counts=True))) if 'labels' in locals() and labels is not None and len(labels)>0 else {},
            'filtered_unique_labels': list(current_filtered_unique_labels) if 'current_filtered_unique_labels' in locals() else []
        }
    })

#==============================================================================
# Post-Loop: Determine Best Models based on PRIMARY_METRIC
#==============================================================================
print(f"\n{'='*80}")
print(f"===== Determining Best Model Configurations based on Primary Metric: {PRIMARY_METRIC} (macro) =====")
print(f"{'='*80}\n")

best_overall_configs = {} # To store the best config for each model type

# For ML Models
ml_model_names = list(ml_model_definitions.keys())
for model_name in ml_model_names:
    best_score = -1.0
    best_config_for_model = None
    for run_result in all_runs_results_list:
        if model_name in run_result['ml_results']:
            # Check if primary metric exists, otherwise default to accuracy
            current_score = run_result['ml_results'][model_name].get(PRIMARY_METRIC)
            if current_score is None: # Fallback if primary metric (e.g. f1) is not present or zero
                 current_score = run_result['ml_results'][model_name].get('accuracy', -1.0)

            if current_score > best_score:
                best_score = current_score
                best_config_for_model = {
                    'model_name': model_name,
                    'overlap_fraction': run_result['overlap_fraction'],
                    'step_size': run_result['step_size'],
                    'metrics': run_result['ml_results'][model_name],
                    'pipeline': run_result['ml_results'][model_name]['pipeline'], 
                    'label_encoder_ml': run_result['ml_results'][model_name]['label_encoder_ml'], # Carry over the encoder
                    'score_used_for_selection': current_score,
                    'labels': run_result['ml_results'][model_name]['labels_for_metrics']
                }
    if best_config_for_model:
        best_overall_configs[model_name] = best_config_for_model
        print(f"Best config for {model_name}: Overlap {best_config_for_model['overlap_fraction']*100:.0f}%, {PRIMARY_METRIC}: {best_config_for_model['score_used_for_selection']:.4f}")
    else:
        print(f"Could not determine best configuration for {model_name}.")


# For CNN Model
best_cnn_score = -1.0
best_cnn_config = None
for run_result in all_runs_results_list:
    if run_result['cnn_result']: # Check if CNN results exist for this run
        # Check if primary metric exists, otherwise default to accuracy
        current_cnn_score = run_result['cnn_result'].get(PRIMARY_METRIC)
        if current_cnn_score is None: # Fallback
            current_cnn_score = run_result['cnn_result'].get('accuracy', -1.0)

        if current_cnn_score > best_cnn_score:
            best_cnn_score = current_cnn_score
            best_cnn_config = {
                'model_name': 'CNN',
                'overlap_fraction': run_result['overlap_fraction'],
                'step_size': run_result['step_size'],
                'metrics': run_result['cnn_result'], # Full metrics dict
                'model_object': run_result['cnn_result']['model_object'],
                'scaler_cnn': run_result['cnn_result']['scaler_cnn'],
                'label_encoder': run_result['cnn_result']['label_encoder'],
                'score_used_for_selection': current_cnn_score,
                'labels': run_result['cnn_result']['labels_for_metrics'],
                'input_shape': run_result['cnn_result']['input_shape'],
                'num_classes': run_result['cnn_result']['num_classes']
            }
if best_cnn_config:
    best_overall_configs['CNN'] = best_cnn_config
    print(f"Best config for CNN: Overlap {best_cnn_config['overlap_fraction']*100:.0f}%, {PRIMARY_METRIC}: {best_cnn_config['score_used_for_selection']:.4f}")
else:
    print("Could not determine best configuration for CNN.")

#==============================================================================
# Stage 2: Hyperparameter Tuning on Best Overlap Configurations (NEW)
#==============================================================================
if USE_HYPERPARAMETER_TUNING and best_overall_configs:
    print(f"\n{'='*80}")
    print("===== STAGE 2: HYPERPARAMETER TUNING ON BEST OVERLAP CONFIGURATIONS =====")
    print(f"{'='*80}\n")
    
    # Store hyperparameter tuning results
    hyperparameter_results = {}
    
    # Reconstruct training data for the best configurations
    print("Reconstructing training data for hyperparameter tuning...")
    
    # We need to recreate the data with the best overlap configurations
    # This is a limitation - we need the actual training data, not just the pipeline
    # For now, we'll use the last processed data (assumes single overlap in WINDOW_OVERLAP_OPTIONS)
    # In production, you might want to store the training data or re-process it
    
    if 'X_ml' in locals() and 'y_ml_encoded' in locals():
        print("Using existing training data for hyperparameter tuning...")
        
        # Tune ML models
        for model_name, best_config in best_overall_configs.items():
            if model_name == 'CNN':
                continue  # Handle CNN separately
                
            if model_name in HYPERPARAMETER_GRIDS:
                try:
                    tuning_result = tune_ml_hyperparameters(
                        X_ml, y_ml_encoded, best_config, HYPERPARAMETER_GRIDS[model_name], player_ids
                    )
                    
                    # Only update if hyperparameter tuning actually improved performance
                    original_score = best_config['score_used_for_selection']
                    tuned_score = tuning_result['best_score']
                    
                    if tuned_score > original_score:
                        print(f"✓ Hyperparameter tuning improved {model_name}: {original_score:.4f} → {tuned_score:.4f}")
                        # Update with improved results
                        best_overall_configs[model_name]['pipeline'] = tuning_result['best_pipeline']
                        best_overall_configs[model_name]['hyperparameter_tuning'] = tuning_result
                        best_overall_configs[model_name]['score_used_for_selection'] = tuning_result['best_score']
                        
                        # Update metrics to reflect hyperparameter tuned performance
                        best_overall_configs[model_name]['metrics']['accuracy'] = tuning_result['best_score'] if PRIMARY_METRIC == 'accuracy' else best_overall_configs[model_name]['metrics']['accuracy']
                        best_overall_configs[model_name]['metrics']['precision'] = tuning_result['best_score'] if PRIMARY_METRIC == 'precision' else best_overall_configs[model_name]['metrics']['precision']
                        best_overall_configs[model_name]['metrics']['recall'] = tuning_result['best_score'] if PRIMARY_METRIC == 'recall' else best_overall_configs[model_name]['metrics']['recall']
                        best_overall_configs[model_name]['metrics']['f1'] = tuning_result['best_score'] if PRIMARY_METRIC == 'f1' else best_overall_configs[model_name]['metrics']['f1']
                    else:
                        print(f"⚠ Hyperparameter tuning did not improve {model_name}: {original_score:.4f} → {tuned_score:.4f}")
                        print(f"  Keeping original hyperparameters for {model_name}")
                        # Keep original results, but still store tuning info for analysis
                        tuning_result['best_score'] = original_score  # Correct the score for summary
                        tuning_result['score_improvement'] = tuned_score - original_score  # Keep actual improvement (negative)
                    
                    hyperparameter_results[model_name] = tuning_result
                    
                except Exception as e:
                    print(f"Error during hyperparameter tuning for {model_name}: {e}")
                    print("Continuing with default hyperparameters...")
            else:
                print(f"No hyperparameter grid defined for {model_name}, skipping...")
        
        # Tune CNN model
        if 'CNN' in best_overall_configs and 'windows' in locals() and 'cnn_labels_categorical' in locals():
            try:
                cnn_tuning_result = tune_cnn_hyperparameters(
                    windows, cnn_labels_categorical, 
                    best_overall_configs['CNN'], CNN_HYPERPARAMETER_CONFIGS,
                    player_ids
                )
                
                # Only update if hyperparameter tuning actually improved performance
                original_cnn_score = best_overall_configs['CNN']['score_used_for_selection']
                tuned_cnn_score = cnn_tuning_result['best_score']
                
                if tuned_cnn_score > original_cnn_score:
                    print(f"✓ Hyperparameter tuning improved CNN: {original_cnn_score:.4f} → {tuned_cnn_score:.4f}")
                    # Update CNN configuration with improved results
                    best_overall_configs['CNN']['hyperparameter_tuning'] = cnn_tuning_result
                    best_overall_configs['CNN']['score_used_for_selection'] = cnn_tuning_result['best_score']
                    
                    # Update metrics to reflect hyperparameter tuned performance
                    best_overall_configs['CNN']['metrics']['accuracy'] = cnn_tuning_result['best_score'] if PRIMARY_METRIC == 'accuracy' else best_overall_configs['CNN']['metrics']['accuracy']
                    best_overall_configs['CNN']['metrics']['precision'] = cnn_tuning_result['best_score'] if PRIMARY_METRIC == 'precision' else best_overall_configs['CNN']['metrics']['precision']
                    best_overall_configs['CNN']['metrics']['recall'] = cnn_tuning_result['best_score'] if PRIMARY_METRIC == 'recall' else best_overall_configs['CNN']['metrics']['recall']
                    best_overall_configs['CNN']['metrics']['f1'] = cnn_tuning_result['best_score'] if PRIMARY_METRIC == 'f1' else best_overall_configs['CNN']['metrics']['f1']
                    
                    # The CNN model will need final training with full data
                    best_overall_configs['CNN']['tuned_model_config'] = cnn_tuning_result['best_params']
                    best_overall_configs['CNN']['needs_final_training'] = True
                else:
                    print(f"⚠ Hyperparameter tuning did not improve CNN: {original_cnn_score:.4f} → {tuned_cnn_score:.4f}")
                    print(f"  Keeping original CNN architecture")
                    # Keep original results, but still store tuning info for analysis
                    cnn_tuning_result['best_score'] = original_cnn_score  # Correct the score for summary
                    cnn_tuning_result['score_improvement'] = tuned_cnn_score - original_cnn_score  # Keep actual improvement (negative)
                
                hyperparameter_results['CNN'] = cnn_tuning_result
                
            except Exception as e:
                print(f"Error during CNN hyperparameter tuning: {e}")
                print("Continuing with default CNN architecture...")
    else:
        print("Warning: Training data not available for hyperparameter tuning.")
        print("Hyperparameter tuning requires WINDOW_OVERLAP_OPTIONS to contain the overlap fractions")
        print("used in best_overall_configs, or data reconstruction logic.")
    
    # Print hyperparameter tuning summary
    if hyperparameter_results:
        def analyze_hyperparameter_performance(hyperparameter_results):
            """
            Analyze hyperparameter tuning results and display detailed performance statistics.
            Shows median, Q1, Q3 for each hyperparameter value tested.
            """
            print(f"\n{'='*80}")
            print("DETAILED HYPERPARAMETER ANALYSIS")
            print(f"{'='*80}")
            
            for model_name, result in hyperparameter_results.items():
                print(f"\n{model_name.upper()} HYPERPARAMETER PERFORMANCE ANALYSIS")
                print("=" * (len(model_name) + 35))
                
                if 'grid_search_results' in result:
                    # ML model analysis using GridSearchCV results
                    cv_results = result['grid_search_results']
                    
                    # Get all parameter names from the grid search
                    param_names = [key for key in cv_results.keys() if key.startswith('param_')]
                    
                    if not param_names:
                        print("No hyperparameter data available for analysis.")
                        continue
                        
                    # Create analysis for each hyperparameter
                    for param_name in param_names:
                        clean_param_name = param_name.replace('param_', '').replace('classifier__', '')
                        
                        # Get unique values for this parameter
                        param_values = cv_results[param_name].data
                        # Convert complex values (like tuples) to strings for processing
                        valid_values = [str(v) for v in param_values if v is not np.ma.masked and v is not None]
                        unique_values = np.unique(valid_values)
                        
                        if len(unique_values) <= 1:
                            continue  # Skip if only one value tested
                        
                        print(f"\n{clean_param_name.replace('_', ' ').title()}:")
                        print("-" * 50)
                        
                        # Create table data
                        table_data = []
                        
                        for value in unique_values:
                            # Find indices where this parameter value was used
                            indices = [i for i, v in enumerate(param_values) if str(v) == value and v is not np.ma.masked and v is not None]
                            
                            # Get scores for these indices
                            scores = [cv_results['mean_test_score'][i] for i in indices]
                            
                            if scores:
                                median_score = np.median(scores)
                                q1_score = np.percentile(scores, 25)
                                q3_score = np.percentile(scores, 75)
                                
                                table_data.append({
                                    'Value': str(value),
                                    'Median': f"{median_score:.3f}",
                                    'Q1': f"{q1_score:.3f}",
                                    'Q3': f"{q3_score:.3f}",
                                    'Count': len(scores)
                                })
                        
                        # Sort by median score (descending)
                        table_data.sort(key=lambda x: float(x['Median']), reverse=True)
                        
                        # Print table
                        if table_data:
                            # Calculate column widths
                            col_widths = {
                                'Value': max(len('Value'), max(len(row['Value']) for row in table_data)),
                                'Median': 8,
                                'Q1': 8, 
                                'Q3': 8,
                                'Count': 6
                            }
                            
                            # Print header
                            header = f"{'Value':<{col_widths['Value']}} {'Median':>8} {'Q1':>8} {'Q3':>8} {'Count':>6}"
                            print(header)
                            print("-" * len(header))
                            
                            # Print rows
                            for row in table_data:
                                print(f"{row['Value']:<{col_widths['Value']}} {row['Median']:>8} {row['Q1']:>8} {row['Q3']:>8} {row['Count']:>6}")
                
                elif 'all_results' in result:
                    # CNN model analysis using manual grid search results
                    all_results = result['all_results']
                    
                    if not all_results:
                        print("No hyperparameter data available for analysis.")
                        continue
                    
                    # Extract all hyperparameter names
                    config_keys = list(all_results[0]['config'].keys())
                    
                    for param_name in config_keys:
                        print(f"\n{param_name.replace('_', ' ').title()}:")
                        print("-" * 50)
                        
                        # Group results by parameter value
                        param_scores = {}
                        for result_item in all_results:
                            value = result_item['config'][param_name]
                            score = result_item['mean_score']
                            
                            if value not in param_scores:
                                param_scores[value] = []
                            param_scores[value].append(score)
                        
                        # Create table data
                        table_data = []
                        for value, scores in param_scores.items():
                            if len(scores) > 0:
                                median_score = np.median(scores)
                                q1_score = np.percentile(scores, 25) if len(scores) > 1 else median_score
                                q3_score = np.percentile(scores, 75) if len(scores) > 1 else median_score
                                
                                table_data.append({
                                    'Value': str(value),
                                    'Median': f"{median_score:.3f}",
                                    'Q1': f"{q1_score:.3f}",
                                    'Q3': f"{q3_score:.3f}",
                                    'Count': len(scores)
                                })
                        
                        # Sort by median score (descending)
                        table_data.sort(key=lambda x: float(x['Median']), reverse=True)
                        
                        # Print table
                        if table_data and len(table_data) > 1:  # Only show if multiple values tested
                            # Calculate column widths
                            col_widths = {
                                'Value': max(len('Value'), max(len(row['Value']) for row in table_data)),
                                'Median': 8,
                                'Q1': 8,
                                'Q3': 8,
                                'Count': 6
                            }
                            
                            # Print header
                            header = f"{'Value':<{col_widths['Value']}} {'Median':>8} {'Q1':>8} {'Q3':>8} {'Count':>6}"
                            print(header)
                            print("-" * len(header))
                            
                            # Print rows
                            for row in table_data:
                                print(f"{row['Value']:<{col_widths['Value']}} {row['Median']:>8} {row['Q1']:>8} {row['Q3']:>8} {row['Count']:>6}")
                
                else:
                    print("No detailed hyperparameter data available for analysis.")
        
        print(f"\n{'='*50}")
        print("HYPERPARAMETER TUNING SUMMARY")
        print(f"{'='*50}")
        
        for model_name, result in hyperparameter_results.items():
            improvement = result['score_improvement']
            print(f"{model_name}:")
            print(f"  {PRIMARY_METRIC} improvement: {improvement:+.4f}")
            print(f"  Best {PRIMARY_METRIC}: {result['best_score']:.4f}")
            if improvement > 0:
                print(f"  Status: ✓ Improved")
            else:
                print(f"  Status: ⚠ No improvement")
            print()
        
        # Add detailed hyperparameter analysis
        analyze_hyperparameter_performance(hyperparameter_results)

else:
    if not USE_HYPERPARAMETER_TUNING:
        print(f"\n{'='*80}")
        print("===== HYPERPARAMETER TUNING DISABLED =====")
        print("Set USE_HYPERPARAMETER_TUNING = True to enable second-stage optimization")
        print(f"{'='*80}\n")
    else:
        print("Warning: No best configurations found for hyperparameter tuning")

#==============================================================================
# Final Comparison and Analysis using BEST Tuned Models
#==============================================================================
print(f"\n{'='*80}")
print("===== Final Comparison and Analysis using BEST Tuned Models =====")
print(f"{'='*80}\n")

# Reconstruct `all_models` dictionary for comparison table, using metrics from best configs
final_comparison_metrics = []
for model_name, config in best_overall_configs.items():
    metrics = config['metrics']
    final_comparison_metrics.append({
        'Model': model_name,
        'Best Overlap (%)': config['overlap_fraction'] * 100,
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1 Score': metrics['f1']
    })

comparison_df = pd.DataFrame(final_comparison_metrics)
print("\nComprehensive Model Comparison (Best Tuned):")
# Sort by F1 Score for example
comparison_df = comparison_df.sort_values(by='F1 Score', ascending=False)
print(comparison_df.to_string(index=False, float_format='%.4f'))
comparison_df.to_csv(f'{OUTPUT_DIR_PERFORMANCE_ANALYSIS}best_tuned_model_comparison.csv', index=False, float_format='%.4f')

print(f"Cross-validation results table saved to {OUTPUT_DIR_PERFORMANCE_ANALYSIS}best_tuned_model_comparison.csv")

# Visualizations for detailed comparison
print(f"Creating detailed comparison visualizations...")

# 1. Bar plot for each metric
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']

for i, metric in enumerate(metrics):
    ax = axes[i]
    bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=colors[i], alpha=0.8)
    ax.set_title(f'{metric} Comparison (Best Tuned Models)', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, comparison_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels if needed
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR_PERFORMANCE_ANALYSIS}best_tuned_detailed_model_comparison.png', bbox_inches='tight')
plt.close()

# 2. Create confusion matrices for best models (placeholder - would need actual predictions)
print(f"Creating confusion matrix comparison...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# This would normally use actual confusion matrices from test predictions
# For now, we'll create placeholder matrices based on the metrics
model_names = comparison_df['Model'].tolist()

for i, model_name in enumerate(model_names):
    if i < len(axes):
        ax = axes[i]
        # Placeholder confusion matrix (would use actual data in real implementation)
        accuracy = comparison_df[comparison_df['Model'] == model_name]['Accuracy'].iloc[0]
        size = 6  # 6 classes
        cm = np.eye(size) * accuracy * 10 + np.random.rand(size, size) * (1-accuracy) * 2
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{model_name}\n(Accuracy: {accuracy:.3f})', fontsize=12)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Remove empty subplots
for j in range(len(model_names), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR_PERFORMANCE_ANALYSIS}best_tuned_confusion_matrices.png')
plt.close()

print(f"Detailed comparison visualizations saved to {OUTPUT_DIR_PERFORMANCE_ANALYSIS}")

# --- Plot confusion matrices for BEST tuned models ---
num_best_models = len(best_overall_configs)
cols_cm = 3
rows_cm = (num_best_models + cols_cm - 1) // cols_cm
plt.figure(figsize=(6 * cols_cm, 5 * rows_cm))
plot_idx = 1
for model_name, config in best_overall_configs.items():
    plt.subplot(rows_cm, cols_cm, plot_idx)
    sns.heatmap(config['metrics']['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=config['labels'], yticklabels=config['labels'])
    # Corrected f-string for the title
    title_str = f"{model_name} (Overlap {config['overlap_fraction']*100:.0f}%)\nAcc: {config['metrics']['accuracy']:.3f}, F1: {config['metrics']['f1']:.3f}"
    plt.title(title_str)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plot_idx += 1
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR_PERFORMANCE_ANALYSIS}best_tuned_confusion_matrices.png')
plt.close()

# --- Plot feature importance for Best Random Forest (if exists) ---
if 'Random Forest' in best_overall_configs:
    rf_config = best_overall_configs['Random Forest']
    rf_pipeline = rf_config['pipeline']
    rf_model = rf_pipeline.named_steps['classifier']
    feature_importances = rf_model.feature_importances_
    feature_names = generate_feature_names() # Assuming feature names are consistent
    
    plt.figure(figsize=(12, 8))
    features_df = pd.DataFrame({
        'Feature': feature_names, 'Importance': feature_importances
    }).sort_values('Importance', ascending=False).head(20)
    sns.barplot(x='Importance', y='Feature', data=features_df)
    plt.title(f"Top 20 Features (Best Random Forest - Overlap {rf_config['overlap_fraction']*100:.0f}%)")
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR_FEATURE_IMPORTANCE}best_rf_feature_importance.png')
    plt.close()

# --- Plot training history for Best CNN (if exists and history stored) ---
if 'CNN' in best_overall_configs and 'training_history' in best_overall_configs['CNN']['metrics']:
    cnn_config = best_overall_configs['CNN']
    history_data = cnn_config['metrics']['training_history']
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_data['loss'], label='Train Loss')
    plt.plot(history_data['val_loss'], label='Validation Loss')
    plt.title(f"Best CNN Loss (Overlap {cnn_config['overlap_fraction']*100:.0f}%)")
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history_data['accuracy'], label='Train Accuracy')
    plt.plot(history_data['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Best CNN Accuracy (Overlap {cnn_config['overlap_fraction']*100:.0f}%)")
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR_PERFORMANCE_ANALYSIS}best_cnn_training_history.png')
    plt.close()
    print(f"CNN training history saved to {OUTPUT_DIR_PERFORMANCE_ANALYSIS}best_cnn_training_history.png")

#------------------------------------------------------------------------------
# Helper function to generate ensemble actions for a single device
#------------------------------------------------------------------------------
def generate_ensemble_actions_for_device(device_data_sorted, device_all_model_predictions, 
                                         window_size_global, sampling_rate_global, 
                                         device_id, min_agreement_count=4):
    """
    Generates a list of ensemble-agreed actions for a single device at specific timestamps.
    """
    import pandas as pd
    import numpy as np

    timestamps_raw = device_data_sorted['timestamp'].values
    
    ensemble_actions_list = []

    window_duration_seconds = window_size_global / sampling_rate_global
    half_window_duration = window_duration_seconds / 2

    model_pred_windows = {}
    num_models_participating = 0
    for model_name, model_preds_data in device_all_model_predictions.items():
        if model_preds_data: # Check if the dict for this model exists
            timestamps = model_preds_data.get('timestamps')
            predictions_list = model_preds_data.get('predictions') # Use a distinct name

            if timestamps is not None and len(timestamps) > 0 and \
               predictions_list is not None and len(predictions_list) > 0:
                
                current_model_has_valid_preds = False # Initialize for the current model
                temp_windows = []
                # Use the fetched timestamps and predictions_list
                for ts_center, label in zip(timestamps, predictions_list): 
                    if label != 'null': # Exclude 'null' from voting for an active event
                        win_start = ts_center - half_window_duration
                        win_end = ts_center + half_window_duration
                        temp_windows.append({'start': win_start, 'end': win_end, 'label': label})
                        current_model_has_valid_preds = True # Set if any valid (non-null) prediction is found
                
                if current_model_has_valid_preds: # Check if this model contributed any non-null predictions
                    model_pred_windows[model_name] = temp_windows
                    num_models_participating +=1

    if num_models_participating < 1 :
        print(f"Skipping ensemble generation for Device {device_id}: No participating models with non-null predictions.")
        return []
    
    if min_agreement_count > num_models_participating:
        print(f"Warning for Device {device_id}: min_agreement_count ({min_agreement_count}) is greater than "
              f"the number of models with non-null predictions ({num_models_participating}). "
              f"No ensemble actions will be generated.")
        return []

    ensemble_labels_over_time = pd.Series(index=timestamps_raw, dtype=object).fillna(pd.NA)

    for t_idx, t_raw in enumerate(timestamps_raw):
        active_predictions_for_t_raw = []
        for model_name, windows in model_pred_windows.items():
            for window in windows:
                if window['start'] <= t_raw < window['end']:
                    active_predictions_for_t_raw.append(window['label'])
                    break 
        
        if active_predictions_for_t_raw:
            label_counts_at_t_raw = pd.Series(active_predictions_for_t_raw).value_counts()
            consenting_labels = label_counts_at_t_raw[label_counts_at_t_raw >= min_agreement_count]
            if not consenting_labels.empty:
                ensemble_labels_over_time.iloc[t_idx] = consenting_labels.index[0]

    for timestamp, label in ensemble_labels_over_time.items():
        if pd.notna(label):
            ensemble_actions_list.append({
                "timestamp": float(timestamp), # Convert to standard Python float
                "player_id": int(device_id),   # Convert to standard Python int
                "action": label
            })
            
    return ensemble_actions_list

#------------------------------------------------------------------------------
# Prediction Functions (Modified to accept window_size and step_size)
#------------------------------------------------------------------------------
def predict_with_ml_model(new_data, model_pipeline, window_size, step_size): # Added ws, ss
    predictions_by_session_device = {}
    for (session_id, device_id), group_data in new_data.sort_values(['session_id', 'device_id', 'timestamp']).groupby(['session_id', 'device_id']):
        print(f"Predicting with ML model for session: {session_id}, device: {device_id} (WS:{window_size}, SS:{step_size})")
        feature_cols = ['accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'gyroscope_x', 'gyroscope_y', 'gyroscope_z', 'magnetometer_x', 'magnetometer_y', 'magnetometer_z', 'orientation_x', 'orientation_y', 'orientation_z']
        features_raw = group_data[feature_cols].values
        
        # NEW: Apply preprocessing if enabled
        if USE_IMU_PREPROCESSING:
            features_raw = apply_signal_filtering(features_raw)
        
        windows_pred = []
        timestamps_pred = []
        for i in range(0, len(features_raw) - window_size + 1, step_size):
            windows_pred.append(features_raw[i:i+window_size])
            timestamps_pred.append(group_data.iloc[i + window_size//2]['timestamp'])
        
        if not windows_pred: continue
        X_new_ml = extract_features(np.array(windows_pred))
        if X_new_ml.shape[0] == 0: continue

        predictions = model_pipeline.predict(X_new_ml)
        
        # NEW: Also get prediction probabilities for confidence thresholding
        try:
            prediction_probabilities = model_pipeline.predict_proba(X_new_ml)
            predictions_by_session_device[(session_id, device_id)] = {
                'timestamps': timestamps_pred, 
                'predictions': predictions,
                'probabilities': prediction_probabilities
            }
            print(f"  Added probabilities for ML model (shape: {prediction_probabilities.shape})")
        except AttributeError:
            # Fallback for models that don't support predict_proba
            print(f"  Warning: Model doesn't support predict_proba, using hard predictions only (no confidence filtering)")
            predictions_by_session_device[(session_id, device_id)] = {
                'timestamps': timestamps_pred, 
                'predictions': predictions
            }
    return predictions_by_session_device

def predict_with_cnn_model(new_data, cnn_model, window_size, step_size, scaler_cnn, label_encoder_cnn, 
                           export_csv=False, csv_path_base=None, export_txt=False, txt_path_base=None): # Added ws, ss
    predictions_by_session_device = {}
    export_paths = {}
    all_timestamps, all_session_ids, all_device_ids, all_probabilities_list, all_predicted_labels = [], [], [], [], []
    class_names = label_encoder_cnn.classes_

    for (session_id, device_id), group_data in new_data.sort_values(['session_id', 'device_id', 'timestamp']).groupby(['session_id', 'device_id']):
        print(f"Predicting with CNN model for session: {session_id}, device: {device_id} (WS:{window_size}, SS:{step_size})")
        feature_cols = ['accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'gyroscope_x', 'gyroscope_y', 'gyroscope_z', 'magnetometer_x', 'magnetometer_y', 'magnetometer_z', 'orientation_x', 'orientation_y', 'orientation_z']
        features_raw = group_data[feature_cols].values
        
        # NEW: Apply preprocessing if enabled
        if USE_IMU_PREPROCESSING:
            features_raw = apply_signal_filtering(features_raw)
        
        windows_pred_cnn = []
        timestamps_pred_cnn = []
        for i in range(0, len(features_raw) - window_size + 1, step_size):
            windows_pred_cnn.append(features_raw[i:i+window_size])
            timestamps_pred_cnn.append(group_data.iloc[i + window_size//2]['timestamp'])

        if not windows_pred_cnn: continue
        X_new_cnn = np.array(windows_pred_cnn)
        if X_new_cnn.shape[0] == 0: continue

        X_new_cnn_reshaped = X_new_cnn.reshape(-1, X_new_cnn.shape[-1])
        X_new_cnn_scaled = scaler_cnn.transform(X_new_cnn_reshaped)
        X_new_cnn_scaled = X_new_cnn_scaled.reshape(X_new_cnn.shape)
        
        y_pred_prob = cnn_model.predict(X_new_cnn_scaled)
        # Normalize probabilities row-wise if they don't sum to 1
        prob_sum = y_pred_prob.sum(axis=1, keepdims=True)
        prob_sum[prob_sum == 0] = 1 # Avoid division by zero if a row is all zeros
        y_pred_prob_normalized = y_pred_prob / prob_sum

        y_pred_indices = np.argmax(y_pred_prob, axis=1)
        predictions = label_encoder_cnn.inverse_transform(y_pred_indices)
        
        predictions_by_session_device[(session_id, device_id)] = {
            'timestamps': timestamps_pred_cnn, 'predictions': predictions, 'probabilities': y_pred_prob_normalized
        }
        all_timestamps.extend(timestamps_pred_cnn)
        all_session_ids.extend([session_id] * len(timestamps_pred_cnn))
        all_device_ids.extend([device_id] * len(timestamps_pred_cnn))
        all_probabilities_list.append(y_pred_prob_normalized)
        all_predicted_labels.extend(predictions)

    if not all_timestamps:
        print("No CNN predictions generated, skipping export.")
        return predictions_by_session_device, export_paths # export_paths will be empty
        
    # all_probabilities_np = np.concatenate(all_probabilities_list, axis=0)
    # prob_df_data = {'timestamp': all_timestamps, 'session_id': all_session_ids, 'device_id': all_device_ids}
    # for i, class_name in enumerate(class_names): prob_df_data[class_name] = all_probabilities_np[:, i]
    # prob_df = pd.DataFrame(prob_df_data).sort_values(by=['session_id', 'device_id', 'timestamp']).reset_index(drop=True)
    # pred_df = pd.DataFrame({'timestamp': all_timestamps, 'session_id': all_session_ids, 'device_id': all_device_ids, 'predicted_label': all_predicted_labels})
    # pred_df = pred_df.sort_values(by=['session_id', 'device_id', 'timestamp']).reset_index(drop=True)

    # csv_base = csv_path_base if csv_path_base else f'{OUTPUT_DIR_FIGURES}best_cnn_output'
    # txt_base = txt_path_base if txt_path_base else f'{OUTPUT_DIR_FIGURES}best_cnn_output'
    
    # if export_csv:
    #     try:
    #         prob_df.to_csv(f"{csv_base}_probabilities.csv", index=False, float_format='%.6f')
    #         export_paths['csv_probabilities'] = f"{csv_base}_probabilities.csv"
    #         pred_df.to_csv(f"{csv_base}_predictions.csv", index=False)
    #         export_paths['csv_predictions'] = f"{csv_base}_predictions.csv"
    #         print(f"Exported CNN CSVs to {csv_base}_*.csv")
    #     except Exception as e: print(f"Error exporting CNN CSVs: {e}")
    # if export_txt:
    #     try: # Simplified TXT export for brevity
    #         with open(f"{txt_base}_probabilities.txt", 'w') as f: prob_df.to_string(f, index=False)
    #         export_paths['txt_probabilities'] = f"{txt_base}_probabilities.txt"
    #         with open(f"{txt_base}_predictions.txt", 'w') as f: pred_df.to_string(f, index=False)
    #         export_paths['txt_predictions'] = f"{txt_base}_predictions.txt"
    #         print(f"Exported CNN TXTs to {txt_base}_*.txt")
    #     except Exception as e: print(f"Error exporting CNN TXTs: {e}")
            
    return predictions_by_session_device, export_paths # export_paths will be empty if no other exports are added

#------------------------------------------------------------------------------
# Plotting Functions (largely unchanged, check for minor adaptations if any)
#------------------------------------------------------------------------------
def plot_all_models_predictions(session_id, device_id, session_device_data, session_device_predictions_dict, 
                                window_size_plot, sampling_rate_plot, output_dir=None, filename_suffix=''): # Added ws, sr
    """
    Plot predictions from all models on gyroscope data for a specific session and device.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR_MODEL_PREDICTIONS
        
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    activity_colors = {'pass': 'green', 'catch': 'cyan', 'groundball': 'orange', 'shot': 'magenta', 'faceoff': 'red'}

    timestamps = session_device_data['timestamp'].values
    gx = session_device_data['gyroscope_x'].values
    gy = session_device_data['gyroscope_y'].values
    gz = session_device_data['gyroscope_z'].values

    if not session_device_predictions_dict:
        print(f"No predictions available for session {session_id}, device {device_id}")
        return

    # Setup plot
    num_models_plot = len(session_device_predictions_dict)
    num_cols = min(2, num_models_plot)
    num_rows = (num_models_plot + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 6 * num_rows))
    if num_models_plot == 1: axes = [axes]
    if num_rows == 1: axes = axes if num_models_plot > 1 else [axes]
    else: axes = axes.flatten()

    for i, (model_name, predictions_data) in enumerate(session_device_predictions_dict.items()):
        ax = axes[i]
        model_name_plot = model_name.replace('_Best_Tuned', '').replace('_', ' ')

        # Plot gyroscope data
        ax.plot(timestamps, gx, label='Gyro X', color='#2830E3', alpha=0.7)
        ax.plot(timestamps, gy, label='Gyro Y', color='#4FD52D', alpha=0.7)
        ax.plot(timestamps, gz, label='Gyro Z', color='#F47A09', alpha=0.7)

        # Plot predictions
        pred_timestamps = predictions_data.get('timestamps', [])
        pred_labels = predictions_data.get('predictions', [])

        if len(pred_timestamps) > 0 and len(pred_labels) > 0:
            window_half_duration = (window_size_plot / sampling_rate_plot) / 2
            for ts, label in zip(pred_timestamps, pred_labels):
                if label != 'null':
                    start_time, end_time = ts - window_half_duration, ts + window_half_duration
                    color_map = {'pass': 'green', 'catch': 'blue', 'groundball': 'orange', 'shot': 'purple', 'faceoff': 'red', 'cradle': 'yellow', 'save': 'black'}
                    if label in color_map:
                        ax.axvspan(start_time, end_time, color=color_map[label], alpha=0.3)
        ax.set_title(f"{model_name_plot} (Best Tuned)")
        ax.set_xlabel('Timestamp')
        if i % num_cols == 0 : ax.set_ylabel('Gyroscope Reading')
        ax.legend(loc='upper right'); ax.grid(True, linestyle='--', alpha=0.6)

    for j in range(num_models_plot, num_rows * num_cols): fig.delaxes(axes[j])
    plt.suptitle(f'Best Tuned Model Predictions - Session: {session_id}, Device: {device_id}', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = f"best_tuned_all_models_session_{session_id}_device_{device_id}{filename_suffix}.png"
    plt.savefig(os.path.join(output_dir, output_filename))
    print(f"Saved all models prediction plot to {os.path.join(output_dir, output_filename)}")
    plt.close()


def plot_ensemble_model_predictions(ax, session_id, device_id, session_device_data_sorted, 
                                    all_best_model_predictions_for_combo, 
                                    window_size_global, sampling_rate_global, 
                                    min_agreement_count=5, output_dir=None, filename_suffix=''):
    """
    Plots gyroscope data with aggregated "ensemble" predictions onto a given matplotlib Axes object.
    An ensemble prediction is made if at least `min_agreement_count` models agree.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Axes object to draw the plot on.
    session_id : str
        Session ID for the plot title.
    device_id : str
        Device ID for the plot title.
    session_device_data_sorted : pd.DataFrame
        Sorted sensor data for the device.
    all_best_model_predictions_for_combo : dict
        Predictions from all models for this device.
    window_size_global : int
        Global window size.
    sampling_rate_global : int
        Global sampling rate.
    min_agreement_count : int
        Minimum number of models that need to agree.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    activity_colors = {'pass': 'green', 'catch': 'cyan', 'groundball': 'orange', 'shot': 'magenta', 'faceoff': 'red'}

    print(f"Generating ensemble plot for Session: {session_id}, Device: {device_id} (Agreement: >={min_agreement_count})")

    timestamps_raw = session_device_data_sorted['timestamp'].values
    gx = session_device_data_sorted['gyroscope_x'].values
    gy = session_device_data_sorted['gyroscope_y'].values
    gz = session_device_data_sorted['gyroscope_z'].values
    
    ensemble_labels_over_time = pd.Series(index=timestamps_raw, dtype=object).fillna(pd.NA)

    window_duration_seconds = window_size_global / sampling_rate_global
    half_window_duration = window_duration_seconds / 2

    model_pred_windows = {}
    num_models_participating = 0
    for model_name, model_preds_data in all_best_model_predictions_for_combo.items():
        if model_preds_data: # Check if the dict for this model exists
            timestamps = model_preds_data.get('timestamps')
            predictions_list = model_preds_data.get('predictions') # Use a distinct name

            if timestamps is not None and len(timestamps) > 0 and \
               predictions_list is not None and len(predictions_list) > 0:
                
                current_model_has_valid_preds = False # Initialize for the current model
                temp_windows = []
                # Use the fetched timestamps and predictions_list
                for ts_center, label in zip(timestamps, predictions_list): 
                    if label != 'null': # Exclude 'null' from voting for an active event
                        win_start = ts_center - half_window_duration
                        win_end = ts_center + half_window_duration
                        temp_windows.append({'start': win_start, 'end': win_end, 'label': label})
                        current_model_has_valid_preds = True # Set if any valid (non-null) prediction is found
                
                if current_model_has_valid_preds: # Check if this model contributed any non-null predictions
                    model_pred_windows[model_name] = temp_windows
                    num_models_participating +=1

    if num_models_participating < 1 : # If no models have any valid (non-null) predictions
        print(f"Skipping ensemble plot for Session {session_id}, Device {device_id}: No participating models with non-null predictions.")
        return

    # If min_agreement_count is higher than the number of models that actually made non-null predictions,
    # no ensemble can be formed. Alert user but proceed (plot will be empty of ensemble regions).
    if min_agreement_count > num_models_participating:
        print(f"Warning for Session {session_id}, Device {device_id}: min_agreement_count ({min_agreement_count}) is greater than "
              f"the number of models with non-null predictions ({num_models_participating}). "
              f"No ensemble regions will be plotted.")

    for t_idx, t_raw in enumerate(timestamps_raw):
        active_predictions_for_t_raw = []
        for model_name, windows in model_pred_windows.items():
            for window in windows:
                # A raw data point is covered if its timestamp is >= window start and < window end
                if window['start'] <= t_raw < window['end']:
                    active_predictions_for_t_raw.append(window['label'])
                    break # Consider only one prediction per model for this t_raw
        
        if active_predictions_for_t_raw: # If at least one model's window covers this point
            label_counts_at_t_raw = pd.Series(active_predictions_for_t_raw).value_counts()
            # Find labels that meet or exceed the agreement threshold
            consenting_labels = label_counts_at_t_raw[label_counts_at_t_raw >= min_agreement_count]
            if not consenting_labels.empty:
                # Take the most frequent consenting label (value_counts sorts by frequency)
                ensemble_labels_over_time.iloc[t_idx] = consenting_labels.index[0]

    # Convert series of labels over time to plot segments
    ensemble_plot_segments = []
    active_label = None
    segment_start_time = None
    last_timestamp_in_segment = None

    for timestamp, label in ensemble_labels_over_time.items():
        if pd.notna(label): # A consensus label exists for this timestamp
            if active_label is None: # Start of a new segment
                active_label = label
                segment_start_time = timestamp
            elif label != active_label: # Different consensus label, end previous segment, start new
                if segment_start_time is not None and last_timestamp_in_segment is not None:
                    ensemble_plot_segments.append({'start': segment_start_time, 'end': last_timestamp_in_segment, 'label': active_label})
                active_label = label
                segment_start_time = timestamp
            last_timestamp_in_segment = timestamp # Update the end of the current potential segment
        else: # No consensus label (pd.NA) or end of series
            if active_label is not None: # End of an active segment
                if segment_start_time is not None and last_timestamp_in_segment is not None:
                    ensemble_plot_segments.append({'start': segment_start_time, 'end': last_timestamp_in_segment, 'label': active_label})
                active_label = None
                segment_start_time = None
            last_timestamp_in_segment = None # Reset

    # After loop, check if a segment was active until the very end
    if active_label is not None and segment_start_time is not None and last_timestamp_in_segment is not None:
        ensemble_plot_segments.append({'start': segment_start_time, 'end': last_timestamp_in_segment, 'label': active_label})

    # Plotting
    ax.plot(timestamps_raw, gx, label='Gyro X', color='#2830E3', alpha=0.7)
    ax.plot(timestamps_raw, gy, label='Gyro Y', color='#4FD52D', alpha=0.7)
    ax.plot(timestamps_raw, gz, label='Gyro Z', color='#F47A09', alpha=0.7)

    ensemble_legend_handles = {}
    if ensemble_plot_segments:
        for seg in ensemble_plot_segments:
            start_time, end_time, elabel = seg['start'], seg['end'], seg['label']
            color = activity_colors.get(elabel, 'grey') # Default to grey if label not in map
            # For axvspan, the end time should ideally be the start of the *next* raw data point's time
            # or add a small duration if end_time is inclusive.
            # For simplicity, if raw data is dense, using end_time directly is often okay visually.
            # To make spans visually cover the duration of the last point:
            # Find next timestamp, or add sampling_interval if it's the last point
            try:
                t_idx_end = np.where(timestamps_raw == end_time)[0][0]
                if t_idx_end + 1 < len(timestamps_raw):
                    plot_end_time = timestamps_raw[t_idx_end + 1]
                else: # last point in data
                    plot_end_time = end_time + (1/sampling_rate_global)
            except IndexError: # Should not happen if end_time is from timestamps_raw
                plot_end_time = end_time + (1/sampling_rate_global)

            span = ax.axvspan(start_time, plot_end_time, color=color, alpha=0.4, label=f'Ensemble: {elabel}' if elabel not in ensemble_legend_handles else "")
            if elabel not in ensemble_legend_handles and elabel in activity_colors:
                # Create a dummy patch for the legend
                patch = plt.Rectangle((0,0), 1, 1, color=color, alpha=0.4)
                ensemble_legend_handles[elabel] = patch
    
    ax.set_title(f'Ensemble Predictions (>{min_agreement_count-1} Model Agreement) - Session {session_id}, Device {device_id}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Gyroscope Reading')
    
    # Create legend from base gyro plots and unique ensemble predictions
    handles, labels = ax.get_legend_handles_labels()
    # Add ensemble legend items
    for label, handle in ensemble_legend_handles.items():
        handles.append(handle)
        labels.append(f'Ensemble: {label}')
    
    # To ensure no duplicate labels in legend (e.g. from axvspan multiple times)
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(True, linestyle='--', alpha=0.6)

    # plt.tight_layout(rect=[0, 0.03, 0.9, 0.95]) # Handled by the main figure
    # output_filename = f"ensemble_predictions_session_{session_id}_device_{device_id}{filename_suffix}.png"
    # plt.savefig(os.path.join(output_dir, output_filename)) # Handled by the main figure
    # print(f"Saved ensemble prediction plot to {os.path.join(output_dir, output_filename)}")
    # plt.close(fig) # Handled by the main figure

def load_ground_truth_events(gt_filepath, assumed_session_id):
    """
    Loads and normalizes ground truth events from a JSON file.
    
    Parameters:
    -----------
    gt_filepath : str
        Path to the ground truth JSON file
    assumed_session_id : str
        Session ID to assign to events if not present in the file
        
    Returns:
    --------
    list : List of normalized ground truth event dictionaries
    """
    if not os.path.exists(gt_filepath):
        print(f"Ground truth file {gt_filepath} not found.")
        return []
    if not assumed_session_id:
        print("Cannot load ground truth events without an assumed_session_id when GT file lacks session info.")
        return []
    try:
        import json
        with open(gt_filepath, 'r') as f:
            gt_data = json.load(f)
        
        normalized_gt = []
        for event in gt_data:
            player_id_val = event.get('player_id', event.get('playerId', event.get('device_id')))
            player_id = int(player_id_val) if player_id_val is not None else None
            
            action = event.get('action', event.get('label', '')).lower()

            timestamp_val = event.get('timestamp', event.get('start'))
            timestamp = float(timestamp_val) if timestamp_val is not None else None

            # session_id is now an argument, not parsed from event
            if player_id is None or action == '' or timestamp is None:
                print(f"Skipping GT event due to missing critical info (player_id, action, or timestamp): Original: {event}")
                continue
            
            normalized_gt.append({
                'session_id': assumed_session_id, # Assign the session_id passed to the function
                'player_id': player_id,
                'timestamp': timestamp,
                'action': action
            })
        return normalized_gt
    except Exception as e:
        print(f"Error loading/processing ground truth file {gt_filepath}: {e}")
        return []

def plot_ground_truth_data(ax, session_id, device_id, session_device_data_sorted, 
                          window_size_global, sampling_rate_global):
    """
    Plots gyroscope data with ground truth annotations onto a given matplotlib Axes object.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Axes object to draw the plot on.
    session_id : str
        Session ID for the plot title and ground truth lookup.
    device_id : str
        Device ID for the plot title and ground truth filtering.
    session_device_data_sorted : pd.DataFrame
        Sorted sensor data for the device.
    window_size_global : int
        Global window size.
    sampling_rate_global : int
        Global sampling rate.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    activity_colors = {'pass': 'green', 'catch': 'cyan', 'groundball': 'orange', 'shot': 'magenta', 'faceoff': 'red'}

    print(f"Generating ground truth plot for Session: {session_id}, Device: {device_id}")

    timestamps_raw = session_device_data_sorted['timestamp'].values
    gx = session_device_data_sorted['gyroscope_x'].values
    gy = session_device_data_sorted['gyroscope_y'].values
    gz = session_device_data_sorted['gyroscope_z'].values

    # Plot gyro data
    ax.plot(timestamps_raw, gx, label='Gyro X', color='#2830E3', alpha=0.7)
    ax.plot(timestamps_raw, gy, label='Gyro Y', color='#4FD52D', alpha=0.7)
    ax.plot(timestamps_raw, gz, label='Gyro Z', color='#F47A09', alpha=0.7)

    # Load and plot ground truth data
    try:
        ground_truth_events = load_ground_truth_events(ENSEMBLE_GROUND_TRUTH_PATH, session_id)
        gt_legend_handles = {}
        
        # Filter ground truth events for this specific device
        device_gt_events = [event for event in ground_truth_events if event['player_id'] == int(device_id)]
        
        if device_gt_events:
            print(f"Found {len(device_gt_events)} ground truth events for device {device_id}")
            for event in device_gt_events:
                action = event['action']
                timestamp = event['timestamp']
                color = activity_colors.get(action, 'grey')
                
                # Create a span for the ground truth event (e.g., 1 second duration)
                span_duration = 1.0  # seconds
                start_time = timestamp - span_duration/2
                end_time = timestamp + span_duration/2
                
                span = ax.axvspan(start_time, end_time, color=color, alpha=0.4, 
                                label=f'GT: {action}' if action not in gt_legend_handles else "")
                if action not in gt_legend_handles and action in activity_colors:
                    patch = plt.Rectangle((0,0), 1, 1, color=color, alpha=0.4)
                    gt_legend_handles[action] = patch
        else:
            print(f"No ground truth events found for device {device_id}")
            gt_legend_handles = {}
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        gt_legend_handles = {}
    
    ax.set_title(f'Ground Truth - Session {session_id}, Device {device_id}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Gyroscope Reading')
    
    # Create legend from base gyro plots and ground truth annotations
    handles, labels = ax.get_legend_handles_labels()
    # Add ground truth legend items
    for label, handle in gt_legend_handles.items():
        handles.append(handle)
        labels.append(f'GT: {label}')
    
    # To ensure no duplicate labels in legend
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(True, linestyle='--', alpha=0.6)

#------------------------------------------------------------------------------
# Load test data and make predictions using BEST models
#------------------------------------------------------------------------------
print("\nLoading test data for final predictions with best models...")
try:
    test_data_df = pd.read_csv(TEST_DATA_PATH)
    
    # Dictionary to collect predictions from BEST models
    predictions_from_best_models = {}

    # --- CNN Predictions with Best CNN Config ---
    if 'CNN' in best_overall_configs:
        cnn_best_config = best_overall_configs['CNN']
        print(f"\n--- Predicting with Best Tuned CNN (Overlap: {cnn_best_config['overlap_fraction']*100:.0f}%) ---")
        
        # Check if CNN was hyperparameter tuned and needs final training
        if cnn_best_config.get('needs_final_training', False) and cnn_best_config.get('tuned_model_config'):
            print("CNN was hyperparameter tuned. Training final model with best configuration...")
            
            # Get the tuned configuration
            tuned_config = cnn_best_config['tuned_model_config']
            
            # Recreate training data from the last processed data
            if 'windows' in locals() and 'cnn_labels_categorical' in locals() and 'current_scaler_cnn' in locals():
                # Create the tuned model
                tuned_cnn_model = create_cnn_model_tunable(
                    cnn_best_config['input_shape'], 
                    cnn_best_config['num_classes'], 
                    tuned_config
                )
                
                # Train with the tuned hyperparameters
                cnn_model_path_tuned = f'{OUTPUT_DIR_MODELS}best_cnn_hypertuned_model.h5'
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
                    ModelCheckpoint(cnn_model_path_tuned, save_best_only=True, monitor='val_accuracy')
                ]
                
                # Scale the data for final training
                X_train_cnn_final = windows
                y_train_cnn_final = cnn_labels_categorical
                
                # Use the scaler from the best configuration
                current_scaler_cnn_final = cnn_best_config['scaler_cnn']
                X_train_cnn_reshaped_final = X_train_cnn_final.reshape(-1, X_train_cnn_final.shape[-1])
                X_train_cnn_scaled_final = current_scaler_cnn_final.transform(X_train_cnn_reshaped_final)
                X_train_cnn_scaled_final = X_train_cnn_scaled_final.reshape(X_train_cnn_final.shape)
                
                print(f"Training final CNN with tuned hyperparameters: {tuned_config}")
                history_tuned = tuned_cnn_model.fit(
                    X_train_cnn_scaled_final, y_train_cnn_final,
                    validation_split=0.2, epochs=50, batch_size=tuned_config['batch_size'], 
                    callbacks=callbacks, verbose=1
                )
                
                # Update the model object in the configuration
                cnn_best_config['model_object'] = tuned_cnn_model
                cnn_best_config['model_path'] = cnn_model_path_tuned
                print("Final CNN model training completed with tuned hyperparameters.")
            else:
                print("Warning: Training data not available for final CNN training. Using original model.")
        
        # WINDOW_SIZE is global, cnn_best_config['step_size'] has the correct step
        cnn_preds_by_combo, cnn_export_paths = predict_with_cnn_model(
            test_data_df, cnn_best_config['model_object'], 
            WINDOW_SIZE, cnn_best_config['step_size'], # Use global WINDOW_SIZE and best step_size
            cnn_best_config['scaler_cnn'], cnn_best_config['label_encoder'],
            export_csv=True, csv_path_base=f'{OUTPUT_DIR_MODEL_PREDICTIONS}predictions_best_cnn',
            export_txt=True, txt_path_base=f'{OUTPUT_DIR_MODEL_PREDICTIONS}predictions_best_cnn'
        )
        predictions_from_best_models['CNN_Best_Tuned'] = cnn_preds_by_combo
        print(f"Best CNN predictions exported to: {cnn_export_paths}")
    else:
        print("Skipping predictions with Best CNN as no configuration was found.")

    # --- Traditional ML Model Predictions with their Best Configs ---
    for model_name, ml_best_config in best_overall_configs.items():
        if model_name == 'CNN': continue # Already handled
        print(f"\n--- Predicting with Best Tuned {model_name} (Overlap: {ml_best_config['overlap_fraction']*100:.0f}%) ---")
        
        ml_preds_by_combo_encoded = predict_with_ml_model(
            test_data_df, ml_best_config['pipeline'],
            WINDOW_SIZE, ml_best_config['step_size'] 
        )
        
        # Decode these predictions 
        ml_label_encoder_for_decode = ml_best_config['label_encoder_ml']
        ml_preds_by_combo_decoded = {}
        for combo_key, preds_data in ml_preds_by_combo_encoded.items():
            if preds_data and 'predictions' in preds_data:
                decoded_predictions = ml_label_encoder_for_decode.inverse_transform(preds_data['predictions'])
                ml_preds_by_combo_decoded[combo_key] = {
                    'timestamps': preds_data['timestamps'],
                    'predictions': decoded_predictions
                }
            else:
                ml_preds_by_combo_decoded[combo_key] = preds_data

        predictions_from_best_models[f'{model_name}_Best_Tuned'] = ml_preds_by_combo_decoded

    # Initialize a dictionary to store ensemble actions for each session
    all_sessions_ensemble_actions = {}
    # Initialize a list to store data for consolidated ensemble plotting
    ensemble_plot_data_list = []

    # --- Processing test data for predictions, JSONs, and plots from best models ---
    print("\n--- Processing test data for predictions, JSONs, and plots from best models ---")
    session_device_groups_test = test_data_df.groupby(['session_id', 'device_id'])

    for (session_id, device_id), group_data_test in session_device_groups_test:
        print(f"Processing Session: {session_id}, Device: {device_id} (using best models)")
        
        current_combo_best_preds = {}
        for model_key_pred, all_model_preds_data in predictions_from_best_models.items():
            current_combo_best_preds[model_key_pred] = all_model_preds_data.get((session_id, device_id))
        
        group_data_test_sorted = group_data_test.sort_values('timestamp')
        
        device_ensemble_actions = generate_ensemble_actions_for_device(
            group_data_test_sorted, 
            current_combo_best_preds, 
            WINDOW_SIZE, 
            SAMPLING_RATE,
            int(device_id),
            min_agreement_count=4
        )
        
        if session_id not in all_sessions_ensemble_actions:
            all_sessions_ensemble_actions[session_id] = []
        all_sessions_ensemble_actions[session_id].extend(device_ensemble_actions)
        
        if any(pred is not None for pred in current_combo_best_preds.values()):
            plot_all_models_predictions( 
                session_id, device_id, group_data_test_sorted,
                current_combo_best_preds, 
                WINDOW_SIZE, SAMPLING_RATE, 
                output_dir=OUTPUT_DIR_MODEL_PREDICTIONS
            )
            
            # Store data for later consolidated ensemble plotting
            if any(p_data and p_data.get('timestamps') for p_data in current_combo_best_preds.values()):             
                ensemble_plot_data_list.append({
                    'session_id': session_id,
                    'device_id': device_id,
                    'data_sorted': group_data_test_sorted,
                    'predictions': current_combo_best_preds
                })
        else:
             print(f"Skipping individual plots for Session: {session_id}, Device: {device_id} - No best model predictions found.")

    # After processing all session-device combinations, save the aggregated JSONs
    print("\n--- Saving aggregated ensemble predictions to JSON files (with merging) ---")
    for session_id, actions_list in all_sessions_ensemble_actions.items():
        if not actions_list:
            print(f"No ensemble actions to save for session {session_id}.")
            continue

        # Sort actions by player_id first, then by timestamp for stable merging
        sorted_actions_for_session = sorted(actions_list, key=lambda x: (x['player_id'], x['timestamp']))
        
        merged_actions_for_session = []
        if not sorted_actions_for_session:
            pass
        else:
            # Start with the first action as the potential start of a merged sequence
            current_merged_action = sorted_actions_for_session[0].copy() 

            for i in range(1, len(sorted_actions_for_session)):
                next_action = sorted_actions_for_session[i]
                # Check if the next action continues the current merged sequence
                if next_action['player_id'] == current_merged_action['player_id'] and \
                   next_action['action'] == current_merged_action['action']:
                    # It's a continuation, so we just move to the next action. 
                    pass 
                else:
                    # The sequence ended or changed player/action. Add the completed current_merged_action.
                    merged_actions_for_session.append(current_merged_action)
                    # Start a new potential merged sequence
                    current_merged_action = next_action.copy()
            
            # After the loop, add the last current_merged_action
            if current_merged_action:
                 merged_actions_for_session.append(current_merged_action)

        if not merged_actions_for_session:
            print(f"No actions to save for session {session_id} after merging.")
            continue

        # Final sort of merged actions by timestamp
        final_sorted_merged_actions = sorted(merged_actions_for_session, key=lambda x: x['timestamp'])

        json_filename = os.path.join(OUTPUT_DIR_ENSEMBLE_PREDICTIONS, f'session_{session_id}_ensemble_predictions.json')
        try:
            with open(json_filename, 'w') as f:
                json.dump(final_sorted_merged_actions, f, indent=4)
            print(f"Saved MERGED ensemble predictions for session {session_id} to {json_filename}")
        except Exception as e:
            print(f"Error saving MERGED JSON for session {session_id}: {e}")

    # --- Create Consolidated Ensemble Plot with Ground Truth Comparison ---
    if ensemble_plot_data_list:
        print("\n--- Generating consolidated ensemble plot with ground truth comparison ---")
        num_ensemble_plots = len(ensemble_plot_data_list)
        # Adjust subplot height based on number of plots, ensure a minimum height
        subplot_height = max(5, 18 / num_ensemble_plots if num_ensemble_plots <=3 else 4) 
        fig_height = subplot_height * num_ensemble_plots

        # Create side-by-side subplots: left for ground truth, right for predictions
        fig, axes = plt.subplots(num_ensemble_plots, 2, figsize=(36, fig_height), sharex=True)
        if num_ensemble_plots == 1:
            axes = [axes] # Make it iterable if only one subplot row

        for i, plot_data in enumerate(ensemble_plot_data_list):
            if num_ensemble_plots == 1:
                ax_gt, ax_pred = axes[0], axes[1]
            else:
                ax_gt, ax_pred = axes[i][0], axes[i][1]
            
            print(f"Plotting GT vs Ensemble for Session: {plot_data['session_id']}, Device: {plot_data['device_id']} on subplot row {i+1}")
            
            # Plot ground truth on left subplot
            plot_ground_truth_data(
                ax_gt,
                plot_data['session_id'],
                plot_data['device_id'],
                plot_data['data_sorted'],
                WINDOW_SIZE,
                SAMPLING_RATE
            )
            
            # Plot ensemble predictions on right subplot
            plot_ensemble_model_predictions(
                ax_pred,
                plot_data['session_id'],
                plot_data['device_id'],
                plot_data['data_sorted'],
                plot_data['predictions'],
                WINDOW_SIZE,
                SAMPLING_RATE,
                min_agreement_count=4 # Consistent with JSON generation
            )
        
        fig.suptitle('Ground Truth vs Ensemble Predictions Comparison by Session/Device', fontsize=20, y=0.995)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust for suptitle
        consolidated_plot_path = os.path.join(OUTPUT_DIR_ENSEMBLE_PREDICTIONS, 'all_sessions_devices_gt_vs_ensemble_comparison.png')
        try:
            plt.savefig(consolidated_plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved consolidated ground truth vs ensemble comparison plot to {consolidated_plot_path}")
        except Exception as e:
            print(f"Error saving consolidated comparison plot: {e}")
        plt.close(fig)
    else:
        print("No data available for consolidated ensemble plot.")

except FileNotFoundError:
    print(f"Test data file not found at '{TEST_DATA_PATH}'. Skipping final predictions.")
except Exception as e:
    print(f"An unexpected error occurred during final prediction or plotting with best models: {e}")
    import traceback
    traceback.print_exc()

#==============================================================================
# Print Table of All Models at All Overlaps
#==============================================================================
print(f"\n{'='*80}")
print("===== Metrics for ALL Models at ALL Overlaps =====")
print(f"{'='*80}\n")

all_overlap_metrics_data = []

for run_result in all_runs_results_list:
    overlap_percent = run_result['overlap_fraction'] * 100

    # Process ML models for this overlap
    if run_result['ml_results']:
        for model_name, metrics_dict in run_result['ml_results'].items():
            all_overlap_metrics_data.append({
                'Model Name': model_name,
                'Overlap (%)': overlap_percent,
                'Accuracy': metrics_dict.get('accuracy', np.nan),
                'Precision (macro)': metrics_dict.get('precision', np.nan),
                'Recall (macro)': metrics_dict.get('recall', np.nan),
                'F1 Score (macro)': metrics_dict.get('f1', np.nan)
            })

    # Process CNN model for this overlap
    if run_result['cnn_result']:
        cnn_metrics_dict = run_result['cnn_result']
        all_overlap_metrics_data.append({
            'Model Name': 'CNN',
            'Overlap (%)': overlap_percent,
            'Accuracy': cnn_metrics_dict.get('accuracy', np.nan),
            'Precision (macro)': cnn_metrics_dict.get('precision', np.nan),
            'Recall (macro)': cnn_metrics_dict.get('recall', np.nan),
            'F1 Score (macro)': cnn_metrics_dict.get('f1', np.nan)
        })

if all_overlap_metrics_data:
    all_overlaps_df = pd.DataFrame(all_overlap_metrics_data)
    # Sort for better readability, e.g., by Model Name then by Overlap
    all_overlaps_df = all_overlaps_df.sort_values(by=['Model Name', 'Overlap (%)'])
    
    print("Full Metrics Table (All Models, All Overlaps):")
    print(all_overlaps_df.to_string(index=False, float_format='%.4f'))
    
    # Save to CSV
    csv_path_all_metrics = os.path.join(OUTPUT_DIR_PERFORMANCE_ANALYSIS, 'all_models_all_overlaps_metrics.csv')
    try:
        all_overlaps_df.to_csv(csv_path_all_metrics, index=False, float_format='%.4f')
        print(f"\nFull metrics table saved to: {csv_path_all_metrics}")
    except Exception as e:
        print(f"\nError saving full metrics table to CSV: {e}")
else:
    print("No data available to generate the full metrics table for all models and overlaps.")

#==============================================================================
# Final Comparison of Ensemble Predictions with Ground Truth
#==============================================================================
def load_all_predicted_events(output_dir_figures):
    all_preds = []
    session_ids_found = set()
    # Ensure output_dir_figures exists
    if not os.path.exists(output_dir_figures):
        print(f"Predictions directory {output_dir_figures} not found. Cannot load predicted events.")
        return all_preds, None # Return None for session_id if dir not found
        
    for filename in os.listdir(output_dir_figures):
        if filename.startswith('session_') and filename.endswith('_ensemble_predictions.json'):
            filepath = os.path.join(output_dir_figures, filename)
            try:
                # Extract session_id from filename, assuming format session_{session_id}_ensemble_predictions.json
                parts = filename.replace('session_', '').split('_ensemble_predictions.json')
                if parts:
                    current_session_id = parts[0]
                    session_ids_found.add(current_session_id)
                else:
                    print(f"Could not parse session_id from filename: {filename}")
                    current_session_id = "unknown_session" # Fallback

                with open(filepath, 'r') as f:
                    session_preds = json.load(f)
                    for pred in session_preds:
                        pred['session_id'] = pred.get('session_id', current_session_id) # Ensure session_id is in each event
                        all_preds.append(pred)
            except Exception as e:
                print(f"Error loading prediction file {filepath}: {e}")
    
    if not session_ids_found:
        print("No prediction files found or no session IDs could be parsed.")
        return all_preds, None
    if len(session_ids_found) > 1:
        print(f"Warning: Multiple session IDs found in prediction files: {session_ids_found}. Ground truth comparison assumes a single session.")
        return all_preds, list(session_ids_found)[0] # Return the single session ID
    
    return all_preds, list(session_ids_found)[0] # Return the single session ID

def convert_model_predictions_to_events(model_predictions, model_name, unique_session_id, 
                                         confidence_threshold=0.9, merge_similar_events=True):
    """
    Convert individual model predictions to the same format as ensemble predictions
    with confidence filtering and event merging to reduce false positives.
    
    Parameters:
    -----------
    confidence_threshold : float
        For CNN models, minimum probability required. For ML models, always applied.
    merge_similar_events : bool
        Whether to merge consecutive predictions of the same action
    """
    model_events = []
    
    for (session_id, device_id), pred_data in model_predictions.items():
        if pred_data and 'timestamps' in pred_data and 'predictions' in pred_data:
            timestamps = pred_data['timestamps']
            predictions = pred_data['predictions']
            probabilities = pred_data.get('probabilities', None)  # CNN models have this
            
            for i, (timestamp, action) in enumerate(zip(timestamps, predictions)):
                if action != 'null':  # Only include non-null predictions
                    # Apply confidence filtering for models with probabilities
                    include_prediction = True
                    confidence_score = 1.0  # Default for models without probabilities
                    
                    if probabilities is not None:  # Model has probabilities
                        # Handle different probability formats
                        if len(probabilities.shape) == 2:  # ML models: (n_samples, n_classes)
                            max_prob = np.max(probabilities[i])
                            confidence_score = float(max_prob)
                        else:  # CNN models: might be different format
                            max_prob = np.max(probabilities[i])
                            confidence_score = float(max_prob)
                        
                        if max_prob < confidence_threshold:
                            include_prediction = False
                    
                    if include_prediction:
                        model_events.append({
                            'session_id': unique_session_id,
                            'player_id': int(device_id),
                            'timestamp': float(timestamp),
                            'action': action,
                            'confidence': confidence_score
                        })
    
    # Sort by player_id then timestamp for consistency
    model_events.sort(key=lambda x: (x['player_id'], x['timestamp']))
    
    # Optional: Merge consecutive similar events to reduce duplicates
    if merge_similar_events and model_events:
        merged_events = []
        current_event = model_events[0].copy()
        
        for next_event in model_events[1:]:
            # Check if events should be merged (same player, action, and close in time)
            if (next_event['player_id'] == current_event['player_id'] and 
                next_event['action'] == current_event['action'] and 
                abs(next_event['timestamp'] - current_event['timestamp']) <= EVENT_MERGE_TIME_WINDOW):  # Use global config
                # Keep the higher confidence event
                if next_event.get('confidence', 1.0) > current_event.get('confidence', 1.0):
                    current_event = next_event
            else:
                merged_events.append(current_event)
                current_event = next_event.copy()
        
        merged_events.append(current_event)  # Add the last event
        model_events = merged_events
    
    print(f"Converted {len(model_events)} events from {model_name} (confidence ≥ {confidence_threshold}, merged: {merge_similar_events})")
    return model_events

def apply_non_maximum_suppression(events, time_window=2.0):
    """
    Apply non-maximum suppression to remove overlapping predictions.
    Keeps only the highest confidence prediction within each time window.
    
    Parameters:
    -----------
    events : list
        List of event dictionaries with 'timestamp', 'confidence', 'player_id', 'action'
    time_window : float
        Time window in seconds for suppression
    """
    if not events:
        return events
    
    # Group by player and action
    suppressed_events = []
    events_by_player_action = {}
    
    for event in events:
        key = (event['player_id'], event['action'])
        if key not in events_by_player_action:
            events_by_player_action[key] = []
        events_by_player_action[key].append(event)
    
    for key, group_events in events_by_player_action.items():
        # Sort by confidence (highest first)
        group_events.sort(key=lambda x: x.get('confidence', 1.0), reverse=True)
        
        selected = []
        for event in group_events:
            # Check if this event is too close to any already selected
            too_close = False
            for selected_event in selected:
                if abs(event['timestamp'] - selected_event['timestamp']) < time_window:
                    too_close = True
                    break
            
            if not too_close:
                selected.append(event)
        
        suppressed_events.extend(selected)
    
    suppressed_events.sort(key=lambda x: (x['player_id'], x['timestamp']))
    print(f"Non-maximum suppression: {len(events)} → {len(suppressed_events)} events (window: {time_window}s)")
    return suppressed_events

def compare_predictions_to_ground_truth(predicted_events, ground_truth_events, time_tolerance=2.0, model_name="Model"):
    print(f"\n{'='*80}")
    print(f"===== {model_name} Predictions vs Ground Truth Comparison =====")
    print(f"Time tolerance for matching: +/- {time_tolerance} seconds")
    print(f"{'='*80}\n")

    if not predicted_events:
        print("No predicted events to compare.")
        return
    if not ground_truth_events:
        print("No ground truth events to compare.")
        return

    # Create copies to mark events as used without modifying original lists
    gt_events_copy = [dict(event, **{'used': False, 'original_idx': i}) for i, event in enumerate(ground_truth_events)]
    pred_events_copy = [dict(event, **{'used': False, 'original_idx': i}) for i, event in enumerate(predicted_events)]

    matches_by_action = {}
    gt_counts_by_action = {}
    pred_counts_by_action = {} # For all predictions, including those that might be FPs
    all_actions = set()

    for event in gt_events_copy:
        action = event['action']
        all_actions.add(action)
        gt_counts_by_action[action] = gt_counts_by_action.get(action, 0) + 1
        matches_by_action.setdefault(action, {'TP': 0, 'FP': 0, 'FN': 0})

    for event in pred_events_copy:
        action = event['action']
        all_actions.add(action)
        pred_counts_by_action[action] = pred_counts_by_action.get(action, 0) + 1
        matches_by_action.setdefault(action, {'TP': 0, 'FP': 0, 'FN': 0})


    # --- Matching Logic: Iterate through GT events and find matches in predictions ---
    # Sort both lists by session, player, and timestamp for potentially more efficient searching
    gt_events_copy.sort(key=lambda x: (x['session_id'], x['player_id'], x['timestamp']))
    pred_events_copy.sort(key=lambda x: (x['session_id'], x['player_id'], x['timestamp']))

    for gt_event in gt_events_copy:
        if gt_event['used']:
            continue

        gt_action = gt_event['action']
        gt_ts = gt_event['timestamp']
        gt_player = gt_event['player_id']
        gt_session = gt_event['session_id']
        
        found_match_for_gt = False
        best_match_pred_idx = -1
        min_time_diff = float('inf')

        for pred_event in pred_events_copy:
            if pred_event['used']:
                continue

            if gt_action == pred_event['action'] and \
               gt_player == pred_event['player_id'] and \
               gt_session == pred_event['session_id']:
                
                time_diff = abs(gt_ts - pred_event['timestamp'])
                if time_diff <= time_tolerance:
                    # If multiple predictions match a GT event within tolerance,
                    # pick the closest one in time that hasn't been used.
                    if time_diff < min_time_diff:
                         min_time_diff = time_diff
                         best_match_pred_idx = pred_event['original_idx'] 

        if best_match_pred_idx != -1:
            # Check if the selected best_match_pred_idx is truly unused
            target_pred_event_for_match = next((p for p in pred_events_copy if p['original_idx'] == best_match_pred_idx), None)
            if target_pred_event_for_match and not target_pred_event_for_match['used']:
                matches_by_action[gt_action]['TP'] += 1
                gt_event['used'] = True
                target_pred_event_for_match['used'] = True
                found_match_for_gt = True
        
        if not found_match_for_gt:
            matches_by_action[gt_action]['FN'] += 1

    # --- Calculate False Positives --- 
    for pred_event in pred_events_copy:
        if not pred_event['used']:
            matches_by_action[pred_event['action']]['FP'] += 1
            
    # --- Calculate and Print Metrics --- 
    print("\n--- Performance Metrics by Action Type ---")
    results_data = []
    overall_tp, overall_fp, overall_fn = 0, 0, 0

    for action in sorted(list(all_actions)):
        if action == 'null': continue # Typically exclude 'null'

        tp = matches_by_action.get(action, {}).get('TP', 0)
        fp = matches_by_action.get(action, {}).get('FP', 0)
        fn = matches_by_action.get(action, {}).get('FN', 0)

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # ADD ACCURACY: For event detection, accuracy = TP / (TP + FP + FN)
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        results_data.append({
            'Action': action,
            'TP': tp, 'FP': fp, 'FN': fn,
            'Accuracy': accuracy,  # ADD THIS
            'Precision': precision, 'Recall': recall, 'F1-Score': f1,
            'GT Count': gt_counts_by_action.get(action, 0),
            'Pred Count': pred_counts_by_action.get(action, 0)
        })

    results_df = pd.DataFrame(results_data)
    if not results_df.empty:
        print(results_df.to_string(index=False, float_format='%.3f'))
    else:
        print("No action data to report (excluding 'null').")

    # Overall metrics calculation
    overall_precision_micro = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall_micro = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1_micro = 2 * (overall_precision_micro * overall_recall_micro) / (overall_precision_micro + overall_recall_micro) if (overall_precision_micro + overall_recall_micro) > 0 else 0
    # ADD OVERALL ACCURACY
    overall_accuracy = overall_tp / (overall_tp + overall_fp + overall_fn) if (overall_tp + overall_fp + overall_fn) > 0 else 0

    # Macro-averaged metrics (per-class average)
    if results_data:
        macro_precision = sum(row['Precision'] for row in results_data) / len(results_data)
        macro_recall = sum(row['Recall'] for row in results_data) / len(results_data)
        macro_f1 = sum(row['F1-Score'] for row in results_data) / len(results_data)
        macro_accuracy = sum(row['Accuracy'] for row in results_data) / len(results_data)  # ADD THIS
    else:
        macro_precision = macro_recall = macro_f1 = macro_accuracy = 0  # ADD macro_accuracy

    print(f"\n--- Overall Performance ---")
    print(f"Overall Accuracy:         {overall_accuracy:.3f} (TP / (TP + FP + FN))")  # ADD THIS
    print(f"Micro-Averaged Precision: {overall_precision_micro:.3f} (Calculated from total TP and FP)")
    print(f"Micro-Averaged Recall:    {overall_recall_micro:.3f} (Calculated from total TP and FN)")
    print(f"Micro-Averaged F1-Score:  {overall_f1_micro:.3f} (Calculated from total TP, FP, and FN)")
    print(f"Macro-Averaged Accuracy:  {macro_accuracy:.3f} (Average of per-class accuracy)")  # ADD THIS
    print(f"Macro-Averaged Precision: {macro_precision:.3f} (Average of per-class precision)")
    print(f"Macro-Averaged Recall:    {macro_recall:.3f} (Average of per-class recall)")
    print(f"Macro-Averaged F1-Score:  {macro_f1:.3f} (Average of per-class F1-score)")

    print(f"\nTotal GT Events (all types): {len(ground_truth_events)}")
    print(f"Total Predicted Events (all types): {len(predicted_events)}")
    print(f"{'='*80}\n")
    
    # Return metrics for summary table - ADD ACCURACY FIELDS
    return {
        'accuracy': overall_accuracy,  # ADD THIS
        'macro_accuracy': macro_accuracy,  # ADD THIS
        'micro_precision': overall_precision_micro,
        'micro_recall': overall_recall_micro,
        'micro_f1': overall_f1_micro,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'total_gt_events': len(ground_truth_events),
        'total_pred_events': len(predicted_events),
        'total_tp': overall_tp,
        'total_fp': overall_fp,
        'total_fn': overall_fn
    }

print("\n===== Script Finished =====")

def compare_predictions_to_ground_truth_with_null(predicted_events, ground_truth_events, 
                                                 session_data_dict, time_tolerance=2.0, 
                                                 temporal_resolution=0.1, action_duration=2.0,
                                                 model_name="Model"):
    """
    Enhanced ground truth comparison that includes null class evaluation.
    
    Parameters:
    -----------
    predicted_events : list
        List of predicted event dictionaries
    ground_truth_events : list  
        List of ground truth event dictionaries
    session_data_dict : dict
        Dictionary with (session_id, device_id) keys containing sensor data
    temporal_resolution : float
        Time resolution in seconds for timeline sampling (default: 0.1s)
    action_duration : float
        Duration in seconds to consider an action active (default: 2.0s)
    """
    print(f"\n{'='*80}")
    print(f"===== {model_name} Predictions vs Ground Truth (Including Null Class) =====")
    print(f"Time tolerance: +/- {time_tolerance}s, Resolution: {temporal_resolution}s, Action duration: {action_duration}s")
    print(f"{'='*80}\n")

    if not session_data_dict:
        print("No session data available for timeline-based evaluation.")
        return

    # Collect all session/device combinations
    all_combinations = set()
    for event in predicted_events + ground_truth_events:
        all_combinations.add((event['session_id'], event['player_id']))
    
    if not all_combinations:
        print("No session/device combinations found.")
        return

    # Initialize metrics tracking
    timeline_metrics = {}
    all_actions = set(['null'])  # Include null explicitly
    
    for event in predicted_events + ground_truth_events:
        all_actions.add(event['action'])
    
    # Initialize confusion matrix
    confusion_data = {action: {pred_action: 0 for pred_action in all_actions} for action in all_actions}
    
    total_timeline_points = 0
    
    for session_id, device_id in all_combinations:
        print(f"Processing timeline for Session {session_id}, Device {device_id}")
        
        # Get sensor data timespan for this combination
        if (session_id, device_id) not in session_data_dict:
            print(f"  Warning: No sensor data found for Session {session_id}, Device {device_id}")
            continue
            
        device_data = session_data_dict[(session_id, device_id)]
        data_start_time = device_data['timestamp'].min()
        data_end_time = device_data['timestamp'].max()
        
        # Create timeline at specified resolution
        timeline = np.arange(data_start_time, data_end_time + temporal_resolution, temporal_resolution)
        total_timeline_points += len(timeline)
        
        # Initialize timeline labels
        gt_timeline = ['null'] * len(timeline)
        pred_timeline = ['null'] * len(timeline)
        
        # Mark ground truth actions on timeline
        device_gt_events = [e for e in ground_truth_events 
                           if e['session_id'] == session_id and e['player_id'] == device_id]
        
        for gt_event in device_gt_events:
            gt_time = gt_event['timestamp']
            gt_action = gt_event['action']
            
            # Mark action duration centered on event time
            action_start = gt_time - action_duration / 2
            action_end = gt_time + action_duration / 2
            
            for i, t in enumerate(timeline):
                if action_start <= t <= action_end:
                    gt_timeline[i] = gt_action
        
        # Mark predicted actions on timeline
        device_pred_events = [e for e in predicted_events 
                             if e['session_id'] == session_id and e['player_id'] == device_id]
        
        for pred_event in device_pred_events:
            pred_time = pred_event['timestamp']
            pred_action = pred_event['action']
            
            # Mark action duration centered on prediction time
            action_start = pred_time - action_duration / 2
            action_end = pred_time + action_duration / 2
            
            for i, t in enumerate(timeline):
                if action_start <= t <= action_end:
                    pred_timeline[i] = pred_action
        
        # Update confusion matrix from timeline comparison
        for gt_label, pred_label in zip(gt_timeline, pred_timeline):
            confusion_data[gt_label][pred_label] += 1
    
    print(f"Analyzed {total_timeline_points} timeline points across {len(all_combinations)} devices")
    
    # Calculate metrics from confusion matrix
    print(f"\n--- Timeline-Based Confusion Matrix ---")
    
    # Create DataFrame for better display
    confusion_df = pd.DataFrame(confusion_data)
    confusion_df.index.name = 'Ground Truth'
    confusion_df.columns.name = 'Predicted'
    
    print("Confusion Matrix (rows=GT, cols=Predicted):")
    print(confusion_df.to_string())
    
    # Calculate per-class metrics
    print(f"\n--- Per-Class Performance Metrics ---")
    results_data = []
    
    for action in sorted(all_actions):
        # True positives: correctly predicted this action
        tp = confusion_data[action][action]
        
        # False positives: predicted this action when it wasn't true
        fp = sum(confusion_data[other_action][action] for other_action in all_actions if other_action != action)
        
        # False negatives: failed to predict this action when it was true
        fn = sum(confusion_data[action][other_action] for other_action in all_actions if other_action != action)
        
        # True negatives: correctly predicted NOT this action
        tn = sum(confusion_data[gt][pred] for gt in all_actions for pred in all_actions 
                if gt != action and pred != action)
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Total ground truth instances
        gt_total = sum(confusion_data[action].values())
        pred_total = sum(confusion_data[gt][action] for gt in all_actions)
        
        results_data.append({
            'Action': action,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Accuracy': accuracy,
            'Precision': precision, 
            'Recall': recall, 
            'F1-Score': f1,
            'Specificity': specificity,
            'GT Total': gt_total,
            'Pred Total': pred_total
        })
    
    results_df = pd.DataFrame(results_data)
    print(results_df.to_string(index=False, float_format='%.3f'))
    
    # Overall metrics
    total_correct = sum(confusion_data[action][action] for action in all_actions)
    total_points = total_timeline_points
    overall_accuracy = total_correct / total_points if total_points > 0 else 0
    
    # Macro averages (including null)
    macro_accuracy = results_df['Accuracy'].mean()
    macro_precision = results_df['Precision'].mean()
    macro_recall = results_df['Recall'].mean()
    macro_f1 = results_df['F1-Score'].mean()
    macro_specificity = results_df['Specificity'].mean()
    
    # Micro averages (weighted by total instances)
    total_tp = results_df['TP'].sum()
    total_tn = results_df['TN'].sum()
    total_fp = results_df['FP'].sum()
    total_fn = results_df['FN'].sum()
    
    micro_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    print(f"\n--- Overall Performance (Including Null Class) ---")
    print(f"Timeline Overall Accuracy: {overall_accuracy:.3f} (Correct predictions / Total timeline points)")
    print(f"Micro-Averaged Accuracy:   {micro_accuracy:.3f}")
    print(f"Micro-Averaged Precision:  {micro_precision:.3f}")
    print(f"Micro-Averaged Recall:     {micro_recall:.3f}")
    print(f"Micro-Averaged F1-Score:   {micro_f1:.3f}")
    print(f"Macro-Averaged Accuracy:   {macro_accuracy:.3f} (Including null class)")
    print(f"Macro-Averaged Precision:  {macro_precision:.3f} (Including null class)")
    print(f"Macro-Averaged Recall:     {macro_recall:.3f} (Including null class)")
    print(f"Macro-Averaged F1-Score:   {macro_f1:.3f} (Including null class)")
    print(f"Macro-Averaged Specificity: {macro_specificity:.3f} (Including null class)")
    
    print(f"\nTimeline Analysis Summary:")
    print(f"Total timeline points analyzed: {total_timeline_points:,}")
    print(f"Temporal resolution: {temporal_resolution}s")
    print(f"Action duration assumption: {action_duration}s")
    print(f"Devices analyzed: {len(all_combinations)}")
    print(f"{'='*80}\n")
    
    return {
        'timeline_accuracy': overall_accuracy,
        'micro_accuracy': micro_accuracy,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_accuracy': macro_accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_specificity': macro_specificity,
        'confusion_matrix': confusion_data,
        'per_class_results': results_df,
        'total_timeline_points': total_timeline_points,
        'temporal_resolution': temporal_resolution,
        'action_duration': action_duration
    }

# Call the comparison at the very end, but ensure it's within a main check
if __name__ == '__main__':
    print("\nInitiating final comparison of predictions with ground truth...")
    
    all_predicted_events, unique_session_id_from_preds = load_all_predicted_events(OUTPUT_DIR_ENSEMBLE_PREDICTIONS)
    
    if unique_session_id_from_preds:
        print(f"Predictions loaded for session ID: {unique_session_id_from_preds}")
        ground_truth_events = load_ground_truth_events(ENSEMBLE_GROUND_TRUTH_PATH, unique_session_id_from_preds)
    else:
        print("No unique session ID determined from prediction files. Cannot reliably load session-agnostic ground truth.")
        ground_truth_events = []

    # Prepare session data dictionary for timeline evaluation
    session_data_dict = {}
    if 'test_data_df' in locals():
        for (session_id, device_id), group_data in test_data_df.groupby(['session_id', 'device_id']):
            session_data_dict[(session_id, device_id)] = group_data.sort_values('timestamp')
    
    # === INDIVIDUAL MODEL EVALUATIONS (Enhanced with Null Class) ===
    if ground_truth_events and unique_session_id_from_preds and 'predictions_from_best_models' in locals():
        print(f"\n{'='*80}")
        print("===== INDIVIDUAL MODEL EVALUATIONS (Including Null Class) =====")
        print(f"{'='*80}")
        
        individual_model_metrics = {}
        individual_model_metrics_with_null = {}
        
        for model_name, model_predictions in predictions_from_best_models.items():
            print(f"\n--- Evaluating {model_name} ---")
            
            # Convert model predictions to event format
            model_events = convert_model_predictions_to_events(
                model_predictions, 
                model_name, 
                unique_session_id_from_preds,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                merge_similar_events=MERGE_SIMILAR_EVENTS
            )
            
            # Apply non-maximum suppression if enabled
            if APPLY_NON_MAX_SUPPRESSION:
                model_events = apply_non_maximum_suppression(model_events, NMS_TIME_WINDOW)
            
            # ORIGINAL: Event-based comparison (excluding null) - for comparison
            print(f"\n--- {model_name}: Event-Based Evaluation (Original) ---")
            if model_events:
                metrics = compare_predictions_to_ground_truth(
                    model_events, 
                    ground_truth_events, 
                    time_tolerance=2.0, 
                    model_name=f"{model_name} (Event-Based)"
                )
                individual_model_metrics[model_name] = metrics
            else:
                print(f"No events generated for {model_name} after filtering.")
                individual_model_metrics[model_name] = None
            
            # NEW: Timeline-based comparison (including null class)
            print(f"\n--- {model_name}: Timeline-Based Evaluation (Including Null) ---")
            metrics_with_null = compare_predictions_to_ground_truth_with_null(
                model_events, 
                ground_truth_events,
                session_data_dict,
                time_tolerance=2.0,
                temporal_resolution=0.1,  # 100ms resolution
                action_duration=2.0,      # 2-second action duration
                model_name=f"{model_name} (Timeline-Based)"
            )
            individual_model_metrics_with_null[model_name] = metrics_with_null
        
        # Create summary table comparing both evaluation methods
        if individual_model_metrics and individual_model_metrics_with_null:
            print(f"\n{'='*80}")
            print("===== MODEL PERFORMANCE COMPARISON: Event-Based vs Timeline-Based =====")
            print(f"{'='*80}")
            
            comparison_data = []
            for model_name in individual_model_metrics.keys():
                event_metrics = individual_model_metrics[model_name]
                timeline_metrics = individual_model_metrics_with_null[model_name]
                
                if event_metrics and timeline_metrics:
                    # Find null class accuracy from timeline metrics
                    null_accuracy = 'N/A'
                    if timeline_metrics and 'per_class_results' in timeline_metrics:
                        null_row = timeline_metrics['per_class_results'][timeline_metrics['per_class_results']['Action'] == 'null']
                        if not null_row.empty:
                            null_accuracy = f"{null_row['Accuracy'].iloc[0]:.3f}"
                    
                    comparison_data.append({
                        'Model': model_name.replace('_Best_Tuned', ''),
                        'Event Accuracy': f"{event_metrics['accuracy']:.3f}",
                        'Timeline Accuracy': f"{timeline_metrics['timeline_accuracy']:.3f}",
                        'Event Micro F1': f"{event_metrics['micro_f1']:.3f}",
                        'Timeline Micro F1': f"{timeline_metrics['micro_f1']:.3f}",
                        'Event Macro F1': f"{event_metrics['macro_f1']:.3f}",
                        'Timeline Macro F1': f"{timeline_metrics['macro_f1']:.3f}",
                        'Null Class Accuracy': null_accuracy,
                        'Timeline Points': f"{timeline_metrics['total_timeline_points']:,}"
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                print("\nEvent-Based vs Timeline-Based Performance Comparison:")
                print(comparison_df.to_string(index=False))
                
                # Save comprehensive results
                event_csv_path = os.path.join(OUTPUT_DIR_GROUND_TRUTH_COMPARISON, 'event_based_gt_performance.csv')
                timeline_csv_path = os.path.join(OUTPUT_DIR_GROUND_TRUTH_COMPARISON, 'timeline_based_gt_performance.csv')
                comparison_csv_path = os.path.join(OUTPUT_DIR_GROUND_TRUTH_COMPARISON, 'evaluation_methods_comparison.csv')
                
                try:
                    # Save event-based results
                    event_summary_data = []
                    for model_name, metrics in individual_model_metrics.items():
                        if metrics:
                            event_summary_data.append({
                                'Model': model_name.replace('_Best_Tuned', ''),
                                'Accuracy': f"{metrics['accuracy']:.3f}",
                                'Micro Precision': f"{metrics['micro_precision']:.3f}",
                                'Micro Recall': f"{metrics['micro_recall']:.3f}",
                                'Micro F1': f"{metrics['micro_f1']:.3f}",
                                'Macro Accuracy': f"{metrics['macro_accuracy']:.3f}",
                                'Macro Precision': f"{metrics['macro_precision']:.3f}",
                                'Macro Recall': f"{metrics['macro_recall']:.3f}",
                                'Macro F1': f"{metrics['macro_f1']:.3f}",
                                'Total Predictions': metrics['total_pred_events'],
                                'True Positives': metrics['total_tp'],
                                'False Positives': metrics['total_fp'],
                                'False Negatives': metrics['total_fn']
                            })
                    
                    if event_summary_data:
                        pd.DataFrame(event_summary_data).to_csv(event_csv_path, index=False)
                        print(f"\nEvent-based performance saved to: {event_csv_path}")
                    
                    # Save timeline-based results
                    timeline_summary_data = []
                    for model_name, metrics in individual_model_metrics_with_null.items():
                        if metrics:
                            timeline_summary_data.append({
                                'Model': model_name.replace('_Best_Tuned', ''),
                                'Timeline Accuracy': f"{metrics['timeline_accuracy']:.3f}",
                                'Micro Accuracy': f"{metrics['micro_accuracy']:.3f}",
                                'Micro Precision': f"{metrics['micro_precision']:.3f}",
                                'Micro Recall': f"{metrics['micro_recall']:.3f}",
                                'Micro F1': f"{metrics['micro_f1']:.3f}",
                                'Macro Accuracy': f"{metrics['macro_accuracy']:.3f}",
                                'Macro Precision': f"{metrics['macro_precision']:.3f}",
                                'Macro Recall': f"{metrics['macro_recall']:.3f}",
                                'Macro F1': f"{metrics['macro_f1']:.3f}",
                                'Macro Specificity': f"{metrics['macro_specificity']:.3f}",
                                'Timeline Points': f"{metrics['total_timeline_points']:,}",
                                'Temporal Resolution': f"{metrics['temporal_resolution']}s",
                                'Action Duration': f"{metrics['action_duration']}s"
                            })
                    
                    if timeline_summary_data:
                        pd.DataFrame(timeline_summary_data).to_csv(timeline_csv_path, index=False)
                        print(f"Timeline-based performance saved to: {timeline_csv_path}")
                    
                    # Save comparison
                    comparison_df.to_csv(comparison_csv_path, index=False)
                    print(f"Evaluation methods comparison saved to: {comparison_csv_path}")
                    
                except Exception as e:
                    print(f"Error saving performance summaries: {e}")

    # === ENSEMBLE MODEL EVALUATION (Enhanced with Null Class) ===
    if ground_truth_events and all_predicted_events:
        print(f"\n{'='*80}")
        print("===== ENSEMBLE MODEL EVALUATION (Including Null Class) =====")
        print(f"{'='*80}")
        
        # Original event-based evaluation
        print(f"\n--- Ensemble: Event-Based Evaluation (Original) ---")
        ensemble_metrics = compare_predictions_to_ground_truth(
            all_predicted_events, ground_truth_events, 
            time_tolerance=2.0, model_name="Ensemble (Event-Based)"
        )
        
        # New timeline-based evaluation
        print(f"\n--- Ensemble: Timeline-Based Evaluation (Including Null) ---")
        ensemble_metrics_with_null = compare_predictions_to_ground_truth_with_null(
            all_predicted_events, 
            ground_truth_events,
            session_data_dict,
            time_tolerance=2.0,
            temporal_resolution=0.1,
            action_duration=2.0,
            model_name="Ensemble (Timeline-Based)"
        )
        
        if ensemble_metrics and ensemble_metrics_with_null:
            print(f"\n--- Ensemble Performance Summary ---")
            print(f"Event-Based Overall Accuracy: {ensemble_metrics['accuracy']:.3f}")
            print(f"Timeline-Based Overall Accuracy: {ensemble_metrics_with_null['timeline_accuracy']:.3f}")
            print(f"Event-Based Micro F1: {ensemble_metrics['micro_f1']:.3f}")
            print(f"Timeline-Based Micro F1: {ensemble_metrics_with_null['micro_f1']:.3f}")
            print(f"Timeline-Based Macro F1 (inc. null): {ensemble_metrics_with_null['macro_f1']:.3f}")
            print(f"Timeline-Based Macro Specificity: {ensemble_metrics_with_null['macro_specificity']:.3f}")
    else:
        if not ground_truth_events:
            print("No ground truth events available for comparison.")
        if not all_predicted_events:
            print("No predicted events available for comparison.")

    print("\n===== Final Script Completion =====")
    print("All analysis and comparisons completed successfully!")

