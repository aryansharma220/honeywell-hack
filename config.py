"""
Configuration module for the anomaly detection system.

This module contains constants and configuration parameters used throughout
the anomaly detection pipeline.
"""

from datetime import datetime

# Time period definitions
TRAINING_START = datetime(2004, 1, 1, 0, 0)
TRAINING_END = datetime(2004, 1, 5, 23, 59)
ANALYSIS_START = datetime(2004, 1, 1, 0, 0)  
ANALYSIS_END = datetime(2004, 1, 19, 7, 59)

# Model parameters
DEFAULT_CONTAMINATION = 0.01  # Lower contamination for better training scores
DEFAULT_RANDOM_STATE = 42
N_ESTIMATORS = 200  # More estimators for better stability

# Feature importance parameters
MIN_CONTRIBUTION_THRESHOLD = 1.0  # Minimum 1% contribution
MAX_TOP_FEATURES = 7

# Validation thresholds
MIN_TRAINING_HOURS = 72
MAX_TRAINING_MEAN_SCORE = 10
MAX_TRAINING_MAX_SCORE = 25

# Data processing parameters
DATETIME_FORMAT = '%m/%d/%Y %H:%M'
IMPUTATION_STRATEGY = 'median'
SCALING_METHOD = 'standard'

# Output column names
ANOMALY_SCORE_COLUMN = 'Abnormality_score'
TOP_FEATURE_COLUMNS = [f'top_feature_{i}' for i in range(1, MAX_TOP_FEATURES + 1)]

# Performance parameters
MAX_DATASET_ROWS = 10000
MAX_RUNTIME_MINUTES = 15
