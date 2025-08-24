"""
Configuration settings for anomaly detection project.
"""

from datetime import datetime

TRAINING_START = datetime(2004, 1, 1, 0, 0)
TRAINING_END = datetime(2004, 1, 5, 23, 59)
ANALYSIS_START = datetime(2004, 1, 1, 0, 0)  
ANALYSIS_END = datetime(2004, 1, 19, 7, 59)

DEFAULT_CONTAMINATION = 0.05 
DEFAULT_RANDOM_STATE = 42
N_ESTIMATORS = 200

MIN_CONTRIBUTION_THRESHOLD = 1.0
MAX_TOP_FEATURES = 7

MIN_TRAINING_HOURS = 72
MAX_TRAINING_MEAN_SCORE = 10
MAX_TRAINING_MAX_SCORE = 25

DATETIME_FORMAT = '%m/%d/%Y %H:%M'
IMPUTATION_STRATEGY = 'median'
SCALING_METHOD = 'standard'

ANOMALY_SCORE_COLUMN = 'Abnormality_score'
TOP_FEATURE_COLUMNS = [f'top_feature_{i}' for i in range(1, MAX_TOP_FEATURES + 1)]

# Progress indicator settings
ENABLE_PROGRESS_BARS = True
