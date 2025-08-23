"""
Utility functions for the anomaly detection project.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional
import config


def parse_datetime(time_str: str) -> datetime:
    """
    Convert time string to datetime object.
    
    Args:
        time_str: Time string to parse
        
    Returns:
        datetime object
        
    Raises:
        ValueError: If datetime format is invalid
    """
    try:
        return datetime.strptime(time_str.strip(), config.DATETIME_FORMAT)
    except ValueError as e:
        raise ValueError(f"Invalid datetime format: {time_str}. Expected format: {config.DATETIME_FORMAT}") from e


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the input dataset for basic requirements."""
    if df.empty:
        raise ValueError("Dataset is empty")
    
    if 'Time' not in df.columns:
        raise ValueError("Dataset must contain a 'Time' column")
    
    expected_numeric = len(df.columns) - 1
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < expected_numeric:
        non_numeric = [col for col in df.columns if col != 'Time' and col not in numeric_columns]
        print(f"Warning: Found non-numeric columns: {non_numeric}")


def validate_time_range(df: pd.DataFrame) -> Tuple[datetime, datetime]:
    """Check time range in the dataset."""
    if 'Time' not in df.columns:
        raise ValueError("Dataset must contain a 'Time' column")
    
    try:
        time_values = [parse_datetime(time_str) for time_str in df['Time']]
        start_time = min(time_values)
        end_time = max(time_values)
        
        if start_time > config.ANALYSIS_START:
            print(f"Warning: Dataset starts after expected analysis start ({start_time} > {config.ANALYSIS_START})")
        
        if end_time < config.ANALYSIS_END:
            print(f"Warning: Dataset ends before expected analysis end ({end_time} < {config.ANALYSIS_END})")
        
        return start_time, end_time
        
    except Exception as e:
        raise ValueError(f"Error parsing time column: {e}")


def split_time_series_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and analysis periods."""
    validate_dataset(df)
    
    training_data = []
    analysis_data = []
    
    for _, row in df.iterrows():
        time_value = parse_datetime(row['Time'])
        
        if config.TRAINING_START <= time_value <= config.TRAINING_END:
            training_data.append(row)
        
        if config.ANALYSIS_START <= time_value <= config.ANALYSIS_END:
            analysis_data.append(row)
    
    if not training_data:
        raise ValueError(f"No data found in training period ({config.TRAINING_START} to {config.TRAINING_END})")
    
    if not analysis_data:
        raise ValueError(f"No data found in analysis period ({config.ANALYSIS_START} to {config.ANALYSIS_END})")
    
    training_df = pd.DataFrame(training_data).reset_index(drop=True)
    analysis_df = pd.DataFrame(analysis_data).reset_index(drop=True)
    
    training_hours = len(training_df)
    if training_hours < config.MIN_TRAINING_HOURS:
        print(f"Warning: Training period has only {training_hours} hours (minimum recommended: {config.MIN_TRAINING_HOURS})")
    
    print(f"Training period: {len(training_df)} samples")
    print(f"Analysis period: {len(analysis_df)} samples")
    
    return training_df, analysis_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns (all columns except Time and DateTime columns)."""
    exclude_patterns = ['time', 'date', 'timestamp']
    feature_columns = []
    
    for col in df.columns:
        if not any(pattern in col.lower() for pattern in exclude_patterns):
            feature_columns.append(col)
    
    return feature_columns


def validate_training_scores(scores: pd.Series, period_name: str = "training") -> None:
    """Validate that training period scores meet requirements."""
    mean_score = scores.mean()
    max_score = scores.max()
    
    print(f"\n{period_name.title()} Period Validation:")
    print(f"Mean score: {mean_score:.2f} (requirement: < {config.MAX_TRAINING_MEAN_SCORE})")
    print(f"Max score: {max_score:.2f} (requirement: < {config.MAX_TRAINING_MAX_SCORE})")
    
    validation_passed = True
    
    if mean_score >= config.MAX_TRAINING_MEAN_SCORE:
        print(f"❌ FAIL: Mean score {mean_score:.2f} >= {config.MAX_TRAINING_MEAN_SCORE}")
        validation_passed = False
    else:
        print(f"✅ PASS: Mean score {mean_score:.2f} < {config.MAX_TRAINING_MEAN_SCORE}")
    
    if max_score >= config.MAX_TRAINING_MAX_SCORE:
        print(f"❌ FAIL: Max score {max_score:.2f} >= {config.MAX_TRAINING_MAX_SCORE}")
        validation_passed = False
    else:
        print(f"✅ PASS: Max score {max_score:.2f} < {config.MAX_TRAINING_MAX_SCORE}")
    
    if not validation_passed:
        raise ValueError(f"{period_name.title()} period validation failed")


def calculate_summary_statistics(scores: pd.Series) -> dict:
    """Calculate summary statistics for anomaly scores."""
    return {
        'count': len(scores),
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        '25%': scores.quantile(0.25),
        '50%': scores.median(),
        '75%': scores.quantile(0.75),
        'max': scores.max()
    }


def print_summary_statistics(scores: pd.Series, title: str = "Anomaly Score Statistics") -> None:
    """Print summary statistics for anomaly scores."""
    stats = calculate_summary_statistics(scores)
    
    print(f"\n{title}:")
    print("-" * len(title))
    for key, value in stats.items():
        if key == 'count':
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.2f}")


def save_results_with_validation(df: pd.DataFrame, output_path: str) -> None:
    """Save results with validation checks."""
    required_columns = [config.ANOMALY_SCORE_COLUMN] + config.TOP_FEATURE_COLUMNS
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    anomaly_scores = df[config.ANOMALY_SCORE_COLUMN]
    
    if not (0 <= anomaly_scores.min() and anomaly_scores.max() <= 100):
        raise ValueError(f"Anomaly scores out of valid range [0, 100]: {anomaly_scores.min():.2f} to {anomaly_scores.max():.2f}")
    
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
