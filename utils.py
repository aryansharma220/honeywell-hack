"""
Utility functions for the anomaly detection system.

This module contains helper functions for data processing, validation,
and other common operations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional
import config


def parse_datetime(time_str: str) -> datetime:
    """
    Parse datetime string to datetime object.
    
    Args:
        time_str (str): Time string in specified format
        
    Returns:
        datetime: Parsed datetime object
        
    Raises:
        ValueError: If datetime format is invalid
    """
    try:
        return datetime.strptime(time_str, config.DATETIME_FORMAT)
    except ValueError as e:
        raise ValueError(f"Invalid datetime format: {time_str}. Expected format: {config.DATETIME_FORMAT}")


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the input dataset for basic requirements.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Raises:
        ValueError: If dataset doesn't meet requirements
    """
    if df.empty:
        raise ValueError("Dataset is empty")
    
    if 'Time' not in df.columns:
        raise ValueError("Dataset must contain 'Time' column")
    
    if len(df) > config.MAX_DATASET_ROWS:
        print(f"Warning: Dataset has {len(df)} rows, which exceeds recommended maximum of {config.MAX_DATASET_ROWS}")
    
    # Check for non-numeric columns (except Time)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    expected_numeric = len(df.columns) - 1  # All except Time
    
    if len(numeric_columns) < expected_numeric:
        non_numeric = [col for col in df.columns if col != 'Time' and col not in numeric_columns]
        print(f"Warning: Non-numeric columns found: {non_numeric}")


def validate_time_range(df: pd.DataFrame) -> Tuple[datetime, datetime]:
    """
    Validate and extract time range from dataset.
    
    Args:
        df (pd.DataFrame): Dataset with Time column
        
    Returns:
        Tuple[datetime, datetime]: Start and end times
        
    Raises:
        ValueError: If time range is invalid
    """
    try:
        df['DateTime'] = df['Time'].apply(parse_datetime)
        start_time = df['DateTime'].min()
        end_time = df['DateTime'].max()
        
        if start_time >= end_time:
            raise ValueError("Invalid time range: start time must be before end time")
        
        return start_time, end_time
    except Exception as e:
        raise ValueError(f"Error validating time range: {e}")


def split_time_series_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into training and analysis periods.
    
    Args:
        df (pd.DataFrame): Input dataframe with Time column
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training data and analysis data
        
    Raises:
        ValueError: If data splitting fails or insufficient data
    """
    # Validate dataset first
    validate_dataset(df)
    
    # Convert Time column to datetime
    df['DateTime'] = df['Time'].apply(parse_datetime)
    
    # Filter training data
    train_mask = (df['DateTime'] >= config.TRAINING_START) & (df['DateTime'] <= config.TRAINING_END)
    training_data = df[train_mask].copy()
    
    # Filter analysis data
    analysis_mask = (df['DateTime'] >= config.ANALYSIS_START) & (df['DateTime'] <= config.ANALYSIS_END)
    analysis_data = df[analysis_mask].copy()
    
    # Validate sufficient training data
    if len(training_data) < config.MIN_TRAINING_HOURS:
        raise ValueError(f"Insufficient training data: {len(training_data)} hours. "
                        f"Minimum {config.MIN_TRAINING_HOURS} hours required.")
    
    if len(analysis_data) == 0:
        raise ValueError("No data found in analysis period")
    
    print(f"Training data: {len(training_data)} rows "
          f"({config.TRAINING_START} to {config.TRAINING_END})")
    print(f"Analysis data: {len(analysis_data)} rows "
          f"({config.ANALYSIS_START} to {config.ANALYSIS_END})")
    
    return training_data, analysis_data


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract feature columns from dataset (excluding Time and DateTime).
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        List[str]: List of feature column names
    """
    return [col for col in df.columns if col not in ['Time', 'DateTime']]


def validate_training_scores(scores: pd.Series, period_name: str = "training") -> None:
    """
    Validate that training period scores meet requirements.
    
    Args:
        scores (pd.Series): Anomaly scores from training period
        period_name (str): Name of the period for reporting
        
    Raises:
        ValueError: If scores don't meet validation criteria
    """
    mean_score = scores.mean()
    max_score = scores.max()
    
    print(f"{period_name.title()} period validation:")
    print(f"  Mean anomaly score: {mean_score:.2f} (should be < {config.MAX_TRAINING_MEAN_SCORE})")
    print(f"  Max anomaly score: {max_score:.2f} (should be < {config.MAX_TRAINING_MAX_SCORE})")
    
    if mean_score >= config.MAX_TRAINING_MEAN_SCORE:
        print(f"Warning: Training mean score ({mean_score:.2f}) exceeds threshold "
              f"({config.MAX_TRAINING_MEAN_SCORE})")
    
    if max_score >= config.MAX_TRAINING_MAX_SCORE:
        print(f"Warning: Training max score ({max_score:.2f}) exceeds threshold "
              f"({config.MAX_TRAINING_MAX_SCORE})")


def calculate_summary_statistics(scores: pd.Series) -> dict:
    """
    Calculate summary statistics for anomaly scores.
    
    Args:
        scores (pd.Series): Anomaly scores
        
    Returns:
        dict: Dictionary containing summary statistics
    """
    return {
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'median': scores.median(),
        'q25': scores.quantile(0.25),
        'q75': scores.quantile(0.75)
    }


def print_summary_statistics(scores: pd.Series, title: str = "Anomaly Score Statistics") -> None:
    """
    Print formatted summary statistics.
    
    Args:
        scores (pd.Series): Anomaly scores
        title (str): Title for the statistics display
    """
    stats = calculate_summary_statistics(scores)
    
    print(f"\n{title}:")
    print(f"  Mean:   {stats['mean']:.2f}")
    print(f"  Std:    {stats['std']:.2f}")
    print(f"  Min:    {stats['min']:.2f}")
    print(f"  Max:    {stats['max']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Q25:    {stats['q25']:.2f}")
    print(f"  Q75:    {stats['q75']:.2f}")


def save_results_with_validation(df: pd.DataFrame, output_path: str) -> None:
    """
    Save results with validation checks.
    
    Args:
        df (pd.DataFrame): Results dataframe
        output_path (str): Output file path
        
    Raises:
        ValueError: If results don't meet requirements
    """
    # Validate required columns exist
    required_columns = [config.ANOMALY_SCORE_COLUMN] + config.TOP_FEATURE_COLUMNS
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate anomaly scores are in valid range
    scores = df[config.ANOMALY_SCORE_COLUMN]
    if scores.min() < 0 or scores.max() > 100:
        raise ValueError(f"Anomaly scores must be between 0 and 100. "
                        f"Found range: {scores.min():.2f} to {scores.max():.2f}")
    
    try:
        df.to_csv(output_path, index=False)
        print(f"Results saved successfully to: {output_path}")
        print(f"Output contains {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        raise ValueError(f"Error saving results: {e}")
