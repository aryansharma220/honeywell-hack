"""
Demo script for the anomaly detection system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from anomaly_detector import detect_anomalies
import config
import utils


def demonstrate_anomaly_detection():
    """
    Run a demo of the anomaly detection system.
    """
    print("=" * 80)
    print("ANOMALY DETECTION DEMO")
    print("=" * 80)
    
    input_file = "sample_dataset.csv"
    output_file = "anomaly_results_demo.csv"
    
    print(f"\n1. Running anomaly detection on: {input_file}")
    detect_anomalies(input_file, output_file)
    
    print(f"\n2. Analysis of results")
    results_df = pd.read_csv(output_file)
    
    print(f"Results shape: {results_df.shape}")
    print(f"Time range: {results_df['Time'].iloc[0]} to {results_df['Time'].iloc[-1]}")
    
    anomaly_scores = results_df[config.ANOMALY_SCORE_COLUMN]
    utils.print_summary_statistics(anomaly_scores, "Anomaly Score Statistics")
    
    train_mask = results_df['Time'].apply(
        lambda x: utils.parse_datetime(x) <= config.TRAINING_END
    )
    training_scores = anomaly_scores[train_mask]
    utils.print_summary_statistics(training_scores, "Training Period Statistics")
    
    print(f"\n3. Top 10 Anomalies")
    print("-" * 40)
    top_anomalies = results_df.nlargest(10, config.ANOMALY_SCORE_COLUMN)
    
    for i, (_, row) in enumerate(top_anomalies.iterrows(), 1):
        score = row[config.ANOMALY_SCORE_COLUMN]
        time = row['Time']
        top_feature = row[config.TOP_FEATURE_COLUMNS[0]]
        print(f"{i:2d}. {time} | Score: {score:6.2f} | Top Feature: {top_feature}")
    
    print(f"\n4. Most Frequent Contributing Features")
    print("-" * 40)
    
    all_features = []
    for col in config.TOP_FEATURE_COLUMNS:
        all_features.extend(results_df[col].dropna().tolist())
    
    feature_counts = pd.Series(all_features).value_counts()
    feature_counts = feature_counts[feature_counts.index != ""]
    
    for feature, count in feature_counts.head(10).items():
        percentage = (count / len(results_df)) * 100
        print(f"{feature:25s}: {count:4d} times ({percentage:5.1f}%)")
    
    print(f"\n5. Validation Results")
    print("-" * 40)
    
    required_cols = [config.ANOMALY_SCORE_COLUMN] + config.TOP_FEATURE_COLUMNS
    missing_columns = [col for col in required_cols if col not in results_df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
    else:
        print("All required columns present")
    
    min_score = anomaly_scores.min()
    max_score = anomaly_scores.max()
    
    if 0 <= min_score and max_score <= 100:
        print(f"Anomaly scores in valid range: {min_score:.2f} to {max_score:.2f}")
    else:
        print(f"Anomaly scores out of range: {min_score:.2f} to {max_score:.2f}")
    
    train_mean = training_scores.mean()
    train_max = training_scores.max()
    
    if train_mean < config.MAX_TRAINING_MEAN_SCORE:
        print(f"Training mean score: {train_mean:.2f} < {config.MAX_TRAINING_MEAN_SCORE} ✓")
    else:
        print(f"Training mean score: {train_mean:.2f} >= {config.MAX_TRAINING_MEAN_SCORE} ✗")
    
    if train_max < config.MAX_TRAINING_MAX_SCORE:
        print(f"Training max score: {train_max:.2f} < {config.MAX_TRAINING_MAX_SCORE} ✓")
    else:
        print(f"Training max score: {train_max:.2f} >= {config.MAX_TRAINING_MAX_SCORE} ✗")
    
    print(f"\n6. Output Files")
    print("-" * 40)
    print(f"Results saved to: {output_file}")
    print(f"Rows: {len(results_df)}")
    print(f"Columns: {len(results_df.columns)}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_anomaly_detection()
