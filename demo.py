"""
Demonstration script for the Multivariate Time Series Anomaly Detection System

This script demonstrates how to use the anomaly detection system
and provides examples of the output format.
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
    Demonstrate the anomaly detection system with sample data.
    """
    print("=" * 80)
    print("MULTIVARIATE TIME SERIES ANOMALY DETECTION DEMONSTRATION")
    print("=" * 80)
    
    # File paths
    input_file = "sample_dataset.csv"
    output_file = "anomaly_results_demo.csv"
    
    print(f"\n1. Running anomaly detection on: {input_file}")
    print("-" * 60)
    
    # Run anomaly detection
    detect_anomalies(input_file, output_file)
    
    print(f"\n2. Analyzing results from: {output_file}")
    print("-" * 60)
    
    # Load and analyze results
    results_df = pd.read_csv(output_file)
    
    # Display basic information
    print(f"Results shape: {results_df.shape}")
    print(f"Time range: {results_df['Time'].iloc[0]} to {results_df['Time'].iloc[-1]}")
    
    # Analyze anomaly scores
    anomaly_scores = results_df[config.ANOMALY_SCORE_COLUMN]
    utils.print_summary_statistics(anomaly_scores, "Overall Anomaly Score Statistics")
    
    # Analyze training period separately
    train_mask = results_df['Time'].apply(
        lambda x: utils.parse_datetime(x) <= config.TRAINING_END
    )
    training_scores = anomaly_scores[train_mask]
    utils.print_summary_statistics(training_scores, "Training Period Anomaly Score Statistics")
    
    # Find top anomalies
    print(f"\n3. Top 10 Anomalies")
    print("-" * 60)
    top_anomalies = results_df.nlargest(10, config.ANOMALY_SCORE_COLUMN)
    
    for i, (idx, row) in enumerate(top_anomalies.iterrows(), 1):
        print(f"{i:2d}. Time: {row['Time']:<15} Score: {row[config.ANOMALY_SCORE_COLUMN]:6.2f}")
        features = [row[col] for col in config.TOP_FEATURE_COLUMNS if row[col] != ""]
        print(f"    Top contributing features: {', '.join(features)}")
    
    # Analyze feature frequency
    print(f"\n4. Most Frequently Contributing Features")
    print("-" * 60)
    
    feature_counts = {}
    for col in config.TOP_FEATURE_COLUMNS:
        for feature in results_df[col]:
            if feature != "":
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    # Sort by frequency
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("Feature Name                                  Frequency")
    print("-" * 60)
    for feature, count in sorted_features[:15]:
        print(f"{feature:<45} {count:>8}")
    
    # Show sample output format
    print(f"\n5. Sample Output Format")
    print("-" * 60)
    sample_cols = ['Time', config.ANOMALY_SCORE_COLUMN] + config.TOP_FEATURE_COLUMNS
    print(results_df[sample_cols].head(10).to_string(index=False))
    
    # Validate requirements
    print(f"\n6. Validation Against Requirements")
    print("-" * 60)
    
    # Check training period scores
    utils.validate_training_scores(training_scores)
    
    # Check output format
    required_columns = [config.ANOMALY_SCORE_COLUMN] + config.TOP_FEATURE_COLUMNS
    missing_columns = [col for col in required_columns if col not in results_df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
    else:
        print("All required columns present")
    
    # Check score range
    min_score = anomaly_scores.min()
    max_score = anomaly_scores.max()
    
    if 0 <= min_score and max_score <= 100:
        print(f"Anomaly scores in valid range: {min_score:.2f} to {max_score:.2f}")
    else:
        print(f"Anomaly scores out of range: {min_score:.2f} to {max_score:.2f}")
    
    # Check training validation
    train_mean = training_scores.mean()
    train_max = training_scores.max()
    
    if train_mean < config.MAX_TRAINING_MEAN_SCORE:
        print(f"Training mean score: {train_mean:.2f} < {config.MAX_TRAINING_MEAN_SCORE}")
    else:
        print(f"Training mean score: {train_mean:.2f} >= {config.MAX_TRAINING_MEAN_SCORE}")
    
    if train_max < config.MAX_TRAINING_MAX_SCORE:
        print(f"Training max score: {train_max:.2f} < {config.MAX_TRAINING_MAX_SCORE}")
    else:
        print(f"Training max score: {train_max:.2f} >= {config.MAX_TRAINING_MAX_SCORE}")
    
    print(f"\n7. Output Files Generated")
    print("-" * 60)
    print(f"Main output: {output_file}")
    print(f"   - Contains {len(results_df)} rows")
    print(f"   - Contains {len(results_df.columns)} columns")
    print(f"   - All original columns preserved")
    print(f"   - Added 8 new columns: 1 score + 7 feature columns")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


def create_usage_example():
    """
    Create a simple usage example script.
    """
    usage_script = '''"""
Simple Usage Example for Anomaly Detection System

Run this script to detect anomalies in your time series data.
"""

from anomaly_detector import detect_anomalies

# Define input and output file paths
input_csv = "your_dataset.csv"  # Replace with your CSV file path
output_csv = "anomaly_results.csv"  # Output will be saved here

# Run anomaly detection
print("Starting anomaly detection...")
detect_anomalies(input_csv, output_csv)
print("Anomaly detection completed!")

# The output CSV will contain:
# - All original columns from input
# - Abnormality_score: Float values from 0.0 to 100.0
# - top_feature_1 through top_feature_7: Contributing feature names
'''
    
    with open("usage_example.py", "w") as f:
        f.write(usage_script)
    
    print("Created usage_example.py")


if __name__ == "__main__":
    demonstrate_anomaly_detection()
    create_usage_example()
