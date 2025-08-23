"""
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
