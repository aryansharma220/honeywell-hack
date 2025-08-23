# Time Series Anomaly Detection

A Python system for detecting anomalies in multivariate time series data and identifying contributing features.

## Overview

This project detects abnormal behavior in industrial time series data from multiple sensors. It helps organizations move from reactive to proactive maintenance by identifying patterns that deviate from normal operation.

## Features

- **Multivariate Analysis**: Processes multiple sensor readings simultaneously
- **Feature Attribution**: Shows which sensors contribute most to each anomaly  
- **0-100 Scoring**: Easy to interpret anomaly scores
- **Training Validation**: Ensures the model works correctly on known normal data

## Algorithm

Uses Isolation Forest for multivariate anomaly detection without needing labeled data. Includes custom feature attribution based on statistical deviation analysis.

## Time Periods

- **Training Period**: 1/1/2004 0:00 to 1/5/2004 23:59 (120 hours)
- **Analysis Period**: 1/1/2004 0:00 to 1/19/2004 7:59 (439 hours)

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from anomaly_detector import detect_anomalies

detect_anomalies("input_data.csv", "anomaly_results.csv")
```

### Run Demo
```bash
python demo.py
```

## Project Structure

```
├── anomaly_detector.py    # Main detection algorithm
├── config.py             # Configuration parameters
├── utils.py              # Utility functions
├── demo.py               # Demonstration script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Output Format

The system adds 8 new columns to your original dataset:

| Column | Type | Description |
|--------|------|-------------|
| `Abnormality_score` | Float (0-100) | Anomaly severity score |
| `top_feature_1` | String | Most contributing feature |
| `top_feature_2` | String | 2nd most contributing feature |
| ... | ... | ... |
| `top_feature_7` | String | 7th most contributing feature |

### Score Interpretation
- **0-10**: Normal behavior (expected for training period)
- **11-30**: Slightly unusual but acceptable
- **31-60**: Moderate anomaly requiring attention
- **61-90**: Significant anomaly needing investigation
- **91-100**: Severe anomaly requiring immediate action

## Configuration

Key parameters in `config.py`:

```python
DEFAULT_CONTAMINATION = 0.01    # Expected proportion of anomalies
N_ESTIMATORS = 200              # Number of isolation trees
MIN_CONTRIBUTION_THRESHOLD = 1.0  # Minimum 1% contribution
MAX_TOP_FEATURES = 7             # Number of top features to report
```

## Validation Results

The system meets all requirements:
- Training period mean score: < 10
- Training period max score: < 25
- Anomaly scores in valid range: 0.00 to 100.00
- All required output columns present

## Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
