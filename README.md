# 🔍 Time Series Anomaly Detection System

A comprehensive Python system for detecting anomalies in multivariate time series data with **multiple interactive user interfaces** and advanced visualization capabilities.

## 🌟 New Features & Enhanced UI

### 🎛️ Multiple User Interfaces
- **🚀 Interactive Launcher**: Easy-to-use menu system for all features
- **🌐 Streamlit Dashboard**: Modern web-based interface with real-time interaction
- **📊 Real-time Monitor**: Live monitoring dashboard with alerts and notifications
- **📈 Enhanced Demo**: Rich visualizations with animated output and charts
- **🔧 Command Line**: Traditional CLI with improved output formatting

### 🎨 Advanced Visualizations
- **Interactive Time Series Plots**: Plotly-powered charts with zoom, pan, hover
- **Feature Importance Analysis**: Bar charts, pie charts, and heatmaps
- **Real-time Monitoring**: Live updating charts and alert systems  
- **Statistical Dashboards**: Comprehensive analytics with multiple chart types
- **Export Capabilities**: Save charts as PNG, HTML, and interactive formats

### 📊 Dashboard Features
- **Drag-and-drop file upload**
- **Configurable parameters** (contamination rate, thresholds)
- **Interactive filtering and selection**
- **Downloadable results** in multiple formats
- **Real-time metrics** and alerts
- **Responsive design** for desktop and mobile

## 🚀 Quick Start

### Option 1: Interactive Launcher (Recommended)
```bash
python launcher.py
```
This provides a menu-driven interface to access all features!

### Option 2: Streamlit Web Dashboard
```bash
streamlit run dashboard.py
```
Then open http://localhost:8501 in your browser

### Option 3: Real-time Monitoring
```bash
python realtime_monitor.py
```
Then open http://localhost:8050 in your browser

### Option 4: Enhanced Demo
```bash
python demo.py
```

## 📋 Installation

### Quick Install
```bash
pip install -r requirements.txt
```

### Dependencies
- Core: `pandas`, `numpy`, `scikit-learn`
- Visualization: `matplotlib`, `seaborn`, `plotly`
- Web UI: `streamlit`, `dash`, `dash-bootstrap-components`
- Additional: `streamlit-aggrid`, `streamlit-plotly-events`

## 🎯 Features Overview

### 🔍 Core Algorithm
- **Multivariate Analysis**: Processes multiple sensor readings simultaneously
- **Feature Attribution**: Shows which sensors contribute most to each anomaly  
- **0-100 Scoring**: Easy to interpret anomaly scores
- **Training Validation**: Ensures the model works correctly on known normal data
- **Isolation Forest**: Unsupervised anomaly detection without labeled data

### 📊 User Interfaces

#### 1. 🚀 Interactive Launcher
- Menu-driven access to all features
- Dependency management and installation
- Documentation and help system
- File management and batch processing

#### 2. 🌐 Streamlit Dashboard
- **File Upload**: Drag-and-drop CSV files
- **Real-time Analysis**: Instant processing and visualization
- **Interactive Charts**: Zoom, pan, hover, and selection
- **Parameter Control**: Adjust detection sensitivity
- **Export Options**: Download results and charts

#### 3. 📊 Real-time Monitor  
- **Live Data Simulation**: Streaming anomaly detection
- **Alert System**: Configurable thresholds and notifications
- **Metrics Dashboard**: Real-time statistics and KPIs
- **Historical Charts**: Trend analysis and pattern recognition

#### 4. 📈 Enhanced Demo
- **Rich Visualizations**: Multiple chart types and plots
- **Animated Output**: Progress indicators and styled text
- **Comprehensive Analysis**: Statistical summaries and insights
- **Auto-generated Reports**: Detailed findings and recommendations

### 🎨 Visualization Types

#### Static Plots (Matplotlib/Seaborn)
- Time series with highlighted training periods
- Score distributions and histograms
- Feature importance bar charts
- Risk category pie charts
- Statistical summary tables

#### Interactive Charts (Plotly)
- Zoomable time series with hover details
- Multi-panel dashboards with linked views
- 3D scatter plots for multi-dimensional analysis
- Animation and transition effects

#### Real-time Dashboards (Dash)
- Live updating charts and metrics
- Interactive filters and controls
- Alert notifications and status indicators
- Responsive layout for mobile devices

## 📁 Project Structure

```
├── launcher.py              # 🚀 Interactive launcher (NEW)
├── dashboard.py             # 🌐 Streamlit web dashboard (NEW)
├── realtime_monitor.py      # 📊 Real-time monitoring (NEW)
├── demo.py                  # 📈 Enhanced demo with visualizations (UPDATED)
├── anomaly_detector.py      # 🔍 Core detection algorithm
├── config.py               # ⚙️ Configuration parameters
├── utils.py                # 🛠️ Utility functions (ENHANCED)
├── requirements.txt        # 📦 Dependencies (UPDATED)
├── sample_dataset.csv      # 📊 Sample data
└── README.md              # 📖 Documentation (UPDATED)
```

## 🎛️ Configuration Options

### Algorithm Parameters
- **Contamination Rate**: Expected proportion of outliers (0.001-0.1)
- **Feature Threshold**: Minimum contribution for top features (0.1-10.0)
- **Random State**: Reproducibility seed (any integer)

### Visualization Settings
- **Chart Themes**: Light/dark mode, color schemes
- **Update Intervals**: Real-time refresh rates (1-60 seconds)
- **Alert Thresholds**: Custom risk levels (0-100)

### Export Formats
- **CSV**: Comma-separated values
- **JSON**: JavaScript Object Notation with metadata
- **Excel**: Multi-sheet workbooks with summaries
- **Parquet**: Columnar storage for big data tools

## 🎯 Use Cases

### 🏭 Industrial Monitoring
- Equipment health monitoring
- Predictive maintenance alerts
- Process optimization insights
- Quality control automation

### 🔬 Research & Development
- Experimental data analysis
- Pattern discovery in complex datasets
- Hypothesis testing and validation
- Automated report generation

### 📈 Business Intelligence
- KPI anomaly detection
- Customer behavior analysis
- Financial fraud detection
- Operational efficiency monitoring

## 📊 Sample Outputs

### 🎨 Visualizations Generated
- `anomaly_timeseries.png`: Time series analysis
- `feature_importance.png`: Contributing factors
- `advanced_analytics.png`: Statistical dashboard
- `interactive_dashboard.html`: Web-based exploration

### 📄 Reports Created
- `anomaly_report_YYYYMMDD_HHMMSS.txt`: Comprehensive text summary
- `anomaly_results_YYYYMMDD_HHMMSS.csv`: Detailed numerical results
- `anomaly_overview.png`: Quick visual summary

## 🎓 Algorithm Details

### Isolation Forest
- **Principle**: Anomalies are easier to isolate than normal points
- **Advantage**: No need for labeled training data
- **Scalability**: Efficient for large datasets
- **Robustness**: Handles noise and missing values

### Feature Attribution
- **Method**: Statistical deviation analysis
- **Output**: Ranked list of contributing sensors
- **Interpretation**: Higher values = greater contribution to anomaly
- **Threshold**: Configurable minimum contribution level

## ⏰ Time Periods

- **Training Period**: 1/1/2004 0:00 to 1/5/2004 23:59 (120 hours)
- **Analysis Period**: 1/1/2004 0:00 to 1/19/2004 7:59 (439 hours)
- **Format**: MM/DD/YYYY HH:MM

## 🚨 Risk Categories

- **🔴 High Risk**: Score ≥ 75 (Immediate attention required)
- **🟠 Medium Risk**: Score 50-74 (Monitor closely)  
- **🟡 Low Risk**: Score 25-49 (Investigate when possible)
- **🟢 Normal**: Score < 25 (No action needed)

## 🛠️ Troubleshooting

### Common Issues

#### Dependencies Not Installed
```bash
python launcher.py  # Choose option 7
# OR
pip install -r requirements.txt
```

#### Port Already in Use
```bash
# For Streamlit (default 8501)
streamlit run dashboard.py --server.port 8502

# For Dash (default 8050)  
# Edit realtime_monitor.py and change port=8051
```

#### Memory Issues with Large Files
- Use smaller datasets for testing
- Increase system memory or use cloud computing
- Enable data sampling in configuration

### Getting Help
1. Run `python launcher.py` and choose option 8 for documentation
2. Check the generated log files for error details
3. Verify data format matches requirements
4. Ensure all dependencies are correctly installed

## 🔮 Future Enhancements

- **🤖 Machine Learning**: Advanced algorithms (LSTM, Transformer)
- **☁️ Cloud Integration**: AWS, Azure, GCP deployment
- **📱 Mobile App**: Smartphone monitoring interface  
- **🔔 Notifications**: Email, SMS, Slack alerts
- **🎯 Custom Models**: User-defined anomaly detectors
- **📊 Advanced Analytics**: Seasonal decomposition, forecasting

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**🚀 Get started now with: `python launcher.py`**
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
