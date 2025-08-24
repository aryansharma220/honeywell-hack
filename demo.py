"""
Enhanced Demo script for the anomaly detection system with visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

from anomaly_detector import detect_anomalies
import config
import utils


def setup_plotting_style():
    """Setup plotting style for better visualizations"""
    # Use a safe, available style
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
    
    sns.set_palette("husl")
    
    # Configure matplotlib
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'figure.facecolor': 'white'
    })


def print_animated_header(text, char="=", delay=0.05):
    """Print animated header for better visual appeal"""
    import time
    
    border = char * len(text)
    print(f"\n{border}")
    
    # Animate text character by character
    for i, letter in enumerate(text):
        print(letter, end='', flush=True)
        time.sleep(delay)
    
    print(f"\n{border}")


def create_anomaly_time_series_plot(results_df, save_path="anomaly_timeseries.png"):
    """Create comprehensive time series plot"""
    print("Creating time series visualization...")
    
    # Convert time to datetime
    results_df['DateTime'] = pd.to_datetime(results_df['Time'], format=config.DATETIME_FORMAT)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Time Series Anomaly Detection Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Anomaly scores over time
    ax1.plot(results_df['DateTime'], results_df[config.ANOMALY_SCORE_COLUMN], 
             linewidth=1.5, alpha=0.8, color='#2E86AB')
    ax1.axhline(y=75, color='red', linestyle='--', alpha=0.7, label='High Risk (75+)')
    ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Medium Risk (50-75)')
    ax1.axhline(y=25, color='green', linestyle='--', alpha=0.7, label='Low Risk (<25)')
    
    # Highlight training period
    training_end = pd.to_datetime(config.TRAINING_END)
    training_mask = results_df['DateTime'] <= training_end
    ax1.fill_between(results_df[training_mask]['DateTime'], 
                     0, results_df[training_mask][config.ANOMALY_SCORE_COLUMN].max() + 10,
                     alpha=0.2, color='lightblue', label='Training Period')
    
    ax1.set_title('Anomaly Scores Over Time', fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Anomaly Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Anomaly score distribution
    ax2.hist(results_df[config.ANOMALY_SCORE_COLUMN], bins=50, 
             alpha=0.7, color='#A23B72', edgecolor='black')
    ax2.axvline(results_df[config.ANOMALY_SCORE_COLUMN].mean(), 
                color='red', linestyle='--', linewidth=2, label=f'Mean: {results_df[config.ANOMALY_SCORE_COLUMN].mean():.2f}')
    ax2.set_title('Distribution of Anomaly Scores', fontweight='bold')
    ax2.set_xlabel('Anomaly Score')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling statistics
    window_size = min(24, len(results_df) // 10)  # 24-hour window or 10% of data
    rolling_mean = results_df[config.ANOMALY_SCORE_COLUMN].rolling(window=window_size).mean()
    rolling_std = results_df[config.ANOMALY_SCORE_COLUMN].rolling(window=window_size).std()
    
    ax3.plot(results_df['DateTime'], rolling_mean, 
             linewidth=2, color='#F18F01', label=f'{window_size}h Rolling Mean')
    ax3.fill_between(results_df['DateTime'], 
                     rolling_mean - rolling_std, 
                     rolling_mean + rolling_std,
                     alpha=0.3, color='#F18F01', label='±1 Std Dev')
    
    ax3.set_title(f'{window_size}-Hour Rolling Statistics', fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Anomaly Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Time series plot saved as: {save_path}")


def create_feature_importance_visualization(results_df, save_path="feature_importance.png"):
    """Create feature importance visualization"""
    print("Creating feature importance visualization...")
    
    # Extract all features from top feature columns
    all_features = []
    for col in config.TOP_FEATURE_COLUMNS:
        if col in results_df.columns:
            all_features.extend(results_df[col].dropna().tolist())
    
    # Remove empty strings and count
    all_features = [f for f in all_features if f != ""]
    if not all_features:
        print("No feature importance data found")
        return
    
    feature_counts = pd.Series(all_features).value_counts()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Top 15 features bar chart
    top_features = feature_counts.head(15)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    
    bars = ax1.barh(range(len(top_features)), top_features.values, color=colors)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features.index, fontsize=10)
    ax1.set_xlabel('Number of Anomalies')
    ax1.set_title('Top 15 Contributing Features', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_features.values)):
        ax1.text(value + 0.5, i, str(value), va='center', fontweight='bold')
    
    # Plot 2: Feature contribution pie chart
    top_10_features = feature_counts.head(10)
    other_count = feature_counts.iloc[10:].sum() if len(feature_counts) > 10 else 0
    
    if other_count > 0:
        pie_data = list(top_10_features.values) + [other_count]
        pie_labels = list(top_10_features.index) + ['Others']
    else:
        pie_data = top_10_features.values
        pie_labels = top_10_features.index
    
    wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', 
                                       startangle=90, colors=plt.cm.Set3.colors)
    ax2.set_title('Feature Contribution Distribution', fontweight='bold')
    
    # Improve readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Feature importance plot saved as: {save_path}")


def create_advanced_analytics_plot(results_df, save_path="advanced_analytics.png"):
    """Create advanced analytics visualization"""
    print("Creating advanced analytics visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Analytics Dashboard', fontsize=16, fontweight='bold')
    
    anomaly_scores = results_df[config.ANOMALY_SCORE_COLUMN]
    
    # Plot 1: Box plot by risk categories
    risk_categories = []
    for score in anomaly_scores:
        if score >= 75:
            risk_categories.append('High Risk')
        elif score >= 50:
            risk_categories.append('Medium Risk')
        elif score >= 25:
            risk_categories.append('Low Risk')
        else:
            risk_categories.append('Normal')
    
    results_df_temp = results_df.copy()
    results_df_temp['Risk_Category'] = risk_categories
    
    sns.boxplot(data=results_df_temp, x='Risk_Category', y=config.ANOMALY_SCORE_COLUMN, ax=ax1)
    ax1.set_title('Score Distribution by Risk Category', fontweight='bold')
    ax1.set_xlabel('Risk Category')
    ax1.set_ylabel('Anomaly Score')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative anomaly detection
    sorted_scores = np.sort(anomaly_scores)[::-1]  # Sort descending
    cumulative_anomalies = np.arange(1, len(sorted_scores) + 1)
    
    ax2.plot(cumulative_anomalies, sorted_scores, linewidth=2, color='#E74C3C')
    ax2.axhline(y=75, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Medium Risk Threshold')
    ax2.set_title('Cumulative Anomaly Score Distribution', fontweight='bold')
    ax2.set_xlabel('Sample Rank')
    ax2.set_ylabel('Anomaly Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Hourly anomaly pattern
    results_df['DateTime'] = pd.to_datetime(results_df['Time'], format=config.DATETIME_FORMAT)
    results_df['Hour'] = results_df['DateTime'].dt.hour
    
    hourly_stats = results_df.groupby('Hour')[config.ANOMALY_SCORE_COLUMN].agg(['mean', 'max', 'count'])
    
    ax3_twin = ax3.twinx()
    
    bars = ax3.bar(hourly_stats.index, hourly_stats['mean'], alpha=0.7, color='#3498DB', label='Mean Score')
    line = ax3_twin.plot(hourly_stats.index, hourly_stats['count'], color='#E74C3C', 
                        marker='o', linewidth=2, label='Sample Count')
    
    ax3.set_title('Hourly Anomaly Patterns', fontweight='bold')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Mean Anomaly Score', color='#3498DB')
    ax3_twin.set_ylabel('Sample Count', color='#E74C3C')
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot 4: Statistical summary
    stats_data = {
        'Metric': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
        'Value': [
            len(anomaly_scores),
            anomaly_scores.mean(),
            anomaly_scores.std(),
            anomaly_scores.min(),
            anomaly_scores.quantile(0.25),
            anomaly_scores.median(),
            anomaly_scores.quantile(0.75),
            anomaly_scores.max()
        ]
    }
    
    # Create a table-like visualization
    ax4.axis('off')
    table_data = [[metric, f"{value:.2f}" if isinstance(value, float) else str(int(value))] 
                  for metric, value in zip(stats_data['Metric'], stats_data['Value'])]
    
    table = ax4.table(cellText=table_data,
                      colLabels=['Statistic', 'Value'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.4, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(2):
            if i == 0:  # Header
                table[(i, j)].set_facecolor('#3498DB')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
    
    ax4.set_title('Statistical Summary', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Advanced analytics plot saved as: {save_path}")


def create_interactive_plotly_dashboard(results_df):
    """Create interactive Plotly dashboard"""
    print("Creating interactive dashboard...")
    
    # Convert time to datetime
    results_df['DateTime'] = pd.to_datetime(results_df['Time'], format=config.DATETIME_FORMAT)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Anomaly Scores Timeline', 'Score Distribution', 
                       'Feature Importance', 'Risk Categories'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Time series
    fig.add_trace(
        go.Scatter(
            x=results_df['DateTime'],
            y=results_df[config.ANOMALY_SCORE_COLUMN],
            mode='lines+markers',
            name='Anomaly Score',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=3),
            hovertemplate='<b>Time:</b> %{x}<br><b>Score:</b> %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add threshold lines
    fig.add_hline(y=75, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="orange", row=1, col=1)
    
    # Plot 2: Histogram
    fig.add_trace(
        go.Histogram(
            x=results_df[config.ANOMALY_SCORE_COLUMN],
            nbinsx=30,
            name='Distribution',
            marker_color='#ff7f0e',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    # Plot 3: Feature importance (if available)
    all_features = []
    for col in config.TOP_FEATURE_COLUMNS:
        if col in results_df.columns:
            all_features.extend(results_df[col].dropna().tolist())
    
    if all_features:
        feature_counts = pd.Series([f for f in all_features if f != ""]).value_counts().head(10)
        
        fig.add_trace(
            go.Bar(
                x=feature_counts.values,
                y=feature_counts.index,
                orientation='h',
                name='Feature Count',
                marker_color='#2ca02c'
            ),
            row=2, col=1
        )
    
    # Plot 4: Risk categories
    risk_categories = ['Normal', 'Low Risk', 'Medium Risk', 'High Risk']
    risk_counts = [
        len(results_df[results_df[config.ANOMALY_SCORE_COLUMN] < 25]),
        len(results_df[(results_df[config.ANOMALY_SCORE_COLUMN] >= 25) & 
                      (results_df[config.ANOMALY_SCORE_COLUMN] < 50)]),
        len(results_df[(results_df[config.ANOMALY_SCORE_COLUMN] >= 50) & 
                      (results_df[config.ANOMALY_SCORE_COLUMN] < 75)]),
        len(results_df[results_df[config.ANOMALY_SCORE_COLUMN] >= 75])
    ]
    
    colors = ['#2ca02c', '#ffbb33', '#ff7f0e', '#d62728']
    
    fig.add_trace(
        go.Pie(
            labels=risk_categories,
            values=risk_counts,
            name="Risk Distribution",
            marker=dict(colors=colors)
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Interactive Anomaly Detection Dashboard",
        showlegend=True,
        height=800,
        template="plotly_white"
    )
    
    # Save as HTML
    fig.write_html("interactive_dashboard.html")
    print("Interactive dashboard saved as: interactive_dashboard.html")
    
    # Show the plot
    fig.show()


def demonstrate_anomaly_detection():
    """
    Run an enhanced demo of the anomaly detection system with visualizations.
    """
    try:
        setup_plotting_style()
        
        print_animated_header("ENHANCED ANOMALY DETECTION DEMO", "=", 0.02)
        
        input_file = "sample_dataset.csv"
        output_file = "anomaly_results_demo.csv"
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found!")
            print("Make sure the sample dataset is in the current directory")
            return False
        
        print(f"\nStep 1: Running anomaly detection on: {input_file}")
        detect_anomalies(input_file, output_file)
        
        print(f"\nStep 2: Loading and analyzing results")
        results_df = pd.read_csv(output_file)
        
        print(f"Results shape: {results_df.shape}")
        print(f"Time range: {results_df['Time'].iloc[0]} to {results_df['Time'].iloc[-1]}")
        
        anomaly_scores = results_df[config.ANOMALY_SCORE_COLUMN]
        utils.print_summary_statistics(anomaly_scores, "Anomaly Score Statistics")
        
        # Training period analysis
        train_mask = results_df['Time'].apply(
            lambda x: utils.parse_datetime(x) <= config.TRAINING_END
        )
        training_scores = anomaly_scores[train_mask]
        utils.print_summary_statistics(training_scores, "Training Period Statistics")
        
        print(f"\nStep 3: Top 10 Critical Anomalies")
        print("-" * 60)
        top_anomalies = results_df.nlargest(10, config.ANOMALY_SCORE_COLUMN)
        
        for i, (_, row) in enumerate(top_anomalies.iterrows(), 1):
            score = row[config.ANOMALY_SCORE_COLUMN]
            time = row['Time']
            top_feature = row[config.TOP_FEATURE_COLUMNS[0]]
            
            # Add risk level indicator
            if score >= 75:
                risk_level = "[HIGH]"
            elif score >= 50:
                risk_level = "[MED]"
            else:
                risk_level = "[LOW]"
            
            print(f"{risk_level} {i:2d}. {time} | Score: {score:6.2f} | Top Feature: {top_feature}")
        
        print(f"\nStep 4: Most Frequent Contributing Features")
        print("-" * 60)
        
        all_features = []
        for col in config.TOP_FEATURE_COLUMNS:
            all_features.extend(results_df[col].dropna().tolist())
        
        feature_counts = pd.Series(all_features).value_counts()
        feature_counts = feature_counts[feature_counts.index != ""]
        
        for i, (feature, count) in enumerate(feature_counts.head(10).items(), 1):
            percentage = (count / len(results_df)) * 100
            bar_length = int(percentage / 2)  # Scale for display
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"{i:2d}. {feature:20s}: {count:4d} times ({percentage:5.1f}%) {bar}")
        
        print(f"\n✅ Step 5: Validation Results")
        print("-" * 60)
        
        required_cols = [config.ANOMALY_SCORE_COLUMN] + config.TOP_FEATURE_COLUMNS
        missing_columns = [col for col in required_cols if col not in results_df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
        else:
            print("✅ All required columns present")
        
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        
        if 0 <= min_score and max_score <= 100:
            print(f"✅ Anomaly scores in valid range: {min_score:.2f} to {max_score:.2f}")
        else:
            print(f"❌ Anomaly scores out of range: {min_score:.2f} to {max_score:.2f}")
        
        train_mean = training_scores.mean()
        train_max = training_scores.max()
        
        if train_mean < config.MAX_TRAINING_MEAN_SCORE:
            print(f"✅ Training mean score: {train_mean:.2f} < {config.MAX_TRAINING_MEAN_SCORE}")
        else:
            print(f"❌ Training mean score: {train_mean:.2f} >= {config.MAX_TRAINING_MEAN_SCORE}")
        
        if train_max < config.MAX_TRAINING_MAX_SCORE:
            print(f"✅ Training max score: {train_max:.2f} < {config.MAX_TRAINING_MAX_SCORE}")
        else:
            print(f"❌ Training max score: {train_max:.2f} >= {config.MAX_TRAINING_MAX_SCORE}")
        
        print(f"\n📊 Step 6: Creating Visualizations")
        print("-" * 60)
        
        # Create all visualizations
        create_anomaly_time_series_plot(results_df)
        create_feature_importance_visualization(results_df)
        create_advanced_analytics_plot(results_df)
        
        # Create interactive dashboard
        try:
            create_interactive_plotly_dashboard(results_df)
        except Exception as e:
            print(f"⚠️  Could not create interactive dashboard: {e}")
        
        print(f"\n💾 Step 7: Output Files")
        print("-" * 60)
        print(f"📄 Results saved to: {output_file}")
        print(f"📊 Rows: {len(results_df):,}")
        print(f"📈 Columns: {len(results_df.columns)}")
        print(f"📸 Visualization files created:")
        print(f"   • anomaly_timeseries.png")
        print(f"   • feature_importance.png")
        print(f"   • advanced_analytics.png")
        print(f"   • interactive_dashboard.html")
        
        print_animated_header("ENHANCED DEMO COMPLETED SUCCESSFULLY!", "=", 0.02)
        print("\nNext steps:")
        print("   • Open the generated PNG files to view static plots")
        print("   • Open interactive_dashboard.html in your browser for interactive analysis")
        print("   • Run 'streamlit run dashboard.py' for the web dashboard")
        print("   • Upload your own data files for analysis!")
        
        return True

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Make sure all dependencies are installed:")
        print("   pip install pandas numpy matplotlib seaborn plotly scikit-learn")
        return False


if __name__ == "__main__":
    demonstrate_anomaly_detection()
