"""
Utility functions for the anomaly detection project.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional
import json
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


def create_quick_visualization(df: pd.DataFrame, save_plots: bool = True) -> None:
    """
    Create quick visualization plots for anomaly detection results.
    
    Args:
        df: DataFrame with anomaly detection results
        save_plots: Whether to save plots to files
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Setup style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Convert time to datetime if not already
        if 'DateTime' not in df.columns:
            df['DateTime'] = pd.to_datetime(df['Time'], format=config.DATETIME_FORMAT)
        
        anomaly_scores = df[config.ANOMALY_SCORE_COLUMN]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Anomaly Detection Results Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: Time series
        ax1.plot(df['DateTime'], anomaly_scores, linewidth=1, alpha=0.8, color='#2E86AB')
        ax1.axhline(y=75, color='red', linestyle='--', alpha=0.7, label='High Risk')
        ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Medium Risk')
        ax1.set_title('Anomaly Scores Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Anomaly Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Highlight training period
        training_end = pd.to_datetime(config.TRAINING_END)
        training_mask = df['DateTime'] <= training_end
        if training_mask.any():
            ax1.fill_between(df[training_mask]['DateTime'], 
                           0, anomaly_scores.max() + 5,
                           alpha=0.2, color='lightblue', label='Training Period')
        
        # Plot 2: Score distribution
        ax2.hist(anomaly_scores, bins=30, alpha=0.7, color='#A23B72', edgecolor='black')
        ax2.axvline(anomaly_scores.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {anomaly_scores.mean():.2f}')
        ax2.set_title('Score Distribution')
        ax2.set_xlabel('Anomaly Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Risk categories
        risk_counts = {
            'Normal (<25)': len(df[anomaly_scores < 25]),
            'Low (25-50)': len(df[(anomaly_scores >= 25) & (anomaly_scores < 50)]),
            'Medium (50-75)': len(df[(anomaly_scores >= 50) & (anomaly_scores < 75)]),
            'High (75+)': len(df[anomaly_scores >= 75])
        }
        
        colors = ['#2ca02c', '#ffbb33', '#ff7f0e', '#d62728']
        wedges, texts, autotexts = ax3.pie(risk_counts.values(), labels=risk_counts.keys(), 
                                          autopct='%1.1f%%', startangle=90, colors=colors)
        ax3.set_title('Risk Category Distribution')
        
        # Plot 4: Top features
        all_features = []
        for col in config.TOP_FEATURE_COLUMNS:
            if col in df.columns:
                all_features.extend(df[col].dropna().tolist())
        
        if all_features:
            feature_counts = pd.Series([f for f in all_features if f != ""]).value_counts().head(10)
            
            if not feature_counts.empty:
                y_pos = np.arange(len(feature_counts))
                ax4.barh(y_pos, feature_counts.values, color='#2ca02c', alpha=0.7)
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(feature_counts.index, fontsize=8)
                ax4.set_xlabel('Contribution Count')
                ax4.set_title('Top Contributing Features')
                ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No feature data available', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Top Contributing Features')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('anomaly_overview.png', dpi=300, bbox_inches='tight')
            print("📊 Quick visualization saved as: anomaly_overview.png")
        
        plt.show()
        
    except ImportError:
        print("⚠️  Matplotlib/Seaborn not available for visualization")
    except Exception as e:
        print(f"⚠️  Error creating visualization: {e}")


def generate_summary_report(df: pd.DataFrame, output_file: str = None) -> str:
    """
    Generate a comprehensive text summary report.
    
    Args:
        df: DataFrame with anomaly detection results
        output_file: Optional file path to save the report
        
    Returns:
        str: The generated report as a string
    """
    anomaly_scores = df[config.ANOMALY_SCORE_COLUMN]
    
    # Calculate statistics
    stats = calculate_summary_statistics(anomaly_scores)
    
    # Risk categories
    high_risk = len(df[anomaly_scores >= 75])
    medium_risk = len(df[(anomaly_scores >= 50) & (anomaly_scores < 75)])
    low_risk = len(df[(anomaly_scores >= 25) & (anomaly_scores < 50)])
    normal = len(df[anomaly_scores < 25])
    
    # Top anomalies
    top_anomalies = df.nlargest(5, config.ANOMALY_SCORE_COLUMN)
    
    # Feature analysis
    all_features = []
    for col in config.TOP_FEATURE_COLUMNS:
        if col in df.columns:
            all_features.extend(df[col].dropna().tolist())
    
    feature_counts = pd.Series([f for f in all_features if f != ""]).value_counts().head(5)
    
    # Generate report
    report = f"""
{'='*80}
                    ANOMALY DETECTION SUMMARY REPORT
{'='*80}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 DATASET OVERVIEW
{'─'*40}
• Total Samples: {stats['count']:,}
• Time Range: {df['Time'].iloc[0]} to {df['Time'].iloc[-1]}
• Analysis Period: {(pd.to_datetime(df['Time'].iloc[-1], format=config.DATETIME_FORMAT) - pd.to_datetime(df['Time'].iloc[0], format=config.DATETIME_FORMAT)).days + 1} days

📈 ANOMALY SCORE STATISTICS
{'─'*40}
• Mean Score: {stats['mean']:.2f}
• Standard Deviation: {stats['std']:.2f}
• Minimum Score: {stats['min']:.2f}
• Maximum Score: {stats['max']:.2f}
• Median Score: {stats['50%']:.2f}
• 75th Percentile: {stats['75%']:.2f}
• 95th Percentile: {anomaly_scores.quantile(0.95):.2f}

🚨 RISK ASSESSMENT
{'─'*40}
• 🔴 High Risk (Score ≥ 75):     {high_risk:,} samples ({high_risk/len(df)*100:.1f}%)
• 🟠 Medium Risk (50 ≤ Score < 75): {medium_risk:,} samples ({medium_risk/len(df)*100:.1f}%)
• 🟡 Low Risk (25 ≤ Score < 50):    {low_risk:,} samples ({low_risk/len(df)*100:.1f}%)
• 🟢 Normal (Score < 25):        {normal:,} samples ({normal/len(df)*100:.1f}%)

🎯 TOP 5 CRITICAL ANOMALIES
{'─'*40}
"""
    
    for i, (_, row) in enumerate(top_anomalies.iterrows(), 1):
        score = row[config.ANOMALY_SCORE_COLUMN]
        time = row['Time']
        top_feature = row[config.TOP_FEATURE_COLUMNS[0]] if config.TOP_FEATURE_COLUMNS[0] in row else "N/A"
        report += f"  {i}. {time} | Score: {score:6.2f} | Feature: {top_feature}\n"
    
    if not feature_counts.empty:
        report += f"""
🔧 TOP 5 CONTRIBUTING FEATURES
{'─'*40}
"""
        for i, (feature, count) in enumerate(feature_counts.items(), 1):
            percentage = (count / len(df)) * 100
            report += f"  {i}. {feature:30s}: {count:4d} occurrences ({percentage:5.1f}%)\n"
    
    # Training period analysis
    try:
        train_mask = df['Time'].apply(lambda x: parse_datetime(x) <= config.TRAINING_END)
        if train_mask.any():
            training_scores = anomaly_scores[train_mask]
            train_stats = calculate_summary_statistics(training_scores)
            
            report += f"""
🎓 TRAINING PERIOD ANALYSIS
{'─'*40}
• Training Samples: {train_stats['count']:,}
• Training Mean Score: {train_stats['mean']:.2f}
• Training Max Score: {train_stats['max']:.2f}
• Validation Status: {'✅ PASSED' if train_stats['mean'] < config.MAX_TRAINING_MEAN_SCORE and train_stats['max'] < config.MAX_TRAINING_MAX_SCORE else '❌ FAILED'}
"""
    except:
        report += f"\n⚠️  Training period analysis not available\n"
    
    report += f"""
📋 RECOMMENDATIONS
{'─'*40}
"""
    
    if high_risk > 0:
        report += f"• 🚨 URGENT: Investigate {high_risk} high-risk anomalies immediately\n"
    
    if medium_risk > len(df) * 0.1:  # More than 10% medium risk
        report += f"• ⚠️  ATTENTION: High number of medium-risk anomalies ({medium_risk}) detected\n"
    
    if stats['max'] > 90:
        report += f"• 🔍 INVESTIGATE: Extremely high anomaly score detected ({stats['max']:.2f})\n"
    
    if not feature_counts.empty:
        top_feature = feature_counts.index[0]
        report += f"• 🎯 FOCUS: Monitor '{top_feature}' sensor - most frequent contributor\n"
    
    report += f"""
{'='*80}
                           END OF REPORT
{'='*80}
"""
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"📄 Summary report saved to: {output_file}")
    
    return report


def export_data_for_external_tools(df: pd.DataFrame, format_type: str = "json") -> str:
    """
    Export data in various formats for external analysis tools.
    
    Args:
        df: DataFrame with results
        format_type: Export format ('json', 'parquet', 'excel')
        
    Returns:
        str: Path to exported file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        if format_type.lower() == "json":
            output_file = f"anomaly_results_{timestamp}.json"
            
            # Convert to JSON-friendly format
            export_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'total_samples': len(df),
                    'time_range': {
                        'start': df['Time'].iloc[0],
                        'end': df['Time'].iloc[-1]
                    }
                },
                'statistics': calculate_summary_statistics(df[config.ANOMALY_SCORE_COLUMN]),
                'data': df.to_dict('records')
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
        elif format_type.lower() == "parquet":
            output_file = f"anomaly_results_{timestamp}.parquet"
            df.to_parquet(output_file, index=False)
            
        elif format_type.lower() == "excel":
            output_file = f"anomaly_results_{timestamp}.xlsx"
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Main data
                df.to_excel(writer, sheet_name='Anomaly_Results', index=False)
                
                # Summary statistics
                stats_df = pd.DataFrame([calculate_summary_statistics(df[config.ANOMALY_SCORE_COLUMN])]).T
                stats_df.columns = ['Value']
                stats_df.to_excel(writer, sheet_name='Statistics')
                
                # Top anomalies
                top_anomalies = df.nlargest(20, config.ANOMALY_SCORE_COLUMN)
                top_anomalies.to_excel(writer, sheet_name='Top_Anomalies', index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        print(f"💾 Data exported to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return ""
