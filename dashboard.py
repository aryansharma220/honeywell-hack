"""
Streamlit Dashboard for Time Series Anomaly Detection
Interactive web interface for visualizing and analyzing anomalies
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64

from anomaly_detector import detect_anomalies, TimeSeriesAnomalyDetector
import config
import utils


def setup_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="🔍 Anomaly Detection Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def create_sidebar():
    """Create sidebar with controls and information"""
    st.sidebar.title("🎛️ Control Panel")
    
    st.sidebar.markdown("---")
    
    # File upload section
    st.sidebar.subheader("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your time series data for anomaly detection"
    )
    
    # Configuration section
    st.sidebar.subheader("⚙️ Configuration")
    
    contamination = st.sidebar.slider(
        "Contamination Rate",
        min_value=0.001,
        max_value=0.1,
        value=config.DEFAULT_CONTAMINATION,
        step=0.001,
        help="Expected proportion of outliers in the data"
    )
    
    threshold = st.sidebar.slider(
        "Feature Importance Threshold",
        min_value=0.1,
        max_value=10.0,
        value=config.MIN_CONTRIBUTION_THRESHOLD,
        step=0.1,
        help="Minimum contribution percentage for top features"
    )
    
    # Analysis options
    st.sidebar.subheader("📊 Analysis Options")
    show_training_period = st.sidebar.checkbox("Highlight Training Period", value=True)
    show_feature_importance = st.sidebar.checkbox("Show Feature Importance", value=True)
    show_correlations = st.sidebar.checkbox("Show Feature Correlations", value=False)
    
    return uploaded_file, contamination, threshold, show_training_period, show_feature_importance, show_correlations


def create_metrics_dashboard(results_df):
    """Create a metrics dashboard with key statistics"""
    anomaly_scores = results_df[config.ANOMALY_SCORE_COLUMN]
    
    # Calculate metrics
    total_samples = len(results_df)
    high_anomalies = len(results_df[anomaly_scores > 75])
    medium_anomalies = len(results_df[(anomaly_scores > 50) & (anomaly_scores <= 75)])
    avg_score = anomaly_scores.mean()
    max_score = anomaly_scores.max()
    
    # Create columns for metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Samples</h3>
            <h2>{total_samples:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🚨 High Risk</h3>
            <h2>{high_anomalies}</h2>
            <p>Score > 75</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ Medium Risk</h3>
            <h2>{medium_anomalies}</h2>
            <p>Score 50-75</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Avg Score</h3>
            <h2>{avg_score:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Max Score</h3>
            <h2>{max_score:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)


def create_time_series_plot(results_df, show_training_period=True):
    """Create interactive time series plot of anomaly scores"""
    # Convert time column to datetime
    results_df['DateTime'] = pd.to_datetime(results_df['Time'], format=config.DATETIME_FORMAT)
    
    # Create the main plot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Anomaly Scores Over Time', 'Score Distribution'),
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1
    )
    
    # Add anomaly score line
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
    fig.add_hline(y=75, line_dash="dash", line_color="red", 
                  annotation_text="High Risk Threshold", row=1, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Risk Threshold", row=1, col=1)
    
    # Highlight training period if requested
    if show_training_period:
        training_end = pd.to_datetime(config.TRAINING_END)
        fig.add_vrect(
            x0=results_df['DateTime'].min(),
            x1=training_end,
            fillcolor="lightblue",
            opacity=0.3,
            annotation_text="Training Period",
            annotation_position="top left",
            row=1, col=1
        )
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=results_df[config.ANOMALY_SCORE_COLUMN],
            nbinsx=50,
            name='Score Distribution',
            marker_color='#ff7f0e',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Time Series Anomaly Analysis Dashboard',
        height=700,
        showlegend=True,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Anomaly Score", row=1, col=1)
    fig.update_xaxes(title_text="Anomaly Score", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    return fig


def create_feature_importance_plot(results_df):
    """Create feature importance visualization"""
    # Count feature occurrences
    all_features = []
    for col in config.TOP_FEATURE_COLUMNS:
        if col in results_df.columns:
            all_features.extend(results_df[col].dropna().tolist())
    
    # Remove empty strings and count
    all_features = [f for f in all_features if f != ""]
    feature_counts = pd.Series(all_features).value_counts().head(15)
    
    # Create bar plot
    fig = px.bar(
        x=feature_counts.values,
        y=feature_counts.index,
        orientation='h',
        title='Top Contributing Features to Anomalies',
        labels={'x': 'Number of Anomalies', 'y': 'Feature Name'},
        color=feature_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        template="plotly_white"
    )
    
    return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap for numerical features"""
    # Get numerical columns only
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove anomaly score column if present
    if config.ANOMALY_SCORE_COLUMN in numerical_cols:
        numerical_cols.remove(config.ANOMALY_SCORE_COLUMN)
    
    # Limit to top 20 features for readability
    if len(numerical_cols) > 20:
        numerical_cols = numerical_cols[:20]
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    
    fig.update_layout(
        width=800,
        height=600,
        template="plotly_white"
    )
    
    return fig


def create_anomaly_details_table(results_df, top_n=20):
    """Create detailed table of top anomalies"""
    top_anomalies = results_df.nlargest(top_n, config.ANOMALY_SCORE_COLUMN)
    
    # Select relevant columns for display
    display_cols = ['Time', config.ANOMALY_SCORE_COLUMN] + config.TOP_FEATURE_COLUMNS[:3]
    display_df = top_anomalies[display_cols].round(2)
    
    return display_df


def process_uploaded_file(uploaded_file, contamination, threshold):
    """Process uploaded file and run anomaly detection"""
    try:
        # Read uploaded file
        df = pd.read_csv(uploaded_file)
        
        # Validate dataset
        utils.validate_dataset(df)
        
        # Run anomaly detection
        with st.spinner("🔄 Running anomaly detection..."):
            # Create temporary detector with custom parameters
            detector = TimeSeriesAnomalyDetector(contamination=contamination)
            
            # Split data
            training_data, analysis_data = detector._split_training_data(df)
            
            # Train and predict
            detector.train(training_data)
            results_df = detector.predict(analysis_data)
        
        return results_df, None
        
    except Exception as e:
        return None, str(e)


def main():
    """Main dashboard function"""
    setup_page()
    
    # Header
    st.title("🔍 Time Series Anomaly Detection Dashboard")
    
    # Create sidebar
    uploaded_file, contamination, threshold, show_training_period, show_feature_importance, show_correlations = create_sidebar()
    
    # Main content
    if uploaded_file is not None:
        # Process uploaded file
        results_df, error = process_uploaded_file(uploaded_file, contamination, threshold)
        
        if error:
            st.error(f"❌ Error: {error}")
            return
        
        st.success("✅ Analysis completed successfully!")
        
        # Create metrics dashboard
        st.subheader("📊 Key Metrics")
        create_metrics_dashboard(results_df)
        
        # Create time series plot
        st.subheader("📈 Time Series Analysis")
        time_series_fig = create_time_series_plot(results_df, show_training_period)
        st.plotly_chart(time_series_fig, use_container_width=True)
        
        # Feature importance plot
        if show_feature_importance:
            st.subheader("🎯 Feature Importance Analysis")
            feature_fig = create_feature_importance_plot(results_df)
            st.plotly_chart(feature_fig, use_container_width=True)
        
        # Correlation heatmap
        if show_correlations:
            st.subheader("🔗 Feature Correlations")
            # Load original data for correlation analysis
            original_df = pd.read_csv(uploaded_file)
            corr_fig = create_correlation_heatmap(original_df)
            st.plotly_chart(corr_fig, use_container_width=True)
        
        # Detailed anomaly table
        st.subheader("📋 Top Anomalies Details")
        anomaly_table = create_anomaly_details_table(results_df)
        st.dataframe(anomaly_table, use_container_width=True)
        
        # Download results
        st.subheader("💾 Download Results")
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Anomaly Results (CSV)",
            data=csv_data,
            file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_csv"
        )
        
    else:
        # Show demo option when no file is uploaded
        st.info("👆 Please upload a CSV file using the sidebar to get started!")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Run Demo with Sample Data", key="demo_button"):
                try:
                    # Run demo with sample data
                    with st.spinner("🔄 Running demo analysis..."):
                        detect_anomalies("sample_dataset.csv", "anomaly_results_demo.csv")
                    
                    st.success("✅ Demo completed! Please refresh the page to see results.")
                    
                except Exception as e:
                    st.error(f"❌ Demo failed: {str(e)}")
        
        # Show sample data info
        st.subheader("📖 About This Tool")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 What it does:
            - **Detects anomalies** in multivariate time series data
            - **Identifies contributing features** for each anomaly
            - **Provides 0-100 scoring** for easy interpretation
            - **Validates training period** performance
            
            ### 📊 Features:
            - Interactive time series visualization
            - Feature importance analysis
            - Real-time metrics dashboard
            - Downloadable results
            """)
        
        with col2:
            st.markdown("""
            ### 📋 Data Requirements:
            - CSV file with **Time** column
            - Multiple numerical sensor columns
            - Time format: `MM/DD/YYYY HH:MM`
            - Training period: 1/1/2004 - 1/5/2004
            
            ### 🔧 Algorithm:
            - **Isolation Forest** for anomaly detection
            - **Statistical deviation** for feature attribution
            - **Standardized scoring** (0-100 scale)
            """)


if __name__ == "__main__":
    main()
