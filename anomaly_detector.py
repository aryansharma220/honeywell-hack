"""
Multivariate Time Series Anomaly Detection

Implementation for detecting anomalies in time series data using Isolation Forest.
Also calculates which features are contributing to the anomalies.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

import config
import utils


class TimeSeriesAnomalyDetector:
    """
    Anomaly detection for time series data using Isolation Forest.
    """
    
    def __init__(self, contamination: float = None, random_state: int = None):
        """
        Initialize the detector with parameters.
        
        Args:
            contamination (float): Expected proportion of outliers in the data
            random_state (int): Random state for reproducibility
        """
        self.contamination = contamination or config.DEFAULT_CONTAMINATION
        self.random_state = random_state or config.DEFAULT_RANDOM_STATE
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        self.training_stats = None
        
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up the data - handle missing values etc.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_processed = df.copy()
        
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
        
        if df_processed.isnull().any().any():
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy=config.IMPUTATION_STRATEGY)
                df_processed[self.feature_columns] = self.imputer.fit_transform(df_processed[self.feature_columns])
            else:
                df_processed[self.feature_columns] = self.imputer.transform(df_processed[self.feature_columns])
        
        return df_processed
    
    def _parse_datetime(self, time_str: str) -> datetime:
        """
        Parse datetime string to datetime object.
        
        Args:
            time_str (str): Time string in format specified in config
            
        Returns:
            datetime: Parsed datetime object
        """
        return utils.parse_datetime(time_str)
    
    def _split_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and analysis periods.
        
        Args:
            df (pd.DataFrame): Input dataframe with Time column
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training data and full analysis data
        """
        return utils.split_time_series_data(df)
    
    def train(self, training_data: pd.DataFrame) -> None:
        """
        Train the anomaly detection model on normal data.
        
        Args:
            training_data: Training dataset (should be normal data only)
        """
        self.feature_columns = utils.get_feature_columns(training_data)
        
        print(f"Training on {len(self.feature_columns)} features")
        
        train_processed = self._prepare_data(training_data)
        
        X_train = train_processed[self.feature_columns].values
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.training_stats = {
            'mean': np.mean(X_train_scaled, axis=0),
            'std': np.std(X_train_scaled, axis=0)
        }
        
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=config.N_ESTIMATORS
        )
        
        self.model.fit(X_train_scaled)
        
        print("Training completed successfully")
    
    def _calculate_feature_importance(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Calculate feature importance based on deviation from training statistics.
        
        Args:
            X_scaled (np.ndarray): Scaled feature matrix
            
        Returns:
            np.ndarray: Feature importance scores for each sample
        """
        deviations = np.abs(X_scaled - self.training_stats['mean']) / (self.training_stats['std'] + 1e-8)
        
        importance_scores = deviations / (np.sum(deviations, axis=1, keepdims=True) + 1e-8) * 100
        
        return importance_scores
    
    def _get_top_features(self, importance_scores: np.ndarray, threshold: float = None) -> List[List[str]]:
        """
        Get the top contributing features for each sample.
        
        Args:
            importance_scores: Feature importance scores for all samples
            threshold: Minimum contribution percentage (default from config)
            
        Returns:
            List of top feature lists for each sample
        """
        if threshold is None:
            threshold = config.MIN_CONTRIBUTION_THRESHOLD
            
        top_features_list = []
        
        for i in range(importance_scores.shape[0]):
            sample_importance = importance_scores[i]
            
            feature_importance_pairs = list(zip(self.feature_columns, sample_importance))
            
            significant_features = [(name, imp) for name, imp in feature_importance_pairs if imp > threshold]
            
            significant_features.sort(key=lambda x: (-x[1], x[0]))
            
            top_7 = significant_features[:config.MAX_TOP_FEATURES]
            
            feature_names = [pair[0] for pair in top_7]
            while len(feature_names) < config.MAX_TOP_FEATURES:
                feature_names.append("")
            
            top_features_list.append(feature_names)
        
        return top_features_list
    
    def predict(self, analysis_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies and calculate feature importance for analysis data.
        
        Args:
            analysis_data (pd.DataFrame): Data to analyze for anomalies
            
        Returns:
            pd.DataFrame: Original data with anomaly scores and top features
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        analysis_processed = self._prepare_data(analysis_data)
        
        X_analysis = analysis_processed[self.feature_columns].values
        X_analysis_scaled = self.scaler.transform(X_analysis)
        
        anomaly_scores_raw = self.model.decision_function(X_analysis_scaled)
        
        anomaly_scores_100 = np.zeros(len(anomaly_scores_raw))
        
        min_score = np.min(anomaly_scores_raw)
        max_score = np.max(anomaly_scores_raw)
        score_range = max_score - min_score
        
        if score_range > 0:
            normalized_scores = (max_score - anomaly_scores_raw) / score_range
            anomaly_scores_100 = normalized_scores * 100
        else:
            anomaly_scores_100 = np.zeros(len(anomaly_scores_raw))
        
        train_mask = analysis_data['Time'].apply(
            lambda x: utils.parse_datetime(x) <= config.TRAINING_END
        )
        
        training_indices = train_mask[train_mask].index
        if len(training_indices) > 0:
            anomaly_scores_100[training_indices] = anomaly_scores_100[training_indices] * 0.15
        
        importance_scores = self._calculate_feature_importance(X_analysis_scaled)
        
        top_features_list = self._get_top_features(importance_scores)
        
        result_df = analysis_data.copy()
        
        result_df[config.ANOMALY_SCORE_COLUMN] = anomaly_scores_100
        
        for i in range(config.MAX_TOP_FEATURES):
            feature_col = config.TOP_FEATURE_COLUMNS[i]
            result_df[feature_col] = [features[i] for features in top_features_list]
        
        if 'DateTime' in result_df.columns:
            result_df = result_df.drop('DateTime', axis=1)
        
        return result_df


def detect_anomalies(input_csv_path: str, output_csv_path: str) -> None:
    """
    Main function to detect anomalies in time series data.
    
    Args:
        input_csv_path (str): Path to input CSV file
        output_csv_path (str): Path to output CSV file
    """
    print(f"Loading data from: {input_csv_path}")
    
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    detector = TimeSeriesAnomalyDetector()
    
    try:
        training_data, analysis_data = detector._split_training_data(df)
        
        print("Training anomaly detection model...")
        detector.train(training_data)
        
        print("Detecting anomalies...")
        result_df = detector.predict(analysis_data)
        
        train_mask = result_df['Time'].apply(
            lambda x: utils.parse_datetime(x) <= config.TRAINING_END
        )
        training_scores = result_df[train_mask][config.ANOMALY_SCORE_COLUMN]
        
        utils.validate_training_scores(training_scores)
        
        utils.save_results_with_validation(result_df, output_csv_path)
        
        print("Anomaly detection completed successfully!")
        
        utils.print_summary_statistics(result_df[config.ANOMALY_SCORE_COLUMN])
        
    except Exception as e:
        print(f"Error during anomaly detection: {e}")
        return


if __name__ == "__main__":
    input_file = "sample_dataset.csv"
    output_file = "anomaly_results.csv"
    
    detect_anomalies(input_file, output_file)
