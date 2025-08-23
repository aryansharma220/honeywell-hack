"""
Real-time Monitoring Dashboard using Dash
Interactive monitoring interface with live updates and alerts
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import json

from anomaly_detector import TimeSeriesAnomalyDetector
import config
import utils


class RealTimeMonitor:
    """Real-time anomaly monitoring system"""
    
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.detector = None
        self.current_data = pd.DataFrame()
        self.alerts = []
        self.is_monitoring = False
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🔍 Real-Time Anomaly Monitoring", 
                           className="text-center mb-4 text-primary"),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("🎛️ Control Panel", className="text-center")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("🚀 Start Monitoring", id="start-btn", 
                                             color="success", className="me-2"),
                                    dbc.Button("⏹️ Stop Monitoring", id="stop-btn", 
                                             color="danger", disabled=True),
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Update Interval (seconds):"),
                                    dbc.Input(id="interval-input", type="number", 
                                            value=5, min=1, max=60, step=1),
                                ], width=6)
                            ]),
                            html.Hr(),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Alert Threshold:"),
                                    dbc.Input(id="threshold-input", type="number", 
                                            value=75, min=0, max=100, step=1),
                                ], width=6),
                                dbc.Col([
                                    dbc.Badge("🟢 Normal", id="status-badge", 
                                            color="success", className="fs-5"),
                                ], width=6, className="d-flex align-items-end")
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Metrics Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📊", className="text-center"),
                            html.H2("0", id="total-samples", className="text-center"),
                            html.P("Total Samples", className="text-center text-muted")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🚨", className="text-center"),
                            html.H2("0", id="high-alerts", className="text-center text-danger"),
                            html.P("High Alerts", className="text-center text-muted")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("⚠️", className="text-center"),
                            html.H2("0", id="medium-alerts", className="text-center text-warning"),
                            html.P("Medium Alerts", className="text-center text-muted")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📈", className="text-center"),
                            html.H2("0.0", id="avg-score", className="text-center"),
                            html.P("Avg Score", className="text-center text-muted")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🎯", className="text-center"),
                            html.H2("0.0", id="max-score", className="text-center"),
                            html.P("Max Score", className="text-center text-muted")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("⏰", className="text-center"),
                            html.H2("--:--", id="last-update", className="text-center"),
                            html.P("Last Update", className="text-center text-muted")
                        ])
                    ])
                ], width=2)
            ], className="mb-4"),
            
            # Main Charts Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("📈 Real-Time Anomaly Scores")),
                        dbc.CardBody([
                            dcc.Graph(id="realtime-chart")
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("🚨 Recent Alerts")),
                        dbc.CardBody([
                            html.Div(id="alerts-list", style={"height": "400px", "overflow-y": "auto"})
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Secondary Charts Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("📊 Score Distribution")),
                        dbc.CardBody([
                            dcc.Graph(id="distribution-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("🎯 Top Features")),
                        dbc.CardBody([
                            dcc.Graph(id="features-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0,
                disabled=True
            ),
            
            # Store for data
            dcc.Store(id='monitoring-data', data={}),
            dcc.Store(id='alerts-data', data=[])
            
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup all callbacks for interactivity"""
        
        @self.app.callback(
            [Output('start-btn', 'disabled'),
             Output('stop-btn', 'disabled'),
             Output('interval-component', 'disabled'),
             Output('interval-component', 'interval'),
             Output('status-badge', 'children'),
             Output('status-badge', 'color')],
            [Input('start-btn', 'n_clicks'),
             Input('stop-btn', 'n_clicks'),
             Input('interval-input', 'value')]
        )
        def toggle_monitoring(start_clicks, stop_clicks, interval_value):
            ctx = callback_context
            if not ctx.triggered:
                return False, True, True, interval_value*1000, "🟢 Normal", "success"
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'start-btn' and start_clicks:
                self.start_monitoring()
                return True, False, False, interval_value*1000, "🟡 Monitoring", "warning"
            elif button_id == 'stop-btn' and stop_clicks:
                self.stop_monitoring()
                return False, True, True, interval_value*1000, "🔴 Stopped", "danger"
            
            return False, True, True, interval_value*1000, "🟢 Normal", "success"
        
        @self.app.callback(
            [Output('monitoring-data', 'data'),
             Output('alerts-data', 'data')],
            [Input('interval-component', 'n_intervals')],
            [State('threshold-input', 'value'),
             State('monitoring-data', 'data'),
             State('alerts-data', 'data')]
        )
        def update_data(n_intervals, threshold, current_data, current_alerts):
            if not self.is_monitoring:
                return current_data, current_alerts
            
            # Simulate new data point (in real scenario, this would come from sensors)
            new_data = self.generate_simulated_data()
            
            # Update stored data
            if not current_data:
                current_data = {'timestamps': [], 'scores': [], 'features': []}
            
            current_data['timestamps'].append(new_data['timestamp'])
            current_data['scores'].append(new_data['score'])
            current_data['features'].append(new_data['top_feature'])
            
            # Keep only last 100 data points
            if len(current_data['timestamps']) > 100:
                current_data['timestamps'] = current_data['timestamps'][-100:]
                current_data['scores'] = current_data['scores'][-100:]
                current_data['features'] = current_data['features'][-100:]
            
            # Check for alerts
            if new_data['score'] >= threshold:
                alert = {
                    'timestamp': new_data['timestamp'],
                    'score': new_data['score'],
                    'feature': new_data['top_feature'],
                    'level': 'High' if new_data['score'] >= 75 else 'Medium'
                }
                current_alerts.append(alert)
                
                # Keep only last 20 alerts
                if len(current_alerts) > 20:
                    current_alerts = current_alerts[-20:]
            
            return current_data, current_alerts
        
        @self.app.callback(
            [Output('total-samples', 'children'),
             Output('high-alerts', 'children'),
             Output('medium-alerts', 'children'),
             Output('avg-score', 'children'),
             Output('max-score', 'children'),
             Output('last-update', 'children')],
            [Input('monitoring-data', 'data')]
        )
        def update_metrics(data):
            if not data or not data.get('scores'):
                return "0", "0", "0", "0.0", "0.0", "--:--"
            
            scores = data['scores']
            total = len(scores)
            high_alerts = len([s for s in scores if s >= 75])
            medium_alerts = len([s for s in scores if 50 <= s < 75])
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            last_update = datetime.now().strftime("%H:%M")
            
            return str(total), str(high_alerts), str(medium_alerts), f"{avg_score:.1f}", f"{max_score:.1f}", last_update
        
        @self.app.callback(
            Output('realtime-chart', 'figure'),
            [Input('monitoring-data', 'data')]
        )
        def update_realtime_chart(data):
            if not data or not data.get('scores'):
                # Empty chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name='Anomaly Score'))
                fig.update_layout(title="Waiting for data...", template="plotly_white")
                return fig
            
            timestamps = [datetime.fromisoformat(ts) for ts in data['timestamps']]
            scores = data['scores']
            
            fig = go.Figure()
            
            # Add main line
            fig.add_trace(go.Scatter(
                x=timestamps, y=scores,
                mode='lines+markers',
                name='Anomaly Score',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))
            
            # Add threshold lines
            fig.add_hline(y=75, line_dash="dash", line_color="red", 
                         annotation_text="High Risk", annotation_position="top right")
            fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                         annotation_text="Medium Risk", annotation_position="top right")
            
            fig.update_layout(
                title="Real-Time Anomaly Scores",
                xaxis_title="Time",
                yaxis_title="Anomaly Score",
                template="plotly_white",
                showlegend=True
            )
            
            return fig
        
        @self.app.callback(
            Output('distribution-chart', 'figure'),
            [Input('monitoring-data', 'data')]
        )
        def update_distribution_chart(data):
            if not data or not data.get('scores'):
                fig = go.Figure()
                fig.update_layout(title="Waiting for data...", template="plotly_white")
                return fig
            
            scores = data['scores']
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=scores,
                nbinsx=20,
                name='Score Distribution',
                marker_color='#ff7f0e',
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Score Distribution",
                xaxis_title="Anomaly Score",
                yaxis_title="Frequency",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output('features-chart', 'figure'),
            [Input('monitoring-data', 'data')]
        )
        def update_features_chart(data):
            if not data or not data.get('features'):
                fig = go.Figure()
                fig.update_layout(title="Waiting for data...", template="plotly_white")
                return fig
            
            features = [f for f in data['features'] if f]
            if not features:
                fig = go.Figure()
                fig.update_layout(title="No feature data", template="plotly_white")
                return fig
            
            feature_counts = pd.Series(features).value_counts().head(10)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=feature_counts.values,
                y=feature_counts.index,
                orientation='h',
                name='Feature Count',
                marker_color='#2ca02c'
            ))
            
            fig.update_layout(
                title="Top Contributing Features",
                xaxis_title="Count",
                yaxis_title="Feature",
                template="plotly_white",
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output('alerts-list', 'children'),
            [Input('alerts-data', 'data')]
        )
        def update_alerts_list(alerts):
            if not alerts:
                return [html.P("No alerts yet", className="text-muted text-center")]
            
            alert_components = []
            for alert in reversed(alerts[-10:]):  # Show last 10 alerts, newest first
                color = "danger" if alert['level'] == 'High' else "warning"
                icon = "🚨" if alert['level'] == 'High' else "⚠️"
                
                alert_component = dbc.Alert([
                    html.H6(f"{icon} {alert['level']} Alert", className="alert-heading"),
                    html.P(f"Score: {alert['score']:.2f}", className="mb-1"),
                    html.P(f"Feature: {alert['feature']}", className="mb-1"),
                    html.Small(alert['timestamp'], className="text-muted")
                ], color=color, className="mb-2")
                
                alert_components.append(alert_component)
            
            return alert_components
    
    def start_monitoring(self):
        """Start the monitoring process"""
        self.is_monitoring = True
        print("🚀 Monitoring started...")
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.is_monitoring = False
        print("⏹️ Monitoring stopped.")
    
    def generate_simulated_data(self):
        """Generate simulated sensor data for demo purposes"""
        # In a real scenario, this would read from actual sensors or data streams
        
        # Simulate anomaly score with some randomness
        base_score = np.random.normal(15, 10)  # Normal operation around 15
        
        # Occasionally introduce anomalies
        if np.random.random() < 0.1:  # 10% chance of anomaly
            base_score += np.random.normal(60, 20)  # Spike to anomalous levels
        
        score = max(0, min(100, base_score))  # Clamp to valid range
        
        # Simulate top contributing feature
        features = [
            "ReactorTemperatureDegC", "ReactorPressurekPagauge", "CompressorWorkkW",
            "StripperLevel", "PurgeRateStream9", "ProductSepLevel",
            "TotalFeedStream4", "ReactorLevel", "StripperTemperatureDegC"
        ]
        
        top_feature = np.random.choice(features)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'score': round(score, 2),
            'top_feature': top_feature
        }
    
    def run(self, debug=False, port=8050):
        """Run the dashboard"""
        print(f"🌐 Starting Real-Time Monitoring Dashboard...")
        print(f"🔗 Open http://localhost:{port} in your browser")
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')


def main():
    """Main function to run the monitoring dashboard"""
    monitor = RealTimeMonitor()
    monitor.run(debug=False, port=8050)


if __name__ == "__main__":
    main()
