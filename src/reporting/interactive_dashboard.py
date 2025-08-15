#!/usr/bin/env python
"""
Phase 4: Interactive HTML Dashboard with Plotly/Dash
Comprehensive interactive dashboard for robust policy evaluation results visualization.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback_context
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import warnings

# Import from local modules
from src.analysis.descriptive_01 import DescriptiveAnalyzer
from src.analysis.causal_02 import CausalAnalyzer  
from src.analysis.robustness_03 import RobustnessAnalyzer
from src.reporting.diagnostics import DiagnosticReporter
from src.reporting.method_recommendation import MethodRecommendationEngine


class InteractiveDashboard:
    """
    Interactive dashboard for comprehensive policy evaluation results.
    
    Features:
    - Multi-tab interface for different analysis components
    - Dynamic filtering and visualization controls
    - Method comparison and diagnostics
    - Export capabilities for figures and data
    """
    
    def __init__(self, data_path: str = "data/final/analysis_panel.csv", output_dir: Path = None):
        """
        Initialize interactive dashboard.
        
        Args:
            data_path: Path to analysis dataset
            output_dir: Directory for dashboard outputs
        """
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.output_dir = Path(output_dir) if output_dir else Path("output/dashboard")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        self.descriptive_analyzer = DescriptiveAnalyzer()
        self.causal_analyzer = CausalAnalyzer()
        self.robustness_analyzer = RobustnessAnalyzer()
        self.diagnostic_reporter = DiagnosticReporter(data_path)
        self.method_engine = MethodRecommendationEngine()
        
        # Load analysis results
        self._load_analysis_results()
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[
            'https://codepen.io/chriddyp/pen/bWLwgP.css',
            'https://use.fontawesome.com/releases/v5.8.1/css/all.css'
        ])
        
        # Setup dashboard layout
        self._setup_layout()
        self._setup_callbacks()
        
        self.logger = logging.getLogger(__name__)
        
    def _load_analysis_results(self):
        """Load pre-computed analysis results."""
        try:
            # Load descriptive results
            self.descriptive_analyzer.load_data()
            self.descriptive_results = self.descriptive_analyzer.generate_summary_statistics()
            
            # Load causal analysis results
            self.causal_analyzer.load_data()
            self.causal_results = {}
            
            # Load robustness results  
            self.robustness_analyzer.load_data()
            self.robustness_results = {}
            
            # Generate diagnostic report
            self.diagnostic_results = self.diagnostic_reporter.generate_comprehensive_report()
            
            # Get method recommendations
            self.method_recommendations = self.method_engine.recommend_methods(self.data_path)
            
        except Exception as e:
            self.logger.warning(f"Could not load all analysis results: {e}")
            # Initialize with empty results for graceful degradation
            self.descriptive_results = {}
            self.causal_results = {}
            self.robustness_results = {}
            self.diagnostic_results = {}
            self.method_recommendations = {}
    
    def _setup_layout(self):
        """Setup dashboard layout with tabs and components."""
        
        # Header
        header = html.Div([
            html.H1("State Special Education Policy Evaluation Dashboard", 
                    className="dashboard-title",
                    style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#2c3e50'}),
            html.H3("Interactive Analysis of Funding Formula Reforms and Student Outcomes",
                    style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#7f8c8d'}),
            html.Hr()
        ])
        
        # Tab structure
        tabs = dcc.Tabs(id="main-tabs", value='overview', children=[
            dcc.Tab(label='📊 Overview', value='overview'),
            dcc.Tab(label='📈 Descriptive Analysis', value='descriptive'),
            dcc.Tab(label='🎯 Causal Analysis', value='causal'),
            dcc.Tab(label='🔧 Robustness Tests', value='robustness'),
            dcc.Tab(label='🔍 Method Comparison', value='methods'),
            dcc.Tab(label='🩺 Diagnostics', value='diagnostics'),
            dcc.Tab(label='📱 Data Explorer', value='data-explorer')
        ])
        
        # Main content area
        content = html.Div(id='tab-content', style={'padding': '20px'})
        
        # Footer
        footer = html.Div([
            html.Hr(),
            html.P([
                "Generated by Jeff Chen (jeffreyc1@alumni.cmu.edu) with ",
                html.A("Claude Code", href="https://claude.ai/code", target="_blank"),
                f" | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            ], style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '12px'})
        ])
        
        # Complete layout
        self.app.layout = html.Div([
            header,
            tabs,
            content,
            footer
        ], style={'fontFamily': 'Arial, sans-serif'})
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks for interactivity."""
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'value')]
        )
        def render_tab_content(active_tab):
            """Render content based on active tab."""
            if active_tab == 'overview':
                return self._create_overview_tab()
            elif active_tab == 'descriptive':
                return self._create_descriptive_tab()
            elif active_tab == 'causal':
                return self._create_causal_tab()
            elif active_tab == 'robustness':
                return self._create_robustness_tab()
            elif active_tab == 'methods':
                return self._create_methods_tab()
            elif active_tab == 'diagnostics':
                return self._create_diagnostics_tab()
            elif active_tab == 'data-explorer':
                return self._create_data_explorer_tab()
            else:
                return html.Div("Content not available")
    
    def _create_overview_tab(self):
        """Create overview tab with key metrics and summary."""
        
        # Key metrics cards
        total_observations = len(self.data)
        n_states = self.data['state'].nunique() if 'state' in self.data.columns else 0
        year_range = f"{self.data['year'].min()}-{self.data['year'].max()}" if 'year' in self.data.columns else "Unknown"
        
        treated_states = 0
        if 'post_treatment' in self.data.columns:
            treated_states = self.data[self.data['post_treatment'] == 1]['state'].nunique()
        
        metrics_cards = html.Div([
            html.Div([
                html.H3(f"{total_observations:,}", style={'margin': '0', 'color': '#3498db'}),
                html.P("Total Observations", style={'margin': '0'})
            ], className='metric-card', style={
                'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '8px',
                'textAlign': 'center', 'margin': '10px', 'flex': '1'
            }),
            
            html.Div([
                html.H3(f"{n_states}", style={'margin': '0', 'color': '#e74c3c'}),
                html.P("States + DC", style={'margin': '0'})
            ], className='metric-card', style={
                'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '8px',
                'textAlign': 'center', 'margin': '10px', 'flex': '1'
            }),
            
            html.Div([
                html.H3(year_range, style={'margin': '0', 'color': '#f39c12'}),
                html.P("Year Range", style={'margin': '0'})
            ], className='metric-card', style={
                'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '8px',
                'textAlign': 'center', 'margin': '10px', 'flex': '1'
            }),
            
            html.Div([
                html.H3(f"{treated_states}", style={'margin': '0', 'color': '#27ae60'}),
                html.P("Treated States", style={'margin': '0'})
            ], className='metric-card', style={
                'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '8px',
                'textAlign': 'center', 'margin': '10px', 'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'})
        
        # Research overview
        research_overview = html.Div([
            html.H3("🎓 Research Overview", style={'color': '#2c3e50'}),
            html.P([
                "This dashboard presents results from a comprehensive analysis of state-level special education policy reforms. ",
                "Using quasi-experimental methods, we evaluate the causal impact of funding formula changes on student outcomes ",
                "for students with disabilities (SWD) across 51 states from 2009-2023."
            ]),
            
            html.H4("📊 Key Data Sources:", style={'color': '#34495e', 'marginTop': '20px'}),
            html.Ul([
                html.Li("NAEP State Assessments (2017, 2019, 2022) - Achievement outcomes"),
                html.Li("EdFacts/IDEA Reports (2009-2023) - Inclusion and graduation rates"),
                html.Li("Census F-33 Finance Data - Per-pupil spending by source"),
                html.Li("Hand-collected Policy Database - Funding formula reforms and court orders")
            ]),
            
            html.H4("🔬 Analytical Methods:", style={'color': '#34495e', 'marginTop': '20px'}),
            html.Ul([
                html.Li("Staggered Difference-in-Differences with robust standard errors"),
                html.Li("Event Study Analysis for dynamic treatment effects"),
                html.Li("Instrumental Variables using court-ordered funding increases"),
                html.Li("Wild Cluster Bootstrap for inference with moderate cluster counts"),
                html.Li("Comprehensive robustness testing across 12+ specifications")
            ])
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'marginTop': '20px'})
        
        # Navigation help
        navigation_help = html.Div([
            html.H3("🧭 Dashboard Navigation", style={'color': '#2c3e50'}),
            html.Div([
                html.Div([
                    html.H4("📈 Descriptive Analysis", style={'color': '#3498db'}),
                    html.P("Summary statistics, trend plots, and balance tables by treatment status")
                ], style={'flex': '1', 'margin': '10px'}),
                
                html.Div([
                    html.H4("🎯 Causal Analysis", style={'color': '#e74c3c'}),
                    html.P("Main results from TWFE, event studies, and instrumental variables")
                ], style={'flex': '1', 'margin': '10px'}),
                
                html.Div([
                    html.H4("🔧 Robustness Tests", style={'color': '#f39c12'}),
                    html.P("Alternative methods, sensitivity analysis, and specification curves")
                ], style={'flex': '1', 'margin': '10px'})
            ], style={'display': 'flex', 'flexWrap': 'wrap'}),
            
            html.Div([
                html.Div([
                    html.H4("🔍 Method Comparison", style={'color': '#9b59b6'}),
                    html.P("Cross-method result comparison and automated recommendations")
                ], style={'flex': '1', 'margin': '10px'}),
                
                html.Div([
                    html.H4("🩺 Diagnostics", style={'color': '#1abc9c'}),
                    html.P("Data quality checks, assumption testing, and validation reports")
                ], style={'flex': '1', 'margin': '10px'}),
                
                html.Div([
                    html.H4("📱 Data Explorer", style={'color': '#34495e'}),
                    html.P("Interactive data exploration with filtering and visualization tools")
                ], style={'flex': '1', 'margin': '10px'})
            ], style={'display': 'flex', 'flexWrap': 'wrap'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'marginTop': '20px'})
        
        return html.Div([
            metrics_cards,
            research_overview,
            navigation_help
        ])
    
    def _create_descriptive_tab(self):
        """Create descriptive analysis tab."""
        
        # Treatment timeline figure
        timeline_fig = self._create_treatment_timeline()
        
        # Balance table
        balance_table = self._create_balance_table()
        
        # Trend plots
        trend_plots = self._create_trend_plots()
        
        return html.Div([
            html.H2("📈 Descriptive Analysis", style={'color': '#2c3e50'}),
            
            html.Div([
                html.H3("Treatment Timeline"),
                dcc.Graph(figure=timeline_fig)
            ], style={'marginBottom': '30px'}),
            
            html.Div([
                html.H3("Balance Table"),
                balance_table
            ], style={'marginBottom': '30px'}),
            
            html.Div([
                html.H3("Outcome Trends"),
                trend_plots
            ])
        ])
    
    def _create_causal_tab(self):
        """Create causal analysis tab."""
        return html.Div([
            html.H2("🎯 Causal Analysis", style={'color': '#2c3e50'}),
            html.P("Causal analysis results will be displayed here."),
            html.P("This tab will show TWFE results, event studies, and IV estimates.")
        ])
    
    def _create_robustness_tab(self):
        """Create robustness analysis tab."""
        return html.Div([
            html.H2("🔧 Robustness Tests", style={'color': '#2c3e50'}),
            html.P("Robustness analysis results will be displayed here."),
            html.P("This tab will show alternative methods, sensitivity analysis, and specification curves.")
        ])
    
    def _create_methods_tab(self):
        """Create method comparison tab."""
        
        # Method recommendation summary
        method_rec_summary = self._create_method_recommendations_summary()
        
        return html.Div([
            html.H2("🔍 Method Comparison", style={'color': '#2c3e50'}),
            method_rec_summary,
            html.P("Detailed method comparison results will be displayed here.")
        ])
    
    def _create_diagnostics_tab(self):
        """Create diagnostics tab."""
        
        # Diagnostic summary cards
        diagnostic_cards = self._create_diagnostic_cards()
        
        return html.Div([
            html.H2("🩺 Diagnostics & Validation", style={'color': '#2c3e50'}),
            diagnostic_cards,
            html.P("Detailed diagnostic results will be displayed here.")
        ])
    
    def _create_data_explorer_tab(self):
        """Create data explorer tab."""
        
        # Control panel
        controls = html.Div([
            html.H3("🎛️ Controls"),
            
            html.Div([
                html.Label("Select States:"),
                dcc.Dropdown(
                    id='state-selector',
                    options=[{'label': state, 'value': state} for state in sorted(self.data['state'].unique())],
                    value=sorted(self.data['state'].unique())[:5],  # First 5 states
                    multi=True
                )
            ], style={'marginBottom': '20px'}),
            
            html.Div([
                html.Label("Select Years:"),
                dcc.RangeSlider(
                    id='year-slider',
                    min=self.data['year'].min(),
                    max=self.data['year'].max(),
                    step=1,
                    value=[self.data['year'].min(), self.data['year'].max()],
                    marks={year: str(year) for year in range(self.data['year'].min(), self.data['year'].max()+1, 2)}
                )
            ], style={'marginBottom': '20px'}),
            
            html.Div([
                html.Label("Select Variable:"),
                dcc.Dropdown(
                    id='variable-selector',
                    options=[
                        {'label': 'Math Grade 4 SWD Score', 'value': 'math_grade4_swd_score'},
                        {'label': 'Math Grade 4 Gap', 'value': 'math_grade4_gap'},
                        {'label': 'Total Revenue per Pupil', 'value': 'total_revenue_per_pupil'},
                        {'label': 'Federal Revenue per Pupil', 'value': 'federal_revenue_per_pupil'}
                    ],
                    value='math_grade4_swd_score'
                )
            ])
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'})
        
        # Visualization area
        visualization = html.Div([
            dcc.Graph(id='explorer-graph')
        ], style={'width': '70%', 'display': 'inline-block'})
        
        return html.Div([
            html.H2("📱 Data Explorer", style={'color': '#2c3e50'}),
            html.Div([controls, visualization])
        ])
    
    def _create_treatment_timeline(self):
        """Create treatment adoption timeline figure."""
        if 'post_treatment' not in self.data.columns:
            return go.Figure().add_annotation(text="Treatment data not available", x=0.5, y=0.5)
        
        # Calculate treatment adoption by year
        treatment_by_year = self.data.groupby(['year', 'state'])['post_treatment'].first().reset_index()
        treatment_adoption = treatment_by_year.groupby('year')['post_treatment'].sum().reset_index()
        
        fig = px.bar(
            treatment_adoption, 
            x='year', 
            y='post_treatment',
            title='Number of States with Active Funding Formula Reforms by Year',
            labels={'post_treatment': 'Number of Treated States', 'year': 'Year'}
        )
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Treated States",
            showlegend=False
        )
        
        return fig
    
    def _create_balance_table(self):
        """Create balance table comparing treated and control groups."""
        if 'post_treatment' not in self.data.columns:
            return html.P("Treatment data not available for balance table")
        
        # Select key variables for balance table
        balance_vars = ['total_revenue_per_pupil', 'federal_revenue_per_pupil', 'state_revenue_per_pupil']
        balance_vars = [var for var in balance_vars if var in self.data.columns]
        
        if not balance_vars:
            return html.P("No suitable variables available for balance table")
        
        # Calculate means by treatment status
        balance_data = []
        for var in balance_vars:
            treated_mean = self.data[self.data['post_treatment'] == 1][var].mean()
            control_mean = self.data[self.data['post_treatment'] == 0][var].mean()
            difference = treated_mean - control_mean
            
            balance_data.append({
                'Variable': var.replace('_', ' ').title(),
                'Treated Mean': f"{treated_mean:.2f}" if not pd.isna(treated_mean) else "N/A",
                'Control Mean': f"{control_mean:.2f}" if not pd.isna(control_mean) else "N/A", 
                'Difference': f"{difference:.2f}" if not pd.isna(difference) else "N/A"
            })
        
        # Create table
        balance_df = pd.DataFrame(balance_data)
        
        return html.Table([
            html.Thead([
                html.Tr([html.Th(col) for col in balance_df.columns])
            ]),
            html.Tbody([
                html.Tr([html.Td(balance_df.iloc[i][col]) for col in balance_df.columns])
                for i in range(len(balance_df))
            ])
        ], style={'width': '100%', 'textAlign': 'center'})
    
    def _create_trend_plots(self):
        """Create trend plots for key outcomes."""
        # Select variables that exist in the data
        outcome_vars = ['math_grade4_swd_score', 'math_grade4_gap', 'total_revenue_per_pupil']
        available_vars = [var for var in outcome_vars if var in self.data.columns]
        
        if not available_vars:
            return html.P("No outcome variables available for trend plots")
        
        # Create subplot figure
        fig = make_subplots(
            rows=len(available_vars), cols=1,
            subplot_titles=[var.replace('_', ' ').title() for var in available_vars],
            vertical_spacing=0.15
        )
        
        for i, var in enumerate(available_vars, 1):
            # Calculate yearly means by treatment status
            if 'post_treatment' in self.data.columns:
                yearly_data = self.data.groupby(['year', 'post_treatment'])[var].mean().reset_index()
                
                # Treated group
                treated_data = yearly_data[yearly_data['post_treatment'] == 1]
                fig.add_trace(
                    go.Scatter(
                        x=treated_data['year'],
                        y=treated_data[var],
                        mode='lines+markers',
                        name=f'Treated' if i == 1 else None,
                        showlegend=i == 1,
                        line=dict(color='red'),
                        legendgroup='treated'
                    ),
                    row=i, col=1
                )
                
                # Control group
                control_data = yearly_data[yearly_data['post_treatment'] == 0]
                fig.add_trace(
                    go.Scatter(
                        x=control_data['year'],
                        y=control_data[var],
                        mode='lines+markers',
                        name=f'Control' if i == 1 else None,
                        showlegend=i == 1,
                        line=dict(color='blue'),
                        legendgroup='control'
                    ),
                    row=i, col=1
                )
            else:
                # Overall trend if no treatment variable
                yearly_data = self.data.groupby('year')[var].mean().reset_index()
                fig.add_trace(
                    go.Scatter(
                        x=yearly_data['year'],
                        y=yearly_data[var],
                        mode='lines+markers',
                        name='Overall Trend' if i == 1 else None,
                        showlegend=i == 1
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(height=200 * len(available_vars), title_text="Outcome Trends Over Time")
        return dcc.Graph(figure=fig)
    
    def _create_method_recommendations_summary(self):
        """Create method recommendations summary."""
        if not self.method_recommendations:
            return html.P("Method recommendations not available")
        
        # Extract key information
        try:
            primary_method = self.method_recommendations.get('primary_recommendation', {})
            method_name = primary_method.get('method_name', 'Unknown')
            suitability_score = primary_method.get('suitability_score', 0)
            
            return html.Div([
                html.H3("🎯 Recommended Primary Method", style={'color': '#27ae60'}),
                html.Div([
                    html.H4(method_name, style={'margin': '0', 'color': '#2c3e50'}),
                    html.P(f"Suitability Score: {suitability_score:.1f}/100", style={'margin': '5px 0'}),
                    html.P(primary_method.get('justification', 'No justification available'))
                ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '8px'})
            ])
        except Exception as e:
            return html.P(f"Error displaying method recommendations: {e}")
    
    def _create_diagnostic_cards(self):
        """Create diagnostic summary cards."""
        if not self.diagnostic_results:
            return html.P("Diagnostic results not available")
        
        try:
            # Extract key diagnostic information
            data_quality = self.diagnostic_results.get('data_adequacy_assessment', {}).get('overall_score', 0)
            missing_rate = self.diagnostic_results.get('data_adequacy_assessment', {}).get('missing_data_rate', 0)
            
            cards = html.Div([
                html.Div([
                    html.H3(f"{data_quality:.1f}/100", style={'margin': '0', 'color': '#3498db'}),
                    html.P("Data Quality Score", style={'margin': '0'})
                ], style={
                    'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '8px',
                    'textAlign': 'center', 'margin': '10px', 'flex': '1'
                }),
                
                html.Div([
                    html.H3(f"{missing_rate:.1%}", style={'margin': '0', 'color': '#e74c3c'}),
                    html.P("Missing Data Rate", style={'margin': '0'})
                ], style={
                    'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '8px',
                    'textAlign': 'center', 'margin': '10px', 'flex': '1'
                })
            ], style={'display': 'flex', 'flexWrap': 'wrap'})
            
            return cards
            
        except Exception as e:
            return html.P(f"Error displaying diagnostic results: {e}")
    
    def run_dashboard(self, host='127.0.0.1', port=8050, debug=True):
        """
        Run the interactive dashboard.
        
        Args:
            host: Host address
            port: Port number
            debug: Enable debug mode
        """
        print(f"🚀 Starting Interactive Dashboard...")
        print(f"📊 Dashboard URL: http://{host}:{port}")
        print(f"📈 Total observations: {len(self.data):,}")
        print(f"🗓️ Year range: {self.data['year'].min()}-{self.data['year'].max()}")
        print(f"🏛️ States: {self.data['state'].nunique()}")
        
        # Add data explorer callback
        @self.app.callback(
            Output('explorer-graph', 'figure'),
            [Input('state-selector', 'value'),
             Input('year-slider', 'value'), 
             Input('variable-selector', 'value')]
        )
        def update_explorer_graph(selected_states, year_range, selected_variable):
            """Update data explorer graph based on selections."""
            if not selected_states or not selected_variable:
                return go.Figure().add_annotation(text="Please select states and variable", x=0.5, y=0.5)
                
            # Filter data
            filtered_data = self.data[
                (self.data['state'].isin(selected_states)) &
                (self.data['year'] >= year_range[0]) &
                (self.data['year'] <= year_range[1])
            ]
            
            if filtered_data.empty or selected_variable not in filtered_data.columns:
                return go.Figure().add_annotation(text="No data available for selection", x=0.5, y=0.5)
            
            # Create line plot
            fig = px.line(
                filtered_data,
                x='year',
                y=selected_variable,
                color='state',
                title=f'{selected_variable.replace("_", " ").title()} by State and Year'
            )
            
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title=selected_variable.replace('_', ' ').title(),
                hovermode='x unified'
            )
            
            return fig
        
        self.app.run_server(host=host, port=port, debug=debug)


def create_interactive_dashboard(data_path: str = "data/final/analysis_panel.csv") -> InteractiveDashboard:
    """
    Create and return interactive dashboard instance.
    
    Args:
        data_path: Path to analysis dataset
        
    Returns:
        InteractiveDashboard instance
    """
    dashboard = InteractiveDashboard(data_path)
    return dashboard


if __name__ == "__main__":
    # Create and run dashboard
    dashboard = create_interactive_dashboard()
    dashboard.run_dashboard()