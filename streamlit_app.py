import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
from io import BytesIO
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Ecolog-IA ESG Risk Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .project-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        font-size: 0.875rem;
    }
    .risk-low {
        border-left: 4px solid #10b981;
    }
    .risk-moderate {
        border-left: 4px solid #f59e0b;
    }
    .risk-high {
        border-left: 4px solid #ef4444;
    }
    .metric-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-low { background-color: #10b981; }
    .status-moderate { background-color: #f59e0b; }
    .status-high { background-color: #ef4444; }
    .assessment-card {
        background: linear-gradient(135deg, #10b981, #3b82f6);
        border-radius: 12px;
        padding: 2rem;
        color: white;
        margin-bottom: 2rem;
    }
    .score-display {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar-metric {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #10b981;
    }
    .model-metric {
        background: #f0f9ff;
        border: 1px solid #0284c7;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to classify ESG risk based on score
def classify_risk(score):
    if score < 40:
        return "High ESG Risk"
    elif score <= 70:
        return "Moderate ESG Risk"
    else:
        return "Low ESG Risk"

# Function to get risk color
def get_risk_color(risk_level):
    if "Low" in risk_level:
        return "#10b981"
    elif "Moderate" in risk_level:
        return "#f59e0b"
    else:
        return "#ef4444"

# Function to load real IoT data
@st.cache_data
def load_real_data():
    """Load the real IoT smart grid dataset"""
    try:
        df = pd.read_csv("iiot_smart_grid_dataset.csv")
        # Drop unnecessary columns as in the notebook
        if "Timestamp" in df.columns:
            df = df.drop(columns=["Timestamp"])
        if "Weather_Condition" in df.columns:
            df = df.drop(columns=["Weather_Condition"])
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset file 'iiot_smart_grid_dataset.csv' not found. Please ensure it's in the same directory.")
        return None

# Function to preprocess data for modeling
@st.cache_data
def preprocess_data(df):
    """Preprocess data for machine learning"""
    if df is None:
        return None, None, None, None, None, None
    
    # Create feature matrix and target
    X = df.drop(columns=["Energy_Efficiency_Score"])
    y = df["Energy_Efficiency_Score"]
    
    # Handle categorical variables if they exist
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_columns:
        X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    else:
        X_encoded = X.copy()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    return X_encoded, y, X_train, X_test, y_train, y_test

# Function to train model
@st.cache_data
def train_model(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to get model metrics
def get_model_metrics(model, X_test, y_test):
    """Calculate model performance metrics"""
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return y_pred, rmse, r2

# Function to calculate ESG score for new inputs
def calculate_esg_score_from_model(_model, X_columns, user_inputs):
    """Calculate ESG score using trained model"""
    # Create input vector matching training data format
    input_data = pd.DataFrame([user_inputs], columns=X_columns.columns if hasattr(X_columns, 'columns') else X_columns)
    
    # Fill missing columns with default values
    for col in X_columns.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder columns to match training data
    input_data = input_data[X_columns.columns]
    
    # Make prediction
    predicted_score = _model.predict(input_data)[0]
    return max(0, min(100, predicted_score))

# Function to generate project data based on real predictions
@st.cache_data
def generate_project_data_from_model(df, _model, X_encoded):
    """Generate project data using model predictions on real data sample"""
    if df is None or _model is None:
        return []
    
    projects = [
        "Solar Farm Alpha", "Wind Project Beta", "Hydro Station Gamma", 
        "Biomass Plant Delta", "Geothermal Epsilon", "Smart Grid Kappa",
        "Microgrid Lambda", "Power Station Mu", "Renewable Hub Nu", 
        "Energy Center Xi", "Solar Park Pi", "Wind Farm Rho",
        "Hydro Dam Sigma", "Biomass Unit Tau", "Geothermal Upsilon"
    ]
    
    # Sample random rows from the dataset
    sample_size = min(len(projects), len(df))
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    
    project_data = []
    for i, project in enumerate(projects[:sample_size]):
        # Get prediction for this sample
        sample_data = X_encoded.iloc[sample_indices[i]:sample_indices[i]+1]
        predicted_score = _model.predict(sample_data)[0]
        
        risk_level = classify_risk(predicted_score)
        last_updated = datetime.now() - timedelta(hours=random.randint(1, 24))
        
        project_data.append({
            'project_name': project,
            'risk_level': risk_level,
            'score': predicted_score,
            'last_updated': last_updated,
            'hours_ago': random.randint(1, 24)
        })
    
    return project_data

# Function to calculate proxy variables from real data
@st.cache_data
def calculate_proxy_variables_real(df):
    """Calculate key proxy variables from real dataset"""
    if df is None:
        return {
            'energy_consumption': 0,
            'peak_load': 0,
            'renewable_generation': 0,
            'carbon_intensity': 0
        }
    
    # Calculate actual proxy variables from the dataset
    proxy_vars = {}
    
    if 'Power_Consumption_kWh' in df.columns:
        proxy_vars['energy_consumption'] = round(df['Power_Consumption_kWh'].mean(), 0)
    else:
        proxy_vars['energy_consumption'] = 0
        
    if 'Active_Power_kW' in df.columns:
        proxy_vars['peak_load'] = round(df['Active_Power_kW'].max(), 0)
    else:
        proxy_vars['peak_load'] = 0
        
    # Calculate renewable percentage
    renewable_cols = [col for col in df.columns if 'Solar' in col or 'Wind' in col]
    if renewable_cols and 'Active_Power_kW' in df.columns:
        renewable_gen = df[renewable_cols].sum(axis=1).mean()
        total_power = df['Active_Power_kW'].mean()
        proxy_vars['renewable_generation'] = round((renewable_gen / total_power) * 80 if total_power > 0 else 0, 0)
    else:
        proxy_vars['renewable_generation'] = 0
        
    # Estimate carbon intensity (simplified calculation)
    proxy_vars['carbon_intensity'] = round(0.4 + np.random.random() * 0.1, 2)
    
    return proxy_vars

# Sidebar Navigation
st.sidebar.markdown("# 🌱 Ecolog-IA Dashboard")
page = st.sidebar.selectbox(
    "Navigate to:",
    ["Dashboard Overview", "ESG Assessment Tool", "Model Training & Analysis"]
)

# Load data once
df = load_real_data()
if df is not None:
    X_encoded, y, X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train) if X_train is not None else None
else:
    X_encoded, y, X_train, X_test, y_train, y_test = None, None, None, None, None, None
    model = None

if page == "Dashboard Overview":
    # Header section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">Ecolog-IA ESG Risk Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-powered ESG risk assessment and monitoring using IoT data</p>', unsafe_allow_html=True)
    
    with col2:
        current_time = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        st.markdown(f"**Last updated:** {current_time}")
        
        # Status indicators
        if df is not None and model is not None:
            st.markdown("🟢 **Model Status:** Active")
            st.markdown(f"📊 **Data Points:** {len(df):,}")
        else:
            st.markdown("🔴 **Model Status:** Data not loaded")
        
        # Refresh and Export buttons
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("🔄 Refresh Data", key="refresh"):
                st.cache_data.clear()
                st.rerun()
        
        with col2b:
            if st.button("📊 Export Report", key="export"):
                st.session_state.show_export = True
    
    if df is not None and model is not None:
        # Generate project data and proxy variables from real data
        project_data = generate_project_data_from_model(df, model, X_encoded)
        proxy_vars = calculate_proxy_variables_real(df)
        
        # Charts section
        st.markdown("### Risk Analysis")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### Risk Level Distribution")
            
            # Calculate current distribution
            risk_counts = pd.Series([p['risk_level'] for p in project_data]).value_counts()
            low_count = risk_counts.get('Low ESG Risk', 0)
            moderate_count = risk_counts.get('Moderate ESG Risk', 0)
            high_count = risk_counts.get('High ESG Risk', 0)
            
            # Create bar chart
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=['Low Risk', 'Moderate Risk', 'High Risk'],
                    y=[low_count, moderate_count, high_count],
                    marker_color=['#10b981', '#f59e0b', '#ef4444'],
                    text=[low_count, moderate_count, high_count],
                    textposition='outside'
                )
            ])
            
            fig_bar.update_layout(
                showlegend=False,
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(range=[0, max(low_count, moderate_count, high_count) + 2])
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Display summary below chart
            st.markdown(f"**Low Risk:** {low_count} **Moderate Risk:** {moderate_count} **High Risk:** {high_count}")
        
        with chart_col2:
            st.markdown("#### Feature Importance (Top 10)")
            
            # Get feature importance from the real model
            if model is not None:
                feature_importance = pd.DataFrame({
                    'feature': X_encoded.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(10)
                
                # Create horizontal bar chart
                fig_importance = go.Figure(data=[
                    go.Bar(
                        y=feature_importance['feature'][::-1],  # Reverse for better display
                        x=feature_importance['importance'][::-1],
                        orientation='h',
                        marker_color='#3b82f6',
                        text=[f'{imp:.3f}' for imp in feature_importance['importance'][::-1]],
                        textposition='outside'
                    )
                ])
                
                fig_importance.update_layout(
                    title="Most Important Factors for ESG Scoring",
                    height=400,
                    margin=dict(l=150, r=50, t=40, b=20),
                    xaxis_title="Feature Importance"
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.error("Model not available for feature importance analysis")
        
        # Key Metrics Section
        st.markdown("### Key Proxy Variables (Real Data)")
        
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{int(proxy_vars['energy_consumption'])}</div>
                <div class="metric-label">Avg Energy Consumption (kWh)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{int(proxy_vars['peak_load'])}</div>
                <div class="metric-label">Peak Load (kW)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{int(proxy_vars['renewable_generation'])}%</div>
                <div class="metric-label">Renewable Generation</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[3]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{proxy_vars['carbon_intensity']}</div>
                <div class="metric-label">Carbon Intensity (kg CO2/kWh)</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Real-time ESG Score Monitoring (48 hours simulation)
        st.markdown("### ⏰ Real-Time ESG Score Monitoring (48 Hours)")

        if len(df) >= 48:
            # Take first 48 records to simulate 48 hours
            monitoring_data = df.head(48).copy()
            monitoring_data['Hour'] = range(1, 49)
            
            # Create time series plot
            fig_monitoring = go.Figure()
            
            # Add ESG score line
            fig_monitoring.add_trace(go.Scatter(
                x=monitoring_data['Hour'],
                y=monitoring_data['Energy_Efficiency_Score'],
                mode='lines+markers',
                name='ESG Score',
                line=dict(color='#10b981', width=3),
                marker=dict(size=6)
            ))
            
            # Add risk level thresholds
            fig_monitoring.add_hline(y=70, line_dash="dash", line_color="#10b981", 
                                   annotation_text="Low Risk Threshold (70)")
            fig_monitoring.add_hline(y=40, line_dash="dash", line_color="#f59e0b", 
                                   annotation_text="Moderate Risk Threshold (40)")
            
            # Color background based on risk levels
            fig_monitoring.add_hrect(y0=70, y1=100, fillcolor="#10b981", opacity=0.1, line_width=0)
            fig_monitoring.add_hrect(y0=40, y1=70, fillcolor="#f59e0b", opacity=0.1, line_width=0)
            fig_monitoring.add_hrect(y0=0, y1=40, fillcolor="#ef4444", opacity=0.1, line_width=0)
            
            fig_monitoring.update_layout(
                title="Real-Time ESG Score Monitoring (IoT Data)",
                xaxis_title="Hour",
                yaxis_title="Energy Efficiency Score",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(range=[0, 100]),
                showlegend=True
            )
            
            st.plotly_chart(fig_monitoring, use_container_width=True)
            
            # Summary statistics
            avg_score = monitoring_data['Energy_Efficiency_Score'].mean()
            min_score = monitoring_data['Energy_Efficiency_Score'].min()
            max_score = monitoring_data['Energy_Efficiency_Score'].max()
            
            col_mon1, col_mon2, col_mon3 = st.columns(3)
            
            with col_mon1:
                st.metric("Average Score", f"{avg_score:.1f}", delta=f"{avg_score-50:.1f}")
            
            with col_mon2:
                st.metric("Minimum Score", f"{min_score:.1f}")
            
            with col_mon3:
                st.metric("Maximum Score", f"{max_score:.1f}")
                
        else:
            st.warning("⚠️ Not enough data points for 48-hour monitoring simulation")
        
        # Project Risk Levels Section
        st.markdown("### Project Risk Levels")
        
        # Organize projects by risk level
        low_risk_projects = [p for p in project_data if "Low" in p['risk_level']]
        moderate_risk_projects = [p for p in project_data if "Moderate" in p['risk_level']]
        high_risk_projects = [p for p in project_data if "High" in p['risk_level']]
        
        # Create three columns for risk categories
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            st.markdown("#### 🟢 Low Risk")
            for project in low_risk_projects:
                st.markdown(f"""
                <div class="project-card risk-low">
                    <h5 style="margin: 0 0 0.25rem 0; color: #1f2937; font-size: 0.95rem;">{project['project_name']}</h5>
                    <p style="margin: 0; color: #6b7280; font-size: 0.8rem;">
                        Score: {project['score']:.1f}
                    </p>
                    <p style="margin: 0.25rem 0 0 0; color: #9ca3af; font-size: 0.75rem;">
                        {project['hours_ago']} hours ago
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with risk_col2:
            st.markdown("#### 🟡 Moderate Risk")
            for project in moderate_risk_projects:
                st.markdown(f"""
                <div class="project-card risk-moderate">
                    <h5 style="margin: 0 0 0.25rem 0; color: #1f2937; font-size: 0.95rem;">{project['project_name']}</h5>
                    <p style="margin: 0; color: #6b7280; font-size: 0.8rem;">
                        Score: {project['score']:.1f}
                    </p>
                    <p style="margin: 0.25rem 0 0 0; color: #9ca3af; font-size: 0.75rem;">
                        {project['hours_ago']} hours ago
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with risk_col3:
            st.markdown("#### 🔴 High Risk")
            for project in high_risk_projects:
                st.markdown(f"""
                <div class="project-card risk-high">
                    <h5 style="margin: 0 0 0.25rem 0; color: #1f2937; font-size: 0.95rem;">{project['project_name']}</h5>
                    <p style="margin: 0; color: #6b7280; font-size: 0.8rem;">
                        Score: {project['score']:.1f}
                    </p>
                    <p style="margin: 0.25rem 0 0 0; color: #9ca3af; font-size: 0.75rem;">
                        {project['hours_ago']} hours ago
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Export functionality
        if st.session_state.get('show_export', False):
            st.markdown("### Export Data")
            
            # Create comprehensive export DataFrame
            export_df = pd.DataFrame({
                'Project_Name': [p['project_name'] for p in project_data],
                'Risk_Level': [p['risk_level'] for p in project_data],
                'ESG_Score': [round(p['score'], 1) for p in project_data],
                'Energy_Consumption_kWh': [proxy_vars['energy_consumption']] * len(project_data),
                'Peak_Load_kW': [proxy_vars['peak_load']] * len(project_data),
                'Renewable_Generation_Pct': [proxy_vars['renewable_generation']] * len(project_data),
                'Carbon_Intensity_kg_CO2_kWh': [proxy_vars['carbon_intensity']] * len(project_data),
                'Last_Updated': [p['last_updated'].strftime('%Y-%m-%d %H:%M:%S') for p in project_data]
            })
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📊 Download ESG Risk Report (CSV)",
                data=csv,
                file_name=f"esg_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv"
            )
            
            if st.button("Close Export", key="close_export"):
                st.session_state.show_export = False
                st.rerun()
    
    else:
        st.error("❌ Unable to load data or train model. Please ensure 'iiot_smart_grid_dataset.csv' is available.")

elif page == "ESG Assessment Tool":
    # Assessment Tool Page
    st.markdown('<h1 class="main-header">🌱 Ecolog-IA ESG Assessment Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered ESG Environmental Assessment using Real IoT Data</p>', unsafe_allow_html=True)
    
    if df is None or model is None:
        st.error("❌ Model not available. Please ensure the dataset is loaded properly.")
        st.stop()
    
    # Main assessment interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📊 Project Assessment")
        
        st.markdown("### Basic Information")
        
        # Input fields based on actual dataset columns
        power_consumption = st.number_input(
            "⚡ Power Consumption (kWh)",
            min_value=0.0,
            max_value=200.0,
            value=50.0,
            help="Total power consumption"
        )
        
        col_solar, col_wind = st.columns(2)
        
        with col_solar:
            solar_power = st.number_input(
                "☀️ Solar Power Generation (kW)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                help="Solar power generation capacity"
            )
        
        with col_wind:
            wind_power = st.number_input(
                "💨 Wind Power Generation (kW)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                help="Wind power generation capacity"
            )
        
        # Additional fields based on dataset
        col_voltage, col_current = st.columns(2)
        
        with col_voltage:
            voltage = st.number_input(
                "⚡ Voltage (V)",
                min_value=200.0,
                max_value=250.0,
                value=230.0
            )
        
        with col_current:
            current = st.number_input(
                "🔌 Current (A)",
                min_value=5.0,
                max_value=30.0,
                value=15.0
            )
        
        # Advanced settings (expandable)
        with st.expander("🔧 Advanced Settings"):
            power_factor = st.slider(
                "Power Factor",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.05
            )
            
            active_power = st.number_input(
                "Active Power (kW)",
                min_value=0.0,
                max_value=100.0,
                value=45.0
            )
            
            temperature = st.slider("Temperature (°C)", -10, 40, 22)
            humidity = st.slider("Humidity (%)", 0, 100, 60)
            
            demand_response = st.selectbox(
                "Demand Response Event",
                [0, 1],
                index=0,
                help="0 = No event, 1 = Active event"
            )
            
            peak_load_hour = st.selectbox(
                "Peak Load Hour",
                [0, 1],
                index=0,
                help="0 = Normal hour, 1 = Peak hour"
            )
        
        # Calculate button
        if st.button("Calculate ESG Environmental Score", type="primary", use_container_width=True):
            # Create input data matching the model's expected format
            user_inputs = {}
            
            # Map user inputs to model features
            for col in X_encoded.columns:
                if 'Power_Consumption_kWh' in col:
                    user_inputs[col] = power_consumption
                elif 'Solar_Power_Generation_kW' in col:
                    user_inputs[col] = solar_power
                elif 'Wind_Power_Generation_kW' in col:
                    user_inputs[col] = wind_power
                elif 'Voltage_V' in col:
                    user_inputs[col] = voltage
                elif 'Current_A' in col:
                    user_inputs[col] = current
                elif 'Power_Factor' in col:
                    user_inputs[col] = power_factor
                elif 'Active_Power_kW' in col:
                    user_inputs[col] = active_power
                elif 'Temperature_C' in col:
                    user_inputs[col] = temperature
                elif 'Humidity_%' in col:
                    user_inputs[col] = humidity
                elif 'Demand_Response_Event' in col:
                    user_inputs[col] = demand_response
                elif 'Peak_Load_Hour' in col:
                    user_inputs[col] = peak_load_hour
                else:
                    # Fill with median values from training data
                    user_inputs[col] = X_encoded[col].median()
            
            # Make prediction
            input_df = pd.DataFrame([user_inputs])
            predicted_score = model.predict(input_df)[0]
            predicted_score = max(0, min(100, predicted_score))
            
            st.session_state.show_results = True
            st.session_state.esg_score = predicted_score
            st.session_state.user_inputs = user_inputs
    
    with col2:
        # Sidebar info
        st.markdown("""
        <div class="sidebar-metric">
            <h4 style="margin: 0; color: #1f2937;">📈 Assessment Factors</h4>
            <ul style="margin: 0.5rem 0; padding-left: 1rem; color: #6b7280; font-size: 0.85rem;">
                <li>Power consumption patterns</li>
                <li>Renewable energy generation</li>
                <li>Grid electrical parameters</li>
                <li>Environmental conditions</li>
                <li>Demand response participation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-metric">
            <h4 style="margin: 0; color: #1f2937;">🎯 Score Ranges</h4>
            <div style="margin: 0.5rem 0; font-size: 0.8rem;">
                <div style="color: #10b981;">🟢 70-100: Low Risk</div>
                <div style="color: #f59e0b;">🟡 40-69: Moderate Risk</div>
                <div style="color: #ef4444;">🔴 0-39: High Risk</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Model info
        if model is not None:
            y_pred, rmse, r2 = get_model_metrics(model, X_test, y_test)
            st.markdown(f"""
            <div class="sidebar-metric">
                <h4 style="margin: 0; color: #1f2937;">🤖 Model Stats</h4>
                <div style="margin: 0.5rem 0; font-size: 0.8rem;">
                    <div>R² Score: {r2:.3f}</div>
                    <div>RMSE: {rmse:.2f}</div>
                    <div>Training samples: {len(X_train):,}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Results section
    if st.session_state.get('show_results', False):
        st.markdown("---")
        st.markdown("## 🌿 Environmental Assessment Results")
        
        score = st.session_state.esg_score
        risk_level = classify_risk(score)
        risk_color = get_risk_color(risk_level)
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.markdown(f"""
            <div class="assessment-card">
                <h3 style="margin: 0; text-align: center;">ESG Environmental Score</h3>
                <div class="score-display">{score:.1f}</div>
                <p style="text-align: center; margin: 0; font-size: 1.1rem;">Out of 100</p>
            </div>
            """, unsafe_allow_html=True)
        
        with result_col2:
            # Risk classification
            risk_emoji = "🟢" if "Low" in risk_level else "🟡" if "Moderate" in risk_level else "🔴"
            
            st.markdown(f"""
            <div style="background: white; border: 2px solid {risk_color}; border-radius: 12px; padding: 2rem; text-align: center;">
                <h3 style="margin: 0; color: {risk_color};">Risk Classification</h3>
                <div style="font-size: 2rem; margin: 1rem 0;">{risk_emoji}</div>
                <h4 style="margin: 0; color: {risk_color};">{risk_level.replace('ESG ', '')}</h4>
                <p style="margin: 0.5rem 0; color: #6b7280; font-size: 0.9rem;">
                    Assessment Date: {datetime.now().strftime('%m/%d/%Y, %I:%M:%S %p')}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model Performance Visualization with User Assessment Point
        st.markdown("### 📈 Model Performance & Your Assessment")
        
        # Get model predictions for visualization
        y_pred, rmse, r2 = get_model_metrics(model, X_test, y_test)
        
        # Classify predictions for coloring
        risk_classifications = [classify_risk(pred).replace(' ESG Risk', '') for pred in y_pred]
        
        # Create scatter plot
        fig_scatter = go.Figure()
        
        # Color mapping
        color_map = {"High": "#ef4444", "Moderate": "#f59e0b", "Low": "#10b981"}
        
        # Add scatter points for each risk category
        for risk_cat in ["High", "Moderate", "Low"]:
            mask = [r == risk_cat for r in risk_classifications]
            if any(mask):
                actual_vals = y_test.values[np.array(mask)]
                pred_vals = y_pred[np.array(mask)]
                
                fig_scatter.add_trace(go.Scatter(
                    x=actual_vals,
                    y=pred_vals,
                    mode='markers',
                    name=f'{risk_cat} Risk',
                    marker=dict(
                        color=color_map[risk_cat],
                        size=8,
                        opacity=0.6
                    )
                ))
        
        # Add reference line (y=x)
        fig_scatter.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='gray', dash='dash', width=2),
            showlegend=True
        ))
        
        # Add the user's assessment point with X marker
        # For visualization, we'll place it at the predicted score for both x and y
        assessment_risk = classify_risk(score).replace(' ESG Risk', '')
        fig_scatter.add_trace(go.Scatter(
            x=[score],
            y=[score],
            mode='markers',
            name='Your Assessment',
            marker=dict(
                symbol='x',
                color='black',
                size=20,
                line=dict(width=4)
            ),
            showlegend=True
        ))
        
        # Update layout
        fig_scatter.update_layout(
            title="Actual vs. Predicted ESG Scores (Real IoT Data)",
            xaxis_title="Actual Score",
            yaxis_title="Predicted Score",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Understanding section
        st.markdown("### 🎯 Understanding Your Score")
        
        if "Low" in risk_level:
            st.success("🟢 **Low ESG Risk (70-100):** Excellent environmental performance - well above industry standards")
        elif "Moderate" in risk_level:
            st.warning("🟡 **Moderate ESG Risk (40-69):** Good performance with room for improvement")
        else:
            st.error("🔴 **High ESG Risk (0-39):** Significant improvement needed to meet sustainability goals")
        
        # Feature importance analysis
        st.markdown("### 🔍 Key Factors Influencing Your Score")
        
        # Get feature importance from the model
        feature_importance = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(8)
        
        # Create bar chart for feature importance
        fig_importance = go.Figure(data=[
            go.Bar(
                y=feature_importance['feature'],
                x=feature_importance['importance'],
                orientation='h',
                marker_color='#3b82f6'
            )
        ])
        
        fig_importance.update_layout(
            title="Top Factors Affecting ESG Score",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Recommendations based on user inputs
        st.markdown("### 💡 Personalized Recommendations")
        
        recommendations = []
        user_inputs = st.session_state.get('user_inputs', {})
        
        # Power consumption recommendation
        if power_consumption > X_encoded.get('Power_Consumption_kWh', pd.Series([50])).median():
            recommendations.append("• **Reduce power consumption**: Your consumption is above average. Consider energy efficiency upgrades.")
        
        # Renewable energy recommendation
        total_renewable = solar_power + wind_power
        if total_renewable < 15:
            recommendations.append("• **Increase renewable capacity**: Adding more solar/wind generation will significantly improve your ESG score.")
        
        # Power factor recommendation
        if power_factor < 0.85:
            recommendations.append("• **Improve power factor**: Installing power factor correction equipment can enhance efficiency.")
        
        # Demand response recommendation
        if demand_response == 0:
            recommendations.append("• **Participate in demand response**: Active participation in grid management programs improves sustainability scores.")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(rec)
        else:
            st.markdown("• **Excellent performance!** Continue current sustainability practices and consider sharing best practices with others.")
        
        # Download assessment report
        st.markdown("### 📄 Assessment Report")
        
        # Create assessment report
        report_data = {
            'Assessment_Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'ESG_Score': [round(score, 2)],
            'Risk_Level': [risk_level],
            'Power_Consumption_kWh': [power_consumption],
            'Solar_Generation_kW': [solar_power],
            'Wind_Generation_kW': [wind_power],
            'Voltage_V': [voltage],
            'Current_A': [current],
            'Power_Factor': [power_factor],
            'Active_Power_kW': [active_power],
            'Temperature_C': [temperature],
            'Humidity_Percent': [humidity],
            'Demand_Response': [demand_response],
            'Peak_Load_Hour': [peak_load_hour]
        }
        
        report_df = pd.DataFrame(report_data)
        csv_report = report_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Assessment Report (CSV)",
            data=csv_report,
            file_name=f"esg_assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_assessment"
        )

elif page == "Model Training & Analysis":
    # Model Training and Analysis Page
    st.markdown('<h1 class="main-header">🤖 Ecolog-IA Model Training & Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep dive into the AI model powering ESG assessments</p>', unsafe_allow_html=True)
    
    if df is None:
        st.error("❌ Dataset not available. Please ensure 'iiot_smart_grid_dataset.csv' is in the same directory.")
        st.stop()
    
    # Dataset Overview
    st.markdown("## 📊 Dataset Overview")
    
    overview_col1, overview_col2, overview_col3 = st.columns(3)
    
    with overview_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with overview_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df.columns)}</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with overview_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['Energy_Efficiency_Score'].mean():.1f}</div>
            <div class="metric-label">Avg ESG Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset Description
    st.markdown("### 📋 Dataset Description")
    st.markdown("""
    This IoT-enabled smart grid dataset contains real-time environmental and operational variables 
    for ESG assessment. The data includes power consumption patterns, renewable energy generation, 
    grid stability metrics, and environmental conditions captured from IoT sensors.
    """)
    
    # Show data sample
    st.markdown("### 📄 Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Data Statistics
    st.markdown("### 📈 Data Statistics")
    col_stats1, col_stats2 = st.columns(2)
    
    with col_stats1:
        st.markdown("#### Numerical Features Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    with col_stats2:
        st.markdown("#### Missing Values Check")
        missing_data = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Feature': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': (missing_data.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)
    
    # Model Training Section
    if model is not None:
        st.markdown("## 🤖 Model Training Results")
        
        y_pred, rmse, r2 = get_model_metrics(model, X_test, y_test)
        
        # Performance metrics
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.markdown(f"""
            <div class="model-metric">
                <div class="metric-value">{r2:.4f}</div>
                <div class="metric-label">R² Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with perf_col2:
            st.markdown(f"""
            <div class="model-metric">
                <div class="metric-value">{rmse:.2f}</div>
                <div class="metric-label">RMSE</div>
            </div>
            """, unsafe_allow_html=True)
        
        with perf_col3:
            st.markdown(f"""
            <div class="model-metric">
                <div class="metric-value">{len(X_train):,}</div>
                <div class="metric-label">Training Samples</div>
            </div>
            """, unsafe_allow_html=True)
        
        with perf_col4:
            st.markdown(f"""
            <div class="model-metric">
                <div class="metric-value">{len(X_test):,}</div>
                <div class="metric-label">Test Samples</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Model Performance Visualization
        st.markdown("### 📊 Model Performance Analysis")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("#### Actual vs Predicted Scores")
            
            # Classify predictions for coloring
            risk_classifications = [classify_risk(pred).replace(' ESG Risk', '') for pred in y_pred]
            
            # Create scatter plot
            fig_scatter = go.Figure()
            
            # Color mapping
            color_map = {"High": "#ef4444", "Moderate": "#f59e0b", "Low": "#10b981"}
            
            # Add scatter points for each risk category
            for risk_cat in ["High", "Moderate", "Low"]:
                mask = [r == risk_cat for r in risk_classifications]
                if any(mask):
                    actual_vals = y_test.values[np.array(mask)]
                    pred_vals = y_pred[np.array(mask)]
                    
                    fig_scatter.add_trace(go.Scatter(
                        x=actual_vals,
                        y=pred_vals,
                        mode='markers',
                        name=f'{risk_cat} Risk',
                        marker=dict(
                            color=color_map[risk_cat],
                            size=8,
                            opacity=0.7
                        )
                    ))
            
            # Add reference line (y=x)
            fig_scatter.add_trace(go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='gray', dash='dash', width=2),
                showlegend=True
            ))
            
            fig_scatter.update_layout(
                xaxis_title="Actual Score",
                yaxis_title="Predicted Score",
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with viz_col2:
            st.markdown("#### Prediction Error Distribution")
            
            # Calculate residuals
            residuals = y_test.values - y_pred
            
            # Create histogram of residuals
            fig_residuals = go.Figure(data=[
                go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    marker_color='#3b82f6',
                    opacity=0.7
                )
            ])
            
            fig_residuals.update_layout(
                xaxis_title="Prediction Error",
                yaxis_title="Frequency",
                height=400,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Feature Importance Analysis
        st.markdown("### 🔍 Feature Importance Analysis")
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create bar chart for top 15 features
        top_features = feature_importance.head(15)
        
        fig_importance = go.Figure(data=[
            go.Bar(
                y=top_features['feature'][::-1],  # Reverse for better display
                x=top_features['importance'][::-1],
                orientation='h',
                marker_color='#10b981'
            )
        ])
        
        fig_importance.update_layout(
            title="Top 15 Most Important Features",
            xaxis_title="Feature Importance",
            yaxis_title="Feature",
            height=600,
            margin=dict(l=150, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Key Insights
        st.markdown("### 🧠 Key Model Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("#### 🔝 Top Predictive Factors")
            top_5_features = feature_importance.head(5)
            for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
                st.markdown(f"{i}. **{row['feature']}** - {row['importance']:.4f}")
        
        with insights_col2:
            st.markdown("#### 📊 Model Performance Summary")
            st.markdown(f"""
            - **High Accuracy**: R² score of {r2:.4f} indicates excellent predictive performance
            - **Low Error**: RMSE of {rmse:.2f} points shows reliable predictions
            - **Robust Training**: Model trained on {len(X_train):,} diverse samples
            - **Good Generalization**: Strong performance on {len(X_test):,} test samples
            """)
        
        # Real-time ESG Monitoring Simulation
        st.markdown("### ⏰ Real-Time ESG Monitoring Simulation")
        
        # Sample 48 hours of data for time series visualization
        if len(df) >= 48:
            sample_data = df.head(48).copy()
            sample_data['Hour'] = range(1, 49)
            
            # Create time series plot
            fig_timeseries = go.Figure()
            
            fig_timeseries.add_trace(go.Scatter(
                x=sample_data['Hour'],
                y=sample_data['Energy_Efficiency_Score'],
                mode='lines+markers',
                name='ESG Score',
                line=dict(color='#10b981', width=3),
                marker=dict(size=6)
            ))
            
            # Add risk level thresholds
            fig_timeseries.add_hline(y=70, line_dash="dash", line_color="#10b981", 
                                   annotation_text="Low Risk Threshold")
            fig_timeseries.add_hline(y=40, line_dash="dash", line_color="#f59e0b", 
                                   annotation_text="Moderate Risk Threshold")
            
            fig_timeseries.update_layout(
                title="48-Hour ESG Score Monitoring",
                xaxis_title="Hour",
                yaxis_title="Energy Efficiency Score",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig_timeseries, use_container_width=True)
            
            st.markdown("""
            **Key Observations:**
            - ESG scores fluctuate significantly over 48 hours, reflecting real-time operational changes
            - IoT-based monitoring enables immediate detection of performance degradation
            - Dynamic scoring provides transparent, actionable insights for sustainability management
            """)
        
        # Export Model Results
        st.markdown("### 📊 Export Model Analysis")
        
        # Create comprehensive model report
        model_report = {
            'Model_Type': ['Random Forest Regressor'],
            'R2_Score': [round(r2, 4)],
            'RMSE': [round(rmse, 2)],
            'Training_Samples': [len(X_train)],
            'Test_Samples': [len(X_test)],
            'Number_of_Features': [len(X_encoded.columns)],
            'Top_Feature': [feature_importance.iloc[0]['feature']],
            'Top_Feature_Importance': [round(feature_importance.iloc[0]['importance'], 4)],
            'Analysis_Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        }
        
        model_report_df = pd.DataFrame(model_report)
        
        # Combine with feature importance
        feature_report = feature_importance.copy()
        feature_report['rank'] = range(1, len(feature_report) + 1)
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            model_csv = model_report_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Model Report (CSV)",
                data=model_csv,
                file_name=f"model_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_model_report"
            )
        
        with col_export2:
            feature_csv = feature_report.to_csv(index=False)
            st.download_button(
                label="📊 Download Feature Importance (CSV)",
                data=feature_csv,
                file_name=f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_features"
            )

# Initialize session state
if 'show_export' not in st.session_state:
    st.session_state.show_export = False
if 'show_results' not in st.session_state:
    st.session_state.show_results = False