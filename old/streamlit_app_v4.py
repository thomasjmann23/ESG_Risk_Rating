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
import random

# Page configuration
st.set_page_config(
    page_title="ESG Risk Dashboard",
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

# Function to get risk color class
def get_risk_class(risk_level):
    if "Low" in risk_level:
        return "risk-low"
    elif "Moderate" in risk_level:
        return "risk-moderate"
    else:
        return "risk-high"

# Function to get status indicator class
def get_status_class(risk_level):
    if "Low" in risk_level:
        return "status-low"
    elif "Moderate" in risk_level:
        return "status-moderate"
    else:
        return "status-high"

# Function to get risk color
def get_risk_color(risk_level):
    if "Low" in risk_level:
        return "#10b981"
    elif "Moderate" in risk_level:
        return "#f59e0b"
    else:
        return "#ef4444"

# Function to load or generate sample data
@st.cache_data
def load_sample_data():
    """Load sample IoT smart grid data"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data similar to the original dataset
    data = {
        'Power_Consumption_kWh': np.random.normal(50, 15, n_samples),
        'Voltage_V': np.random.normal(230, 10, n_samples),
        'Current_A': np.random.normal(15, 5, n_samples),
        'Power_Factor': np.random.uniform(0.8, 1.0, n_samples),
        'Grid_Frequency_Hz': np.random.normal(50, 0.5, n_samples),
        'Reactive_Power_kVAR': np.random.normal(10, 3, n_samples),
        'Active_Power_kW': np.random.normal(45, 12, n_samples),
        'Solar_Power_Generation_kW': np.random.exponential(8, n_samples),
        'Wind_Power_Generation_kW': np.random.exponential(6, n_samples),
        'Temperature_C': np.random.normal(22, 8, n_samples),
        'Humidity_%': np.random.uniform(30, 80, n_samples),
        'Demand_Response_Event': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Previous_Day_Consumption_kWh': np.random.normal(48, 14, n_samples),
        'Peak_Load_Hour': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'User_Type': np.random.choice(['Residential', 'Industrial', 'Commercial'], n_samples),
        'Normalized_Consumption': np.random.uniform(0.3, 1.2, n_samples),
        'Energy_Source_Type': np.random.choice(['Grid', 'Solar', 'Wind', 'Hybrid'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate Energy Efficiency Score based on features
    df['Energy_Efficiency_Score'] = (
        100 - (df['Power_Consumption_kWh'] - df['Power_Consumption_kWh'].min()) / 
        (df['Power_Consumption_kWh'].max() - df['Power_Consumption_kWh'].min()) * 40 +
        (df['Solar_Power_Generation_kW'] + df['Wind_Power_Generation_kW']) / 20 * 10 +
        df['Power_Factor'] * 20 + 
        df['Demand_Response_Event'] * 10 +
        np.random.normal(0, 5, n_samples)
    ).clip(0, 100)
    
    return df

# Function to train model and make predictions
@st.cache_data
def train_model_and_predict(df):
    """Train Random Forest model and generate predictions"""
    # Prepare features
    feature_columns = [
        'Power_Consumption_kWh', 'Voltage_V', 'Current_A', 'Power_Factor',
        'Grid_Frequency_Hz', 'Reactive_Power_kVAR', 'Active_Power_kW',
        'Solar_Power_Generation_kW', 'Wind_Power_Generation_kW',
        'Temperature_C', 'Humidity_%', 'Demand_Response_Event',
        'Previous_Day_Consumption_kWh', 'Peak_Load_Hour', 'Normalized_Consumption'
    ]
    
    # Encode categorical variables
    df_encoded = df.copy()
    df_encoded['User_Type_Residential'] = (df['User_Type'] == 'Residential').astype(int)
    df_encoded['User_Type_Industrial'] = (df['User_Type'] == 'Industrial').astype(int)
    df_encoded['User_Type_Commercial'] = (df['User_Type'] == 'Commercial').astype(int)
    
    df_encoded['Energy_Source_Grid'] = (df['Energy_Source_Type'] == 'Grid').astype(int)
    df_encoded['Energy_Source_Solar'] = (df['Energy_Source_Type'] == 'Solar').astype(int)
    df_encoded['Energy_Source_Wind'] = (df['Energy_Source_Type'] == 'Wind').astype(int)
    df_encoded['Energy_Source_Hybrid'] = (df['Energy_Source_Type'] == 'Hybrid').astype(int)
    
    feature_columns.extend([
        'User_Type_Residential', 'User_Type_Industrial', 'User_Type_Commercial',
        'Energy_Source_Grid', 'Energy_Source_Solar', 'Energy_Source_Wind', 'Energy_Source_Hybrid'
    ])
    
    X = df_encoded[feature_columns]
    y = df_encoded['Energy_Efficiency_Score']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions, model

# Function to generate project data
@st.cache_data
def generate_project_data():
    """Generate sample project data for dashboard"""
    projects = [
        "Solar Farm Alpha", "Wind Project Beta", "Hydro Station Gamma", 
        "Biomass Plant Delta", "Geothermal Epsilon", "Nuclear Zeta",
        "Coal Plant Eta", "Gas Station Theta", "Battery Storage Iota",
        "Smart Grid Kappa", "Microgrid Lambda", "Power Station Mu",
        "Renewable Hub Nu", "Energy Center Xi", "Grid Station Omicron",
        "Solar Park Pi", "Wind Farm Rho", "Hydro Dam Sigma",
        "Biomass Unit Tau", "Geothermal Upsilon", "Nuclear Plant Phi",
        "Gas Turbine Chi", "Storage Facility Psi"
    ]
    
    # Generate risk scores and classifications
    np.random.seed(42)
    scores = np.random.beta(2, 2, len(projects)) * 100
    
    project_data = []
    for i, project in enumerate(projects):
        score = scores[i]
        risk_level = classify_risk(score)
        last_updated = datetime.now() - timedelta(hours=random.randint(1, 24))
        
        project_data.append({
            'project_name': project,
            'risk_level': risk_level,
            'score': score,
            'last_updated': last_updated,
            'hours_ago': random.randint(1, 24)
        })
    
    return project_data

# Function to generate monthly trend data
@st.cache_data
def generate_trend_data():
    """Generate monthly trend data for charts"""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    np.random.seed(42)
    
    # Generate realistic trends
    low_risk = [8, 9, 10, 11, 12, 12]
    moderate_risk = [6, 7, 8, 7, 8, 8]
    high_risk = [5, 4, 3, 4, 3, 3]
    
    trend_data = pd.DataFrame({
        'Month': months,
        'Low Risk': low_risk,
        'Moderate Risk': moderate_risk,
        'High Risk': high_risk
    })
    
    return trend_data

# Function to calculate proxy variables
@st.cache_data
def calculate_proxy_variables(df):
    """Calculate key proxy variables from the dataset"""
    return {
        'energy_consumption': round(df['Power_Consumption_kWh'].mean(), 0),
        'peak_load': round(df['Active_Power_kW'].max(), 0),
        'renewable_generation': round((df['Solar_Power_Generation_kW'].mean() + 
                                     df['Wind_Power_Generation_kW'].mean()) / 
                                    df['Active_Power_kW'].mean() * 100, 0),
        'carbon_intensity': round(0.4 + np.random.random() * 0.1, 2)
    }

# Function to calculate ESG score based on inputs
def calculate_esg_score(monthly_energy, solar_power, wind_power, power_factor=0.9, user_type='Industrial'):
    """Calculate ESG score based on input parameters"""
    # Normalize energy consumption (lower is better)
    energy_score = max(0, 100 - (monthly_energy - 50) * 0.5)
    
    # Renewable energy score (higher is better)
    renewable_score = min(100, (solar_power + wind_power) * 2)
    
    # Power factor score
    pf_score = power_factor * 100
    
    # User type modifier
    type_modifier = {'Residential': 1.0, 'Commercial': 0.95, 'Industrial': 0.9}
    modifier = type_modifier.get(user_type, 0.9)
    
    # Calculate weighted average
    final_score = (energy_score * 0.4 + renewable_score * 0.4 + pf_score * 0.2) * modifier
    
    return max(0, min(100, final_score))

# Function to create downloadable CSV
def create_download_link(df, filename):
    """Create a download link for CSV export"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV Report</a>'
    return href

# Sidebar Navigation
st.sidebar.markdown("# 🌱 ESG Dashboard")
page = st.sidebar.selectbox(
    "Navigate to:",
    ["Dashboard Overview", "Project Assessment Tool"]
)

if page == "Dashboard Overview":
    # Your original dashboard code
    # Header section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">ESG Risk Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-powered ESG risk assessment and monitoring</p>', unsafe_allow_html=True)
    
    with col2:
        current_time = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        st.markdown(f"**Last updated:** {current_time}")
        
        # Refresh and Export buttons
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("🔄 Refresh Data", key="refresh"):
                st.cache_data.clear()
                st.rerun()
        
        with col2b:
            if st.button("📊 Export Report", key="export"):
                st.session_state.show_export = True
    
    # Load data
    df = load_sample_data()
    predictions, model = train_model_and_predict(df)
    project_data = generate_project_data()
    trend_data = generate_trend_data()
    proxy_vars = calculate_proxy_variables(df)
        
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
        st.markdown("#### Risk Level Trends")
        
        # Create line chart
        fig_line = go.Figure()
        
        fig_line.add_trace(go.Scatter(
            x=trend_data['Month'],
            y=trend_data['Low Risk'],
            mode='lines+markers',
            name='Low Risk',
            line=dict(color='#10b981', width=3),
            marker=dict(size=8)
        ))
        
        fig_line.add_trace(go.Scatter(
            x=trend_data['Month'],
            y=trend_data['Moderate Risk'],
            mode='lines+markers',
            name='Moderate Risk',
            line=dict(color='#f59e0b', width=3),
            marker=dict(size=8)
        ))
        
        fig_line.add_trace(go.Scatter(
            x=trend_data['Month'],
            y=trend_data['High Risk'],
            mode='lines+markers',
            name='High Risk',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=8)
        ))
        
        fig_line.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Key Metrics Section
    st.markdown("### Key Proxy Variables")
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{int(proxy_vars['energy_consumption'])} MWh</div>
            <div class="metric-label">Energy Consumption</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{int(proxy_vars['peak_load'])} MW</div>
            <div class="metric-label">Peak Load</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{int(proxy_vars['renewable_generation'])} %</div>
            <div class="metric-label">Renewable Generation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{proxy_vars['carbon_intensity']} kg CO2/kWh</div>
            <div class="metric-label">Carbon Intensity</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk Summary Section
    st.markdown("### Risk Summary")
    
    summary_col1, summary_col2 = st.columns([1, 2])
    
    with summary_col1:
        total_projects = len(project_data)
        high_pct = round((high_count / total_projects) * 100)
        moderate_pct = round((moderate_count / total_projects) * 100)
        low_pct = round((low_count / total_projects) * 100)
        
        st.markdown(f"""
        **Total Projects:** {total_projects}
        
        **High Risk:** {high_count} ({high_pct}%)
        
        **Moderate Risk:** {moderate_count} ({moderate_pct}%)
        
        **Low Risk:** {low_count} ({low_pct}%)
        """)
    
    with summary_col2:
        # Create summary DataFrame for export
        summary_df = pd.DataFrame({
            'Project Name': [p['project_name'] for p in project_data],
            'Risk Level': [p['risk_level'] for p in project_data],
            'ESG Score': [round(p['score'], 1) for p in project_data],
            'Last Updated': [p['last_updated'].strftime('%Y-%m-%d %H:%M') for p in project_data]
        })
        
        st.dataframe(summary_df.head(10), use_container_width=True)

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
                    <span class="status-indicator status-low"></span>
                    {project['risk_level']}
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
                    <span class="status-indicator status-moderate"></span>
                    {project['risk_level']}
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
                    <span class="status-indicator status-high"></span>
                    {project['risk_level']}
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
            'Energy_Consumption_MWh': [proxy_vars['energy_consumption']] * len(project_data),
            'Peak_Load_MW': [proxy_vars['peak_load']] * len(project_data),
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

elif page == "Project Assessment Tool":
    # Assessment Tool Page
    st.markdown('<h1 class="main-header">🌱 GreenPulse</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered ESG Environmental Assessment Tool</p>', unsafe_allow_html=True)
    
    # Main assessment interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📊 Project Assessment")
        
        st.markdown("### Basic Information")
        
        # Input fields
        monthly_energy = st.number_input(
            "⚡ Monthly Energy Use (kWh)",
            min_value=0,
            max_value=1000,
            value=120,
            help="Total monthly energy consumption in kilowatt-hours"
        )
        
        col_solar, col_wind = st.columns(2)
        
        with col_solar:
            solar_power = st.number_input(
                "☀️ Solar Power (kW)",
                min_value=0,
                max_value=100,
                value=25,
                help="Solar power generation capacity"
            )
        
        with col_wind:
            wind_power = st.number_input(
                "💨 Wind Power (kW)",
                min_value=0,
                max_value=100,
                value=10,
                help="Wind power generation capacity"
            )
        
        # Advanced settings (expandable)
        with st.expander("🔧 Advanced Settings (Optional)"):
            power_factor = st.slider(
                "Power Factor",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.05,
                help="Efficiency of power usage (0.8-1.0 typical range)"
            )
            
            user_type = st.selectbox(
                "User Type",
                ["Residential", "Commercial", "Industrial"],
                index=1
            )
            
            temperature = st.slider("Temperature (°C)", -10, 40, 22)
            humidity = st.slider("Humidity (%)", 0, 100, 60)
        
        # Calculate button
        if st.button("Calculate ESG Environmental Score", type="primary", use_container_width=True):
            st.session_state.show_results = True
            st.session_state.esg_score = calculate_esg_score(
                monthly_energy, solar_power, wind_power, power_factor, user_type
            )
    
    with col2:
        # Sidebar info (same as before)
        st.markdown("""
        <div class="sidebar-metric">
            <h4 style="margin: 0; color: #1f2937;">📈 Assessment Factors</h4>
            <ul style="margin: 0.5rem 0; padding-left: 1rem; color: #6b7280; font-size: 0.85rem;">
                <li>Energy consumption efficiency</li>
                <li>Renewable energy integration</li>
                <li>Grid stability metrics</li>
                <li>Environmental conditions</li>
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
        
        # Scatter plot showing assessment point
        st.markdown("### 📈 Model Performance & Your Assessment")
        
        # Load data and create predictions for the scatter plot
        df = load_sample_data()
        predictions, model = train_model_and_predict(df)
        
        # Create results dataframe for scatter plot
        X_test = df.sample(200, random_state=42)  # Sample for visualization
        y_test = X_test['Energy_Efficiency_Score']
        
        # Get predictions for test set
        feature_columns = [
            'Power_Consumption_kWh', 'Voltage_V', 'Current_A', 'Power_Factor',
            'Grid_Frequency_Hz', 'Reactive_Power_kVAR', 'Active_Power_kW',
            'Solar_Power_Generation_kW', 'Wind_Power_Generation_kW',
            'Temperature_C', 'Humidity_%', 'Demand_Response_Event',
            'Previous_Day_Consumption_kWh', 'Peak_Load_Hour', 'Normalized_Consumption'
        ]
        
        # Encode categorical variables for test set
        X_test_encoded = X_test.copy()
        X_test_encoded['User_Type_Residential'] = (X_test['User_Type'] == 'Residential').astype(int)
        X_test_encoded['User_Type_Industrial'] = (X_test['User_Type'] == 'Industrial').astype(int)
        X_test_encoded['User_Type_Commercial'] = (X_test['User_Type'] == 'Commercial').astype(int)
        
        X_test_encoded['Energy_Source_Grid'] = (X_test['Energy_Source_Type'] == 'Grid').astype(int)
        X_test_encoded['Energy_Source_Solar'] = (X_test['Energy_Source_Type'] == 'Solar').astype(int)
        X_test_encoded['Energy_Source_Wind'] = (X_test['Energy_Source_Type'] == 'Wind').astype(int)
        X_test_encoded['Energy_Source_Hybrid'] = (X_test['Energy_Source_Type'] == 'Hybrid').astype(int)
        
        feature_columns_full = feature_columns + [
            'User_Type_Residential', 'User_Type_Industrial', 'User_Type_Commercial',
            'Energy_Source_Grid', 'Energy_Source_Solar', 'Energy_Source_Wind', 'Energy_Source_Hybrid'
        ]
        
        y_pred = model.predict(X_test_encoded[feature_columns_full])
        
        # Classify predictions
        risk_classifications = [classify_risk(pred).replace(' ESG Risk', '') for pred in y_pred]
        
        # Create scatter plot
        fig_scatter = go.Figure()
        
        # Color mapping
        color_map = {"High": "#ef4444", "Moderate": "#f59e0b", "Low": "#10b981"}
        
        # Add scatter points for each risk category
        for risk_cat in ["High", "Moderate", "Low"]:
            mask = [r == risk_cat for r in risk_classifications]
            if any(mask):
                actual_vals = y_test[mask]
                pred_vals = y_pred[mask]
                
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
        
        # Add the assessment point with X marker
        assessment_risk = classify_risk(score).replace(' ESG Risk', '')
        fig_scatter.add_trace(go.Scatter(
            x=[score],  # Use same value for both x and y to show on diagonal
            y=[score],
            mode='markers',
            name='Your Assessment',
            marker=dict(
                symbol='x',
                color='black',
                size=15,
                line=dict(width=3)
            ),
            showlegend=True
        ))
        
        # Update layout
        fig_scatter.update_layout(
            title="Actual vs. Predicted ESG Scores",
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
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        recommendations = []
        if monthly_energy > 100:
            recommendations.append("• Consider energy efficiency upgrades to reduce consumption")
        if solar_power + wind_power < 20:
            recommendations.append("• Increase renewable energy capacity")
        if power_factor < 0.85:
            recommendations.append("• Improve power factor correction")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(rec)
        else:
            st.markdown("• Excellent performance! Continue current sustainability practices")

# Initialize session state
if 'show_export' not in st.session_state:
    st.session_state.show_export = False
if 'show_results' not in st.session_state:
    st.session_state.show_results = False