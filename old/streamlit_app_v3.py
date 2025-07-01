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
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
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
        padding: 0.75rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
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
    .input-group {
        margin-bottom: 1.5rem;
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

# Function to generate project data
@st.cache_data
def generate_project_data():
    """Generate sample project data for dashboard"""
    projects = [
        "Solar Farm Alpha", "Wind Project Beta", "Hydro Station Gamma", 
        "Biomass Plant Delta", "Geothermal Epsilon", "Nuclear Zeta",
        "Coal Plant Eta", "Gas Station Theta", "Battery Storage Iota",
        "Smart Grid Kappa", "Microgrid Lambda", "Power Station Mu"
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

# Function to generate time series data for real-time chart
@st.cache_data
def generate_realtime_data():
    """Generate 48 hours of ESG scoring data"""
    np.random.seed(42)
    timestamps = pd.date_range(start='2024-01-01', periods=48, freq='H')
    
    # Generate realistic ESG scores with some variation
    base_score = 65
    scores = []
    for i in range(48):
        # Add some cyclical pattern (higher during day, lower at night)
        time_factor = np.sin(2 * np.pi * i / 24) * 5
        # Add random variation
        random_factor = np.random.normal(0, 3)
        # Add slight trend
        trend_factor = i * 0.1
        
        score = base_score + time_factor + random_factor + trend_factor
        scores.append(max(0, min(100, score)))  # Clip between 0 and 100
    
    return pd.DataFrame({
        'Timestamp': timestamps,
        'Energy_Efficiency_Score': scores
    })

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

# Sidebar Navigation
st.sidebar.markdown("# 🌱 ESG Dashboard")
page = st.sidebar.selectbox(
    "Navigate to:",
    ["Dashboard Overview", "Project Assessment Tool"]
)

if page == "Dashboard Overview":
    # Header section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">Ecolog-IA ESG Risk Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-powered ESG risk assessment and monitoring</p>', unsafe_allow_html=True)
    
    with col2:
        current_time = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        st.markdown(f"**Last updated:** {current_time}")
        
        if st.button("🔄 Refresh Data", key="refresh"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    df = load_sample_data()
    project_data = generate_project_data()
    realtime_data = generate_realtime_data()
    
    # Key Metrics Section (more compact)
    st.markdown("### Key Metrics")
    
    metric_cols = st.columns(4)
    
    energy_consumption = round(df['Power_Consumption_kWh'].mean(), 0)
    renewable_generation = round((df['Solar_Power_Generation_kW'].mean() + 
                                 df['Wind_Power_Generation_kW'].mean()) / 
                                df['Active_Power_kW'].mean() * 100, 0)
    
    with metric_cols[0]:
        st.metric("Energy Consumption", f"{int(energy_consumption)} MWh", "↓ 5%")
    
    with metric_cols[1]:
        st.metric("Renewable Generation", f"{int(renewable_generation)}%", "↑ 8%")
    
    with metric_cols[2]:
        st.metric("Projects Monitored", "23", "↑ 3")
    
    with metric_cols[3]:
        st.metric("Avg ESG Score", "67.2", "↑ 2.1")
    
    # Charts section (more compact layout)
    st.markdown("### Analytics")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### Risk Distribution")
        
        # Calculate current distribution
        risk_counts = pd.Series([p['risk_level'] for p in project_data]).value_counts()
        low_count = risk_counts.get('Low ESG Risk', 0)
        moderate_count = risk_counts.get('Moderate ESG Risk', 0)
        high_count = risk_counts.get('High ESG Risk', 0)
        
        # Create donut chart
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Low Risk', 'Moderate Risk', 'High Risk'],
            values=[low_count, moderate_count, high_count],
            hole=.3,
            marker_colors=['#10b981', '#f59e0b', '#ef4444']
        )])
        
        fig_donut.update_layout(
            showlegend=True,
            height=300,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with chart_col2:
        st.markdown("#### Real-Time ESG Monitoring (48 Hours)")
        
        # Create real-time line chart
        fig_realtime = go.Figure()
        
        fig_realtime.add_trace(go.Scatter(
            x=realtime_data['Timestamp'],
            y=realtime_data['Energy_Efficiency_Score'],
            mode='lines+markers',
            name='ESG Score',
            line=dict(color='#10b981', width=3),
            marker=dict(size=4),
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        
        fig_realtime.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Time",
            yaxis_title="ESG Score",
            showlegend=False
        )
        
        st.plotly_chart(fig_realtime, use_container_width=True)
    
    # Project Summary (more compact)
    st.markdown("### Recent Project Updates")
    
    # Show only top 6 projects in a more compact format
    recent_projects = sorted(project_data, key=lambda x: x['hours_ago'])[:6]
    
    proj_cols = st.columns(3)
    
    for i, project in enumerate(recent_projects):
        col_idx = i % 3
        risk_color = get_risk_color(project['risk_level'])
        
        with proj_cols[col_idx]:
            st.markdown(f"""
            <div style="background: white; border-left: 4px solid {risk_color}; padding: 1rem; margin-bottom: 0.5rem; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-size: 0.9rem; color: #1f2937;">{project['project_name']}</h4>
                <p style="margin: 0.25rem 0; color: #6b7280; font-size: 0.75rem;">{project['risk_level']}</p>
                <p style="margin: 0; color: #9ca3af; font-size: 0.7rem;">{project['hours_ago']}h ago</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "Project Assessment Tool":
    # Assessment Tool Page
    st.markdown('<h1 class="main-header">🌱 Ecolog-IA</h1>', unsafe_allow_html=True)
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
        # Sidebar info
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
if 'show_results' not in st.session_state:
    st.session_state.show_results = False