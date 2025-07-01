# %% [markdown]
# 🟩Import Libraries
# 

# %%
# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# %% [markdown]
# # 📄 Dataset Overview: IoT-Enabled Smart Grid for ESG Assessment
# 
# This dataset is designed to support **AI/ML-based ESG ratings** using IoT-generated energy data from smart grids. It includes **real-time and historical environmental and operational variables**, making it ideal for transition finance use cases where financial statements are not available.
# 
# The primary goal is to assess **energy efficiency and sustainability** of infrastructure operations using **proxy variables** — enabling data-driven ESG scoring aligned with net-zero goals.
# 
# ---
# 
# ## 🔧 Key Dataset Features
# 
# | **Variable Name**                 | **Description**                                                                 |
# |----------------------------------|---------------------------------------------------------------------------------|
# | `Timestamp`                      | Date and time of the observation (hourly resolution)                           |
# | `Power_Consumption_kWh`         | Total power consumed during the hour                                           |
# | `Voltage_V`, `Current_A`         | Electrical parameters to assess power behavior                                 |
# | `Power_Factor`                  | Ratio of usable to apparent power (efficiency indicator)                       |
# | `Grid_Frequency_Hz`             | Grid stability and reliability marker                                          |
# | `Reactive_Power_kVAR`           | Energy that does not perform useful work                                       |
# | `Active_Power_kW`               | Actual power used for performing work                                          |
# | `Solar_Power_Generation_kW`     | Amount of power generated from solar sources                                  |
# | `Wind_Power_Generation_kW`      | Power generated from wind sources                                              |
# | `Energy_Source_Type`            | Type of energy source (grid, solar, hybrid, etc.)                              |
# | `Temperature_C`, `Humidity_%`   | Environmental context data captured from IoT sensors                          |
# | `Demand_Response_Event`         | Indicator of load-shifting or energy-saving events (binary flag)               |
# | `Previous_Day_Consumption_kWh`  | Historical power usage reference                                               |
# | `Peak_Load_Hour`                | Binary flag for high-demand periods                                            |
# | `User_Type`                     | Classification of the user (residential, industrial, commercial, etc.)         |
# | `Normalized_Consumption`        | Power consumption scaled for operational comparability                         |
# | `Energy_Efficiency_Score`       | **Target variable**: ESG-aligned score ranging from 0 (low) to 100 (high)      |
# 
# ---
# 
# This dataset allows us to simulate **dynamic ESG scoring** and evaluate sustainability practices using only real-time, observable data — making it ideal for transition finance modeling using AI and IoT.
# 

# %% [markdown]
# 📁 Load the Dataset
# 
# 

# %%
# Step 2: Load the dataset from your local path
df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\MBD 2024\Term 3\Risk & Fraud\Group Project\iiot_smart_grid_dataset.csv")

# Check structure
print("Dataset shape:", df.shape)
print(df.head())


# %% [markdown]
# 🧹 Preprocess the Data
# 

# %%
# Step 3: Drop unnecessary columns
df = df.drop(columns=["Timestamp", "Weather_Condition"])


# %% [markdown]
# 🎯 Define Features and Target
# 
# 

# %%
# Step 4: Set up features and target
X = df.drop(columns=["Energy_Efficiency_Score"])
y = df["Energy_Efficiency_Score"]


# %% [markdown]
# 🧪 Train-Test Split
# 
# 

# %%
# Step 5: Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% [markdown]
# 🤖 Train Random Forest Model
# 
# 

# %%
# Step 6: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# %% [markdown]
# 🧠 Make Predictions & Evaluate
# 
# 

# %%
# Step 7: Predict and evaluate
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# %% [markdown]
# 📊 Plot Feature Importance
# 
# 

# %%
# Step 8: Visualize feature importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance for ESG Rating Model")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 🔍 Key Insights from Feature Importance
# 
# Our ESG Rating model, built using Random Forest, revealed the following insights based on IoT energy data:
# 
# ### 🔝 Top Predictors of Energy Efficiency
# 
# 1. **Power_Consumption_kWh**  
#    - The strongest predictor of ESG performance.  
#    - Lower and optimized power consumption indicates more sustainable operations.
# 
# 2. **Normalized_Consumption**  
#    - Adjusts for project scale, offering a fair comparison between different sizes.  
#    - Highlights how efficiently energy is used per unit baseline.
# 
# 3. **Peak_Load_Hour**  
#    - Measures how well a project manages energy during high-demand periods.  
#    - High peak loads may indicate inefficiencies or stress on the grid.
# 
# 4. **Solar_Power_Generation_kW**  
#    - A clear proxy for renewable integration.  
#    - Projects with higher solar generation contribute positively to ESG outcomes.
# 
# ### 📉 Low-Impact Variables (in this dataset)
# 
# - `Voltage_V`, `Current_A`, `Grid_Frequency_Hz`: Likely redundant or stable across samples.
# - `Temperature_C`, `Humidity_%`: May influence energy usage indirectly but have little standalone predictive power in this context.
# 
# ---
# 
# These insights validate our approach of using **proxy IoT metrics** (instead of traditional financials) to create **transparent, real-time ESG scores** for transition finance monitoring.
# 

# %% [markdown]
# ### 🔍 Proxy Variables for ESG Scoring
# 
# To replace traditional financial metrics, we leveraged real-time, sensor-driven data as proxy indicators of environmental sustainability. These included:
# 
# - **Power consumption and normalization** to assess energy efficiency
# - **Solar and wind generation** as direct indicators of renewable integration
# - **Peak loads and demand response events** to evaluate grid-conscious behavior
# - **Energy source type and power factor** to infer clean vs. dirty energy usage
# 
# These proxies allowed us to build a robust, explainable ESG Ratings model without relying on P&L, balance sheets, or self-reported ESG disclosures.
# 

# %% [markdown]
# ## 🧮 ESG Risk Classification
# 
# Now that we've successfully predicted Energy Efficiency Scores using our ML model, we translate these continuous values into **actionable ESG risk levels**.
# 
# This classification allows stakeholders (e.g., investors, regulators) to quickly assess the sustainability performance of each project based on its predicted environmental score.
# 
# ### 🟢 Classification Criteria
# 
# We define risk levels as:
# 
# - **Low Risk (Green)** → Score > 70  
#   Indicates high energy efficiency and sustainable practices.
# 
# - **Moderate Risk (Orange)** → Score between 40 and 70  
#   Acceptable performance but potential for improvement.
# 
# - **High Risk (Red)** → Score < 40  
#   Signals inefficient energy use or lack of renewable integration.
# 
# ### 🧠 Why This Matters
# 
# - Converts technical predictions into **clear ESG labels**
# - Helps prioritize attention and action
# - Enhances explainability for non-technical stakeholders
# 
# We’ll now apply this classification to our test set and generate a results table for further analysis.
# 

# %%
# Define classification function
def classify_risk(score):
    if score < 40:
        return "High Risk"
    elif score <= 70:
        return "Moderate Risk"
    else:
        return "Low Risk"

# Apply to predicted scores
results_df = X_test.copy()
results_df["Actual_Score"] = y_test
results_df["Predicted_Score"] = y_pred
results_df["ESG_Risk_Level"] = results_df["Predicted_Score"].apply(classify_risk)

# Show preview
results_df.head(10)


# %% [markdown]
# ## 📊 Visualizing ESG Risk Outcomes
# 
# To make our ESG rating model more interpretable and visually impactful, we’ve created two charts:
# 
# 1. **ESG Risk Level Distribution:**  
#    Shows how many projects fall into each ESG risk tier (High, Moderate, Low) based on predicted scores.
# 
# 2. **Actual vs. Predicted Scores:**  
#    Plots the model’s prediction accuracy and visually highlights each project's risk classification using color coding.
# 
# These visuals are useful for both technical validation and business communication.
# 

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Count ESG risk levels
risk_counts = results_df["ESG_Risk_Level"].value_counts().reindex(["High Risk", "Moderate Risk", "Low Risk"])

# Bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x=risk_counts.index, y=risk_counts.values, palette=["#e74c3c", "#f39c12", "#2ecc71"])
plt.title("ESG Risk Level Distribution")
plt.xlabel("ESG Risk Level")
plt.ylabel("Number of Projects")
plt.tight_layout()
plt.show()


# %%
# Scatter plot to visualize prediction performance
plt.figure(figsize=(8, 5))
sns.scatterplot(x=results_df["Actual_Score"], y=results_df["Predicted_Score"], hue=results_df["ESG_Risk_Level"],
                palette={"High Risk": "#e74c3c", "Moderate Risk": "#f39c12", "Low Risk": "#2ecc71"})

plt.plot([0, 100], [0, 100], '--', color='gray')  # reference line
plt.title("Actual vs. Predicted ESG Scores")
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 📊 ESG Risk Distribution Insights
# 
# - The majority of projects fall into the **Moderate Risk** category, indicating stable but improvable energy efficiency.
# - Around **400 projects are High Risk**, signaling a need for immediate action or deeper investigation.
# - Only **a minority of projects qualify as Low Risk**, showing that even in optimized systems, there's room to expand renewable integration and demand response strategies.
# 
# ## 📈 Model Accuracy Insights (Actual vs. Predicted)
# 
# - The model demonstrates **very strong predictive performance**, with most points closely aligned along the diagonal.
# - ESG risk tiers are visually well-separated, confirming that the ML model not only predicts efficiently but does so in a way that aligns with business logic.
# - This gives stakeholders confidence that the scoring system is both **data-driven and explainable**.
# 

# %%
pip install pandas matplotlib seaborn


# %%
# 📌 Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 Step 2: Load your CSV file
file_path = r"C:\Users\hp\OneDrive\Desktop\MBD 2024\Term 3\Risk & Fraud\Group Project\iiot_smart_grid_dataset.csv"
df = pd.read_csv(file_path)

# 📌 Step 3: Convert timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# 📌 Step 4: Select first 48 hours of ESG scoring data
subset = df.sort_values("Timestamp").head(48)

# 📌 Step 5: Plot the Energy Efficiency Score over time
plt.figure(figsize=(12, 5))
sns.lineplot(x=subset["Timestamp"], y=subset["Energy_Efficiency_Score"], marker="o", color="#2ecc71")

plt.title("Simulated Real-Time ESG Scoring via IoT Data (48 Hours)")
plt.xlabel("Timestamp")
plt.ylabel("Energy Efficiency Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 🔄 Insight from Real-Time ESG Monitoring Chart
# 
# - The ESG score fluctuates significantly over 48 hours, highlighting how energy efficiency is affected by operational and environmental changes.
# - These changes reflect real-time system behavior — making IoT-based ESG tracking a dynamic, transparent alternative to static reporting.
# - Our platform’s AI model allows for hourly ESG scoring, enabling stakeholders to flag underperforming assets instantly and take corrective action.
# 


