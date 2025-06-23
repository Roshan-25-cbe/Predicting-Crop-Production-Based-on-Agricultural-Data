import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Set plot style for better aesthetics
sns.set_style("whitegrid")
# Create a directory to save plots if it doesn't exist
PLOTS_DIR = 'eda_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots will be saved in: '{PLOTS_DIR}'\n")

# --- 1. Load the Cleaned Data ---
try:
    df_cleaned = pd.read_csv('FAOSTAT_data_cleaned.csv')
    print("Cleaned dataset loaded successfully for EDA!\n")
except FileNotFoundError:
    print("Error: 'FAOSTAT_data_cleaned.csv' not found. Please ensure it's in the correct directory and run data_preprocessing.py first.\n")
    exit()
except Exception as e:
    print(f"Error loading data: {e}\n")
    exit()

# Ensure 'Value' is numeric and drop NaNs (consistent with model building)
df_cleaned['Value'] = pd.to_numeric(df_cleaned['Value'], errors='coerce')
df_cleaned.dropna(subset=['Value'], inplace=True)

# Pivot the data to get Area harvested, Yield, Production as columns
df_eda = df_cleaned.pivot_table(index=['Year', 'Area', 'Item'], columns='Element', values='Value').reset_index()
df_eda.columns.name = None # Remove the 'Element' name from columns index

# Rename columns for clarity and consistency with model training
df_eda = df_eda.rename(columns={
    'Area harvested': 'Area_harvested',
    'Yield': 'Yield_kg_ha',
    'Production': 'Production_tons'
})

# Drop rows where any of the critical columns are missing after pivoting
initial_rows_after_pivot = df_eda.shape[0]
df_eda.dropna(subset=['Area_harvested', 'Yield_kg_ha', 'Production_tons'], inplace=True)
rows_dropped_after_pivot = initial_rows_after_pivot - df_eda.shape[0]
if rows_dropped_after_pivot > 0:
    print(f"Dropped {rows_dropped_after_pivot} rows with missing critical values after pivoting.\n")

print(f"Data prepared for EDA. Final shape: {df_eda.shape[0]} rows, {df_eda.shape[1]} columns.\n")
print("Sample of data used for EDA:\n", df_eda.head(), "\n")
print("-" * 70)

# --- 2. Exploratory Data Analysis (EDA) ---

# --- 2.1 Analyze Crop and Geographical Distribution ---
print("--- 2.1 Analyzing Crop and Geographical Distribution ---\n")

# Crop Types: Distribution of the Item column
plt.figure(figsize=(12, 7))
top_crops = df_eda['Item'].value_counts().head(15)
sns.barplot(x=top_crops.values, y=top_crops.index, palette='viridis')
plt.title('Top 15 Most Frequent Crop Types in Records', fontsize=16)
plt.xlabel('Number of Records', fontsize=12)
plt.ylabel('Crop Type', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'top_15_crop_types_frequency.png'))
plt.show()
print(f"Most frequent crop types in the dataset:\n{top_crops}\n")

# Geographical Distribution: Explore the Area column
plt.figure(figsize=(12, 7))
top_areas = df_eda['Area'].value_counts().head(15)
sns.barplot(x=top_areas.values, y=top_areas.index, palette='cividis')
plt.title('Top 15 Areas with Most Records (Agricultural Activity)', fontsize=16)
plt.xlabel('Number of Records', fontsize=12)
plt.ylabel('Area/Region', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'top_15_areas_frequency.png'))
plt.show()
print(f"Top areas with most agricultural activity records:\n{top_areas}\n")
print("-" * 70)

# --- 2.2 Temporal Analysis ---
print("--- 2.2 Performing Temporal Analysis ---\n")

# Yearly Trends: Analyze average Area harvested, Yield, and Production over time
plt.figure(figsize=(18, 6)) # Wider figure for 3 subplots

yearly_trends_avg = df_eda.groupby('Year')[['Area_harvested', 'Yield_kg_ha', 'Production_tons']].mean().reset_index()

plt.subplot(1, 3, 1) # Area Harvested Trend
sns.lineplot(x='Year', y='Area_harvested', data=yearly_trends_avg, marker='o', color='#1f77b4')
plt.title('Average Area Harvested Over Time', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Avg. Area Harvested (hectares)', fontsize=12)
plt.xticks(rotation=45)

plt.subplot(1, 3, 2) # Yield Trend
sns.lineplot(x='Year', y='Yield_kg_ha', data=yearly_trends_avg, marker='o', color='#ff7f0e')
plt.title('Average Yield Over Time', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Avg. Yield (kg/ha)', fontsize=12)
plt.xticks(rotation=45)

plt.subplot(1, 3, 3) # Production Trend
sns.lineplot(x='Year', y='Production_tons', data=yearly_trends_avg, marker='o', color='#2ca02c')
plt.title('Average Production Over Time', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Avg. Production (tons)', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'yearly_trends_avg_metrics.png'))
plt.show()
print("Displayed average yearly trends for Area harvested, Yield, and Production.\n")

# Growth Analysis: Example for a specific crop/region
example_item = 'Maize (corn)'
example_area = 'India'
specific_trend_data = df_eda[(df_eda['Item'] == example_item) & (df_eda['Area'] == example_area)]

if not specific_trend_data.empty:
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Year', y='Production_tons', data=specific_trend_data, marker='o', color='purple')
    plt.title(f'{example_item} Production in {example_area} Over Time', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Production (tons)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{example_item.lower().replace(" ", "_")}_production_{example_area.lower()}_trend.png'))
    plt.show()
    print(f"Sample: {example_item} production trend in {example_area} plotted.\n")
else:
    print(f"No data found for {example_item} in {example_area} for specific trend analysis.\n")
print("-" * 70)

# --- 2.3 & 2.4 Environmental/Input-Output Relationships and Correlations ---
print("--- 2.3 & 2.4 Studying Relationships and Correlations ---\n")

# Correlation Matrix
correlation_matrix = df_eda[['Area_harvested', 'Yield_kg_ha', 'Production_tons']].corr()
print("Correlation Matrix of Key Agricultural Factors:\n", correlation_matrix, "\n")

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix of Agricultural Factors', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'correlation_matrix.png'))
plt.show()
print("Heatmap of correlations between Area harvested, Yield, and Production created.\n")
print("Observations: Production is expected to be highly correlated with Area_harvested and Yield_kg_ha.\n")

# Scatter plots to visualize relationships
plt.figure(figsize=(18, 6)) # Wider figure for 3 subplots

# Area Harvested vs. Production
plt.subplot(1, 3, 1)
sns.scatterplot(x='Area_harvested', y='Production_tons', data=df_eda, alpha=0.6, s=10)
plt.title('Area Harvested vs. Production (Log Scale)', fontsize=14)
plt.xlabel('Area Harvested (hectares)', fontsize=12)
plt.ylabel('Production (tons)', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--", c=".7")

# Yield vs. Production
plt.subplot(1, 3, 2)
sns.scatterplot(x='Yield_kg_ha', y='Production_tons', data=df_eda, alpha=0.6, s=10)
plt.title('Yield vs. Production (Log Scale)', fontsize=14)
plt.xlabel('Yield (kg/ha)', fontsize=12)
plt.ylabel('Production (tons)', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--", c=".7")

# Area Harvested vs. Yield
plt.subplot(1, 3, 3)
sns.scatterplot(x='Area_harvested', y='Yield_kg_ha', data=df_eda, alpha=0.6, s=10)
plt.title('Area Harvested vs. Yield (Log Scale)', fontsize=14)
plt.xlabel('Area Harvested (hectares)', fontsize=12)
plt.ylabel('Yield (kg/ha)', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--", c=".7")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'input_output_scatter_plots.png'))
plt.show()
print("Scatter plots generated to visualize relationships between Area harvested, Yield, and Production.\n")
print("-" * 70)

# --- 2.5 Comparative Analysis ---
print("--- 2.5 Performing Comparative Analysis ---\n")

# Across Crops: Compare average yields
plt.figure(figsize=(14, 8))
avg_yield_by_item = df_eda.groupby('Item')['Yield_kg_ha'].mean().nlargest(20).sort_values(ascending=True)
sns.barplot(x=avg_yield_by_item.values, y=avg_yield_by_item.index, palette='tab10')
plt.title('Top 20 Crops by Average Yield (kg/ha)', fontsize=16)
plt.xlabel('Average Yield (kg/ha)', fontsize=12)
plt.ylabel('Crop Type', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'avg_yield_by_crop.png'))
plt.show()
print("Top 20 crops by average yield displayed.\n")

# Across Regions: Compare average production
plt.figure(figsize=(14, 8))
avg_production_by_area = df_eda.groupby('Area')['Production_tons'].mean().nlargest(20).sort_values(ascending=True)
sns.barplot(x=avg_production_by_area.values, y=avg_production_by_area.index, palette='cubehelix')
plt.title('Top 20 Areas by Average Production (tons)', fontsize=16)
plt.xlabel('Average Production (tons)', fontsize=12)
plt.ylabel('Area/Region', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'avg_production_by_area.png'))
plt.show()
print("Top 20 areas by average production displayed.\n")
print("-" * 70)

# --- 2.6 Productivity Analysis ---
print("--- 2.6 Performing Productivity Analysis ---\n")

# Examine variations in Yield using boxplots
plt.figure(figsize=(14, 8))
top_10_items_for_boxplot = df_eda['Item'].value_counts().head(10).index
sns.boxplot(x='Yield_kg_ha', y='Item', data=df_eda[df_eda['Item'].isin(top_10_items_for_boxplot)], palette='pastel')
plt.title('Yield Distribution (kg/ha) for Top 10 Frequent Crop Types', fontsize=16)
plt.xlabel('Yield (kg/ha)', fontsize=12)
plt.ylabel('Crop Type', fontsize=12)
plt.xscale('log')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'yield_distribution_top_crops_boxplot.png'))
plt.show()
print("Yield distribution for top 10 frequent crop types (boxplot) created.\n")

# Calculate productivity ratios: Production/Area harvested to cross-verify yields.
df_eda_prod_area = df_eda[df_eda['Area_harvested'] > 0].copy()
df_eda_prod_area['Calculated_Yield_kg_ha'] = (df_eda_prod_area['Production_tons'] * 1000) / df_eda_prod_area['Area_harvested']

plt.figure(figsize=(10, 8))
sns.scatterplot(x='Yield_kg_ha', y='Calculated_Yield_kg_ha', data=df_eda_prod_area, alpha=0.5, s=20)
max_val = max(df_eda_prod_area['Yield_kg_ha'].max(), df_eda_prod_area['Calculated_Yield_kg_ha'].max())
min_val = min(df_eda_prod_area['Yield_kg_ha'].min(), df_eda_prod_area['Calculated_Yield_kg_ha'].min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Consistency')
plt.title('Actual Yield vs. Calculated Yield from Production/Area', fontsize=16)
plt.xlabel('Actual Yield (kg/ha)', fontsize=12)
plt.ylabel('Calculated Yield (kg/ha)', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", ls="--", c=".7")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'yield_consistency_check.png'))
plt.show()

print("Calculated yield from Production/Area harvested and compared with reported Yield_kg_ha. Scatter plot shows consistency.\n")
print("Points close to the red dashed line indicate good data consistency.\n")
print("-" * 70)

# --- 2.7 Outliers and Anomalies ---
print("--- 2.7 Identifying Outliers and Anomalies ---\n")

# Using box plots to identify outliers in key numerical features
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Box plot for Area_harvested
sns.boxplot(y=df_eda['Area_harvested'].dropna(), ax=axes[0], color='lightgray')
axes[0].set_title('Outliers in Area Harvested', fontsize=14)
axes[0].set_ylabel('Area Harvested (hectares)', fontsize=12)
if df_eda['Area_harvested'].max() / df_eda['Area_harvested'].min() > 100:
    axes[0].set_yscale('log')

# Box plot for Yield_kg_ha
sns.boxplot(y=df_eda['Yield_kg_ha'].dropna(), ax=axes[1], color='tan')
axes[1].set_title('Outliers in Yield', fontsize=14)
axes[1].set_ylabel('Yield (kg/ha)', fontsize=12)
if df_eda['Yield_kg_ha'].max() / df_eda['Yield_kg_ha'].min() > 100:
    axes[1].set_yscale('log')

# Box plot for Production_tons
sns.boxplot(y=df_eda['Production_tons'].dropna(), ax=axes[2], color='skyblue')
axes[2].set_title('Outliers in Production', fontsize=14)
axes[2].set_ylabel('Production (tons)', fontsize=12)
if df_eda['Production_tons'].max() / df_eda['Production_tons'].min() > 100:
    axes[2].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'outliers_boxplots.png'))
plt.show()
print("Box plots for Area Harvested, Yield, and Production created to visualize outliers.\n")

# Further investigate extreme values: Top/Bottom 5 records for Production
print("\nTop 5 highest production records:\n")
print(df_eda.nlargest(5, 'Production_tons')[['Year', 'Area', 'Item', 'Area_harvested', 'Yield_kg_ha', 'Production_tons']], "\n")

print("Top 5 lowest production records (excluding zeros if any, for meaningful 'lowest'):\n")
print(df_eda[df_eda['Production_tons'] > 0].nsmallest(5, 'Production_tons')[['Year', 'Area', 'Item', 'Area_harvested', 'Yield_kg_ha', 'Production_tons']], "\n")

print("\n--- Comprehensive EDA Complete! ---\n")
print(f"All generated plots are saved in the '{PLOTS_DIR}' folder.")
print("Remember to integrate these insights and plots into your documentation (PowerPoint/Report) to fulfill the EDA and Visualization criteria.")
