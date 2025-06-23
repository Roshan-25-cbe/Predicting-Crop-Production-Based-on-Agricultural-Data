import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pickle # Changed from joblib to pickle

# --- 1. Load the Cleaned and Pivoted Data ---
try:
    df_cleaned = pd.read_csv('FAOSTAT_data_cleaned.csv')
    print("Cleaned dataset loaded successfully for Model Building!\n")
except FileNotFoundError:
    print("Error: 'FAOSTAT_data_cleaned.csv' not found. Please ensure it's in the correct directory.\n")
    exit()

# Ensure 'Value' is numeric again, as it's critical
df_cleaned['Value'] = pd.to_numeric(df_cleaned['Value'], errors='coerce')
df_cleaned.dropna(subset=['Value'], inplace=True)

# Pivot the data to get Area harvested, Yield, Production as columns
df_model = df_cleaned.pivot_table(index=['Year', 'Area', 'Item'], columns='Element', values='Value').reset_index()
df_model.columns.name = None # Remove the 'Element' name from columns index

# Rename columns for clarity (matching EDA script)
df_model = df_model.rename(columns={
    'Area harvested': 'Area_harvested',
    'Yield': 'Yield_kg_ha',
    'Production': 'Production_tons'
})

# Drop rows where any of the critical columns are missing after pivoting
df_model.dropna(subset=['Area_harvested', 'Yield_kg_ha', 'Production_tons'], inplace=True)
print(f"Pivoted data prepared. Shape: {df_model.shape[0]} rows, {df_model.shape[1]} columns.\n")
print("Sample of pivoted data for modeling:\n", df_model.head(), "\n")

# --- 2. Define Features (X) and Target (y) ---
X = df_model[['Area', 'Item', 'Year', 'Area_harvested', 'Yield_kg_ha']]
y = df_model['Production_tons']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}\n")

# --- 3. Identify Categorical and Numerical Features ---
numerical_features = ['Year', 'Area_harvested', 'Yield_kg_ha']
categorical_features = ['Area', 'Item']

# --- 4. Create Preprocessing Pipelines for Numerical and Categorical Features ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- 5. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape (X_train): {X_train.shape}")
print(f"Testing data shape (X_test): {X_test.shape}\n")

# --- 6. Model Training and Evaluation ---
print("--- Starting Model Training and Evaluation ---")

# Define a list of models to evaluate
model_instances = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
}

results = {} # To store evaluation metrics
trained_pipelines = {} # To store trained pipelines

for name, model in model_instances.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    pipeline.fit(X_train, y_train) # Train the model
    y_pred = pipeline.predict(X_test) # Make predictions

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
    trained_pipelines[name] = pipeline # Store the trained pipeline

    print(f"{name} Performance:")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  R-squared (R2): {r2:.4f}")

print("\n--- Model Evaluation Results Summary ---")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# --- 7. Best Model Selection and Saving ---
print("\n--- Selecting and Saving the Best Model ---")

best_model_name = None
best_r2_score = -np.inf # Initialize with negative infinity

for name, metrics in results.items():
    if metrics['R2'] > best_r2_score:
        best_r2_score = metrics['R2']
        best_model_name = name

print(f"\nBased on R-squared ({best_r2_score:.4f}), the BEST model is: {best_model_name}")

# Retrieve the best performing pipeline
best_pipeline = trained_pipelines[best_model_name]

# Define the filename for the saved model
model_filename = 'best_crop_production_model.pkl' # Changed extension to .pkl for pickle

# Save the best model pipeline using pickle
with open(model_filename, 'wb') as file: # 'wb' means write in binary mode
    pickle.dump(best_pipeline, file)
print(f"Best model pipeline saved as '{model_filename}' using pickle.")

print("\n--- Model Building, Evaluation, and Best Model Saving Complete! ---")