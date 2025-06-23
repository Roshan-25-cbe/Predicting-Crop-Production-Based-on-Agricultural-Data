import pandas as pd
import numpy as np

# --- 1. Load the Data ---
try:
    df = pd.read_csv('FAOSTAT_data.csv')
    print("Dataset loaded successfully!\n")
except FileNotFoundError:
    print("Error: 'FAOSTAT_data.csv' not found. Ensure the CSV file is in the correct directory.\n")
    exit()

# --- 2. Initial Data Inspection (kept for reference, you've already seen this output) ---
# print("--- Dataset Head ---\n", df.head(), "\n")
# print("--- Dataset Info ---")
# df.info()
# print("\n--- Dataset Description ---\n", df.describe(include='all'), "\n")
# print(f"--- Dataset Shape (Rows, Columns) ---\nRows: {df.shape[0]}, Columns: {df.shape[1]}\n")
# print("--- Missing Values per Column ---\n", df.isnull().sum(), "\n")
# print(f"--- Duplicate Rows ---\nNumber of duplicate rows: {df.duplicated().sum()}\n")
# print("--- Key Categorical Column Unique Values & Filtering Info ---")
# key_cols = ['Element', 'Unit', 'Item', 'Area']
# for col in key_cols:
#     unique_vals = df[col].unique()
#     print(f"--- Unique '{col}' values (first 10 if many) ---")
#     print(unique_vals[:10])
#     if len(unique_vals) > 10:
#         print(f"...and {len(unique_vals) - 10} more unique {col}s.\n")
#     else:
#         print("\n")
# print(f"--- Count of rows for relevant 'Element' types (['Area harvested', 'Yield', 'Production']) ---\n")
# print(df['Element'].value_counts(), "\n")
# print("--- Units for relevant 'Element' types (Verification) ---")
# relevant_elements_check = ['Area harvested', 'Yield', 'Production']
# for element_type in relevant_elements_check:
#     subset = df[df['Element'] == element_type]
#     print(f"Element: '{element_type}', Units: {subset['Unit'].unique()}")
# print("\n--- Sample of Data with Relevant Elements (First 5 rows) ---\n", df[df['Element'].isin(relevant_elements_check)].head())


# --- 3. Data Cleaning and Filtering ---

print("--- Starting Data Cleaning and Filtering ---")
initial_rows = df.shape[0]

# 3.1 Drop irrelevant/redundant columns
columns_to_drop = [
    'Domain Code', 'Domain', 'Area Code (M49)', 'Element Code',
    'Item Code (CPC)', 'Year Code', 'Flag', 'Note'
]
df = df.drop(columns=columns_to_drop, errors='ignore')
print(f"Dropped columns: {columns_to_drop}")

# 3.2 Filter for relevant 'Element' types ('Area harvested', 'Yield', 'Production')
relevant_elements = ['Area harvested', 'Yield', 'Production']
df = df[df['Element'].isin(relevant_elements)].copy() # .copy() to avoid SettingWithCopyWarning
print(f"Filtered for relevant Elements: {relevant_elements}")

# 3.3 Refine Units for 'Yield' and 'Production' to ensure consistency
# Keep 'Yield' only if Unit is 'kg/ha'
df = df[~((df['Element'] == 'Yield') & (df['Unit'] != 'kg/ha'))].copy()
print("Filtered 'Yield' to keep only 'kg/ha' unit.")

# Keep 'Production' only if Unit is 't'
df = df[~((df['Element'] == 'Production') & (df['Unit'] != 't'))].copy()
print("Filtered 'Production' to keep only 't' unit.")

# 3.4 Handle Missing Values
# Drop rows where 'Value' is NaN (critical for target/features)
df.dropna(subset=['Value'], inplace=True)
print("Dropped rows with missing 'Value'.")

# Drop rows with any remaining missing values in 'Unit' or 'Flag Description' (minor)
df.dropna(subset=['Unit', 'Flag Description'], inplace=True)
print("Dropped rows with missing 'Unit' or 'Flag Description'.")

# --- 4. Post-Cleaning Inspection ---

print("\n--- Post-Cleaning Dataset Head (First 5 rows) ---")
print(df.head())

print("\n--- Post-Cleaning Dataset Info ---")
df.info()

print("\n--- Post-Cleaning Missing Values Count ---")
print(df.isnull().sum())

print(f"\nOriginal rows: {initial_rows}")
print(f"Rows after cleaning: {df.shape[0]}")
print(f"Number of rows removed: {initial_rows - df.shape[0]}")

print("\n--- Verify Units after Cleaning ---")
print("Unique Units for 'Area harvested':", df[df['Element'] == 'Area harvested']['Unit'].unique())
print("Unique Units for 'Yield':", df[df['Element'] == 'Yield']['Unit'].unique())
print("Unique Units for 'Production':", df[df['Element'] == 'Production']['Unit'].unique())

print("\n--- Data Cleaning and Filtering Complete! ---")

# save the cleaned DataFrame 
df.to_csv('FAOSTAT_data_cleaned.csv', index=False)
print("\nCleaned data saved to 'FAOSTAT_data_cleaned.csv'")