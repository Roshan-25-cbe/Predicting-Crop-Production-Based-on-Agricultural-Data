import streamlit as st
import pandas as pd
import pickle # Using pickle to load the model
import numpy as np

# --- Configuration for Streamlit Page ---
st.set_page_config(
    page_title="Crop Production Prediction",
    page_icon="🌾", # Added a page icon
    layout="wide" # Changed to wide layout for better visual space
)
st.title("🌾 Crop Production Prediction App")
st.write("Input agricultural factors to estimate crop production (in tons).")

# --- Load Model and Data for Dropdowns (with caching) ---

# Use st.cache_resource for models that are loaded once and don't change
@st.cache_resource
def load_model():
    """Loads the best trained crop production prediction model."""
    try:
        with open('best_crop_production_model.pkl', 'rb') as file:
            model = pickle.load(file)
        # st.success("Prediction model loaded successfully!") # Removed for cleaner UI
        return model
    except FileNotFoundError:
        st.error("Error: Model file 'best_crop_production_model.pkl' not found. Please ensure it's in the same directory as this script.")
        st.stop() # Stop app execution if model is not found
    except Exception as e:
        st.error(f"Error loading model: {e}. Ensure the model was saved correctly with 'pickle'.")
        st.stop() # Stop app execution if model loading fails

# Use st.cache_data for dataframes that are loaded once and don't change frequently
@st.cache_data
def load_and_prepare_data_for_app():
    """
    Loads the cleaned data, pivots it, and prepares it for app use
    (e.g., getting unique values and default means).
    """
    try:
        df_cleaned = pd.read_csv('FAOSTAT_data_cleaned.csv')
        
        # Ensure 'Value' is numeric and drop NaNs, as done in preprocessing
        df_cleaned['Value'] = pd.to_numeric(df_cleaned['Value'], errors='coerce')
        df_cleaned.dropna(subset=['Value'], inplace=True)

        # Pivot the data to get Area harvested, Yield, Production as columns
        # This matches the structure expected by your trained model pipeline
        df_model_ready = df_cleaned.pivot_table(
            index=['Year', 'Area', 'Item'],
            columns='Element',
            values='Value'
        ).reset_index()
        df_model_ready.columns.name = None # Remove the 'Element' name from columns index

        # Rename columns for clarity and consistency with model training
        df_model_ready = df_model_ready.rename(columns={
            'Area harvested': 'Area_harvested',
            'Yield': 'Yield_kg_ha',
            'Production': 'Production_tons'
        })

        # Drop rows where any of the critical columns are missing after pivoting
        df_model_ready.dropna(subset=['Area_harvested', 'Yield_kg_ha', 'Production_tons'], inplace=True)
        
        return df_model_ready

    except FileNotFoundError:
        st.error("Error: 'FAOSTAT_data_cleaned.csv' not found. Please ensure it's in the same directory as this script.")
        st.stop() # Stop app execution if data is not found
    except Exception as e:
        st.error(f"Error loading or processing data for dropdowns/defaults: {e}")
        st.stop() # Stop app execution if data processing fails

# Load model and data
model_pipeline = load_model()
df_app_data = load_and_prepare_data_for_app()

# Get unique values and default means from the loaded and prepared data
unique_areas = sorted(df_app_data['Area'].unique().tolist())
unique_items = sorted(df_app_data['Item'].unique().tolist())
min_year_data = int(df_app_data['Year'].min())
max_year_data = int(df_app_data['Year'].max())

# Calculate mean values for default inputs
# Using .mean() on the dataframe for more realistic defaults
default_area_harvested = float(df_app_data['Area_harvested'].mean()) if not df_app_data.empty else 1000.0
default_yield_kg_ha = float(df_app_data['Yield_kg_ha'].mean()) if not df_app_data.empty else 5000.0


# --- User Input Fields ---
st.header("Enter Details:")

# Using columns for a more organized input layout
col1, col2 = st.columns(2)

with col1:
    selected_area = st.selectbox(
        "Area/Region:",
        unique_areas,
        help="Select the geographical area for prediction."
    )
    selected_item = st.selectbox(
        "Crop Type:",
        unique_items,
        help="Select the type of crop."
    )

with col2:
    selected_year = st.slider(
        "Year:",
        min_value=min_year_data,
        max_value=max_year_data,
        value=max_year_data, # Default to latest year in data
        help="Select the year for which to predict production."
    )
    area_harvested = st.number_input(
        "Area Harvested (hectares):",
        min_value=0.0,
        value=default_area_harvested, # Use data-driven default
        step=100.0,
        format="%.2f", # Format to two decimal places
        help="Enter the area (in hectares) from which the crop was harvested."
    )
    yield_kg_ha = st.number_input(
        "Yield (kg/ha):",
        min_value=0.0,
        value=default_yield_kg_ha, # Use data-driven default
        step=100.0,
        format="%.2f", # Format to two decimal places
        help="Enter the yield (kilograms per hectare) of the crop."
    )

st.markdown("---") # Separator

# --- Prediction Button and Output ---
if st.button("Predict Production", help="Click to get the estimated crop production."):
    # Create a DataFrame for prediction, ensuring column order and names match training
    input_data = pd.DataFrame([[selected_area, selected_item, selected_year, area_harvested, yield_kg_ha]],
                              columns=['Area', 'Item', 'Year', 'Area_harvested', 'Yield_kg_ha'])

    try:
        prediction = model_pipeline.predict(input_data)[0]
        # Ensure production is not negative, it should be at least 0
        if prediction < 0:
            prediction = 0.0
        
        st.success(f"### Estimated Crop Production: **{prediction:,.2f} tons**")
        st.balloons() # Added a fun animation for success!

        # --- Optional: Display Contextual Information ---
        st.markdown("---")
        st.subheader("Historical Context for Your Inputs")

        # Filter historical data based on selected Area and Item
        historical_data_subset = df_app_data[
            (df_app_data['Area'] == selected_area) &
            (df_app_data['Item'] == selected_item)
        ].sort_values('Year')

        if not historical_data_subset.empty:
            st.write(f"Historical data for **{selected_item}** in **{selected_area}**:")
            
            # Display a table of historical data
            st.dataframe(historical_data_subset.set_index('Year')[['Area_harvested', 'Yield_kg_ha', 'Production_tons']])
            
            # Plot historical production trend
            st.line_chart(historical_data_subset.set_index('Year')[['Production_tons']],
                          use_container_width=True,
                          height=300,
                          color="#4CAF50") # Green color for the line chart
            st.caption("Trend of 'Production_tons' over the years for this specific crop and area.")
            
            # Display key statistics
            st.markdown(f"**Key Statistics for {selected_item} in {selected_area} (Historical):**")
            stats = historical_data_subset[['Area_harvested', 'Yield_kg_ha', 'Production_tons']].describe().loc[['mean', 'std', 'min', 'max']]
            st.dataframe(stats)

        else:
            st.info("No historical data available for the selected Area and Crop Type combination.")

    except Exception as e:
        st.error(f"Prediction error: {e}. Please ensure the input values are valid for the model.")
        st.warning("The model might struggle with Area/Crop Type combinations not present in its training data.")

st.markdown("---")
st.caption("Developed as part of the Crop Production Prediction Project by Roshan") # Simple footer
