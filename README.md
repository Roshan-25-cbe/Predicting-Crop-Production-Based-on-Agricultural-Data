🌾 Crop Production Prediction: A Smart Farming Solution
🚀 Project Overview
This project focuses on building a predictive model and an interactive web application to forecast crop production (in tons) based on key agricultural factors. The goal is to provide valuable insights for better agricultural planning, food security, and optimized resource management.

✨ Business Use Cases
Food Planning & Security: Aiding governments and NGOs in managing food supply.

Smart Agricultural Policies: Informing decisions on subsidies, insurance, and relief programs.

Optimized Supply Chains: Helping agribusinesses plan storage and transportation efficiently.

Market Price Forecasting: Empowering farmers and traders with insights for selling decisions.

Precision Farming Guidance: Guiding farmers in optimal crop selection and resource usage.

Agri-Tech Solutions Development: Providing data for innovative farming tools.

💡 Key Features
Data Preparation: Robust cleaning and preprocessing of raw agricultural data.

Exploratory Data Analysis (EDA): In-depth analysis to uncover trends, distributions, and relationships.

Machine Learning Model: A highly accurate regression model for predicting crop production.

Interactive Streamlit App: A user-friendly web interface for real-time predictions and historical context.

📁 Project Structure
Data_preprocessing.py: Script for data cleaning and initial preparation.

EDA.py: Performs comprehensive exploratory data analysis and generates visualizations in the eda_plots/ directory.

model_building.py: Builds, trains, evaluates, and saves the best machine learning model (best_crop_production_model.pkl).

Streamlit_app.py: The main script for the interactive web application.

FAOSTAT_data.csv: The raw input dataset (not committed due to size, but required).

FAOSTAT_data_cleaned.csv: The cleaned and prepared dataset used for modeling and the app.

requirements.txt: Lists all necessary Python libraries to run the project.

.gitignore: Specifies files/folders to be excluded from the Git repository.

eda_plots/: (Directory) Contains all generated plots from EDA.

Your_Presentation_File.pptx: (Optional) Your project presentation slides.

🏃‍♀️ How to Run the Project Locally
Follow these steps to set up and run the project on your machine:

Clone the Repository:

git clone https://github.com/Roshan-25-cbe/Predicting-Crop-Production-Based-on-Agricultural-Data.git
cd Predicting-Crop-Production-Based-on-Agricultural-Data

Download Raw Data:

Please obtain the FAOSTAT_data.csv dataset from its original source (or your project's shared drive/link) and place it directly into the project's root directory. This file is not included in the repository due to its size.

Create and Activate a Virtual Environment:

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install Dependencies:

pip install -r requirements.txt

Run Data Preprocessing:

This script will clean the raw data and save FAOSTAT_data_cleaned.csv.

python Data_preprocessing.py

Run Exploratory Data Analysis (EDA):

This script generates various plots and insights, saving them in the eda_plots/ directory.

python EDA.py

Build and Save the Machine Learning Model:

This script trains the prediction model and saves it as best_crop_production_model.pkl.

python model_building.py

Run the Streamlit Web Application:

Open your web browser and navigate to the local URL displayed in the terminal (usually http://localhost:8501).

streamlit run Streamlit_app.py

Note: When using the Streamlit app, select "Area" and "Crop Type" combinations that exist in your historical FAOSTAT_data_cleaned.csv to see the "Historical Context for Your Inputs" section populated.

📊 Key Insights & Results
Dominant Crops & Regions: Identified top cultivated crops and most agriculturally active regions.

Temporal Trends: Observed how average 'Area harvested', 'Yield', and 'Production' have changed over the years.

Strong Relationships: Confirmed a high correlation between 'Area harvested' and 'Production', and 'Yield' and 'Production', making them strong predictors.

Model Performance: The Random Forest Regressor proved to be the most accurate model, achieving an R-squared (R2) of over 0.99, demonstrating excellent predictive capability.

Actionable Insights: The project provides recommendations for strategic resource allocation, yield improvement, proactive supply chain management, and market stability.

🛠️ Technologies Used
Python

Pandas (Data Manipulation)

NumPy (Numerical Operations)

Scikit-learn (Machine Learning)

Matplotlib (Plotting)

Seaborn (Statistical Data Visualization)

Streamlit (Web Application Framework)

Pickle (Model Persistence)

📞 Contact
For any questions or further details, feel free to reach out:

Roshan A  
roshana36822@gmail.com
