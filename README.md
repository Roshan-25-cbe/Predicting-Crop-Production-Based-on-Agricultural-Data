🌾 Crop Production Prediction: A Smart Farming Solution
Introduction
This project develops a predictive model and an interactive Streamlit web application to forecast crop production (in tons). By analyzing key agricultural factors, the solution aims to provide valuable insights for strategic planning in agriculture, enhancing food security and optimizing resource management.

Business Use Cases
Food Planning & Security: Aids governments and NGOs in effective food supply management.

Smart Agricultural Policies: Informs policymakers for impactful subsidies, insurance, and relief programs.

Optimized Supply Chains: Helps agribusinesses streamline storage, transportation, and market supply.

Market Price Forecasting: Empowers farmers and traders with insights for optimal selling decisions.

Precision Farming Guidance: Guides farmers in selecting suitable crops and optimizing resource usage.

Agri-Tech Solutions Development: Provides crucial data for developing innovative agricultural tools.

Data Sources & Guidelines
The project primarily utilizes the FAOSTAT_data.csv dataset, a comprehensive source of agricultural statistics.

Data Files: FAOSTAT_data.csv (raw data), FAOSTAT_data_cleaned.csv (processed data).

The project adheres to best practices for data handling and analysis.

File Structure
The core project files are organized as follows:

Data_preprocessing.py: Handles initial data loading, consolidation, cleaning, type conversion, and feature engineering.

EDA.py: Performs comprehensive Exploratory Data Analysis, generating insights and static plots.

model_building.py: Contains scripts for training, evaluating, and saving the machine learning model.

Streamlit_app.py: The main Streamlit application script for the interactive dashboard.

requirements.txt: Lists all Python libraries required to run the project.

.gitignore: Specifies files and directories to be ignored by Git (e.g., raw data, trained model, virtual environment).

eda_plots/: (Optional, if you pushed your local plots here) Directory containing EDA generated plots.

Your_Presentation_File.pptx: (Optional) Your project presentation slides.

How to Run Locally
Follow these steps to set up and run the project on your machine:

Clone the repository:

git clone https://github.com/Roshan-25-cbe/Predicting-Crop-Production-Based-on-Agricultural-Data.git
cd Predicting-Crop-Production-Based-on-Agricultural-Data

Download Raw Data:

Ensure your FAOSTAT_data.csv file is placed in the root of your project folder. This file is not included in the repository due to its size.

Create and activate a virtual environment:

python -m venv .venv

Windows (Command Prompt / PowerShell): .\.venv\Scripts\activate

macOS/Linux: source ./.venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Prepare the Data:

Run the data preprocessing script to create the clean dataset:

python Data_preprocessing.py

This will generate FAOSTAT_data_cleaned.csv.

Run Exploratory Data Analysis (EDA):

python EDA.py

This will generate plots in the eda_plots/ directory.

Build and Save the Machine Learning Model:

python model_building.py

This will generate best_crop_production_model.pkl.

Run the Streamlit Dashboard:

streamlit run Streamlit_app.py

This will open the interactive application in your web browser (usually http://localhost:8501).

Note: For the "Historical Context" section in the app to populate, select an "Area" and "Crop Type" combination that exists in your FAOSTAT_data_cleaned.csv historical data.

Technologies Used
Python (Pandas, NumPy, Matplotlib, Seaborn)

Streamlit

Scikit-learn

Pickle

Key Findings & Insights
Through a structured analysis approach (Data Preparation -> EDA -> Model Building -> Evaluation), several key insights were derived:

Dominant Crops & Regions: Identified top cultivated crops and most agriculturally active regions in the dataset.

Temporal Trends: Observed how average 'Area harvested', 'Yield', and 'Production' have changed over the years, revealing overall growth trends.

Strong Relationships: Confirmed a high correlation between 'Area harvested' and 'Production', and 'Yield' and 'Production', validating their significance as predictive features.

Model Performance: The Random Forest Regressor demonstrated exceptional accuracy (R2 > 0.99), proving highly effective for forecasting crop production.

Actionable Insights: The project provides recommendations for strategic resource allocation, yield improvement initiatives, proactive supply chain management, and maintaining market stability.

Conclusion & Recommendations
Based on the analysis, we recommend:

Prioritizing Resource Allocation: Focus investments and support on consistently high-producing regions and crops to maximize overall output.

Optimizing Yield: Implement advanced farming techniques and farmer education in areas with lower yields but significant harvested areas.

Proactive Supply Chain Management: Utilize production forecasts to streamline storage, transportation, and distribution, reducing waste.

Data-Driven Policy Development: Leverage insights for creating adaptive agricultural policies and ensuring food security.

Presentation
The project has been presented, and you can find the slides here

Future Enhancements
Integrate additional environmental factors (e.g., rainfall, temperature) or economic factors (e.g., market prices).

Develop time series forecasting models for more granular future predictions.

Enhance the Streamlit dashboard with more interactive visualizations and comparative tools.

Project Author
Roshan

GitHub: https://github.com/Roshan-25-cbe

LinkedIn: www.linkedin.com/in/roshan-angamuthu-195ba230a

Contact Email: roshana36822@gmail.com
