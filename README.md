**🌾 Crop Production Prediction: A Smart Farming Solution**

**Introduction**
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

You can obtain the original dataset from its official source or your project's shared location. This file is excluded from the repository due to its size.

The project adheres to best practices in data science workflow, including data cleaning, robust modeling, and clear visualization.

File Structure
The core project files are organized as follows:

Data_preprocessing.py: Handles data cleaning, preprocessing, and preparation.

EDA.py: Performs comprehensive Exploratory Data Analysis and generates visualizations.

model_building.py: Contains scripts for training, evaluating, and saving the machine learning model.

Streamlit_app.py: The main Streamlit application script for the interactive dashboard.

FAOSTAT_data_cleaned.csv: The processed and cleaned dataset used by the model and app.

requirements.txt: Lists all Python libraries required to run the project.

eda_plots/: Directory containing generated plots from EDA.

.gitignore: Specifies files and directories to be ignored by Git (e.g., raw data, trained model, virtual environment).

Your_Presentation_File.pptx: (Optional) Your project presentation slides.

How to Run Locally
Follow these steps to set up and run the project on your machine:

Clone the repository:

git clone https://github.com/Roshan-25-cbe/Predicting-Crop-Production-Based-on-Agricultural-Data.git
cd Predicting-Crop-Production-Based-on-Agricultural-Data

Download Raw Data:

Place the FAOSTAT_data.csv file into the project's root directory.

Create and activate a virtual environment:

python -m venv venv

Windows (Command Prompt): .\venv\Scripts\activate

Windows (Git Bash / PowerShell): source venv/Scripts/activate

macOS/Linux: source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run Data Preprocessing:

python Data_preprocessing.py

Run Exploratory Data Analysis (EDA):

python EDA.py

Build and Save the Machine Learning Model:

python model_building.py

Run the Streamlit Dashboard:

streamlit run Streamlit_app.py

This will open the application in your web browser (usually http://localhost:8501).

Note: For the "Historical Context" section in the app to populate, select an "Area" and "Crop Type" combination that exists in your FAOSTAT_data_cleaned.csv historical data.

Technologies Used
Python

Streamlit

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Pickle

Presentation
The project has been presented, and you can find the slides here

Project Author
Roshan

GitHub: Roshan-25-cbe

LinkedIn: www.linkedin.com/in/roshan-angamuthu-195ba230a

Contact
For any inquiries or collaboration opportunities, feel free to contact me:

Email: roshana36822@gmail.com
