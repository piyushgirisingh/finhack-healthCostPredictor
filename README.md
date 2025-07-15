# 🏥 HealthCost AI Predictor

This is a Streamlit web application that predicts yearly healthcare costs based on patient demographic and medical data using a machine learning model.

## 📊 Overview

The app:
- Loads and preprocesses healthcare survey data (`cspuf2022.csv`)
- Trains a Random Forest Regression model
- Shows model performance and feature importance
- Allows users to input patient information to predict healthcare costs

## 🧠 How It Works

1. **Data Preprocessing**:
   - Handles missing values
   - Encodes categorical variables (sex, race, income)
   - Scales numeric data (age, chronic conditions)

2. **Model Training**:
   - Splits the data into training and test sets
   - Trains a `RandomForestRegressor`
   - Evaluates model performance (R² score)

3. **Prediction**:
   - Accepts user input from the sidebar
   - Transforms and scales input
   - Displays a predicted yearly healthcare cost

## 📁 File Structure

```
.
├── main.py                # Streamlit app script
├── data/
│   └── cspuf2022.csv     # Dataset file (must be placed here)
└── README.md             # Project documentation
```

## ⚙️ Requirements

Install the following Python packages:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

## 🚀 Running the App

Make sure your dataset `cspuf2022.csv` is inside a `data` folder, then run:

```bash
streamlit run main.py
```

## 📌 Notes

- The dataset used is from the Medical Expenditure Panel Survey (MEPS) 2022.
- Chronic conditions are categorized from **1 (Low)** to **5 (High)**.
- All inputs are based on predefined categories for age, sex, race, income, and chronic conditions.

## 📷 Example Output

- Feature importance graph
- Predicted yearly healthcare cost
- Explanation of the selected patient profile

## 📬 Contact

For questions or improvements, feel free to reach out or fork this project!


