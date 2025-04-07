# Import libraries 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# page layout
st.set_page_config(page_title="HealthCost AI Predictor", layout="wide")
st.title("HealthCost AI Predictor")
st.write("Predict yearly healthcare costs based on patient information")

# read data file
df = pd.read_csv('data/cspuf2022.csv', low_memory=False)

# features and target 
X = df[['CSP_AGE', 'CSP_SEX', 'CSP_RACE', 'CSP_INCOME', 'CSP_NCHRNCND']]
y = df['PAMTTOT']

# data cleaning 
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())
y = pd.to_numeric(y, errors='coerce').fillna(y.median())

#standard scaling 
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# model training 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate model score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# show model performance 
col1,col2 = st.columns([1,1])

with col1:
    st.header("Feature Importance")
    importance = pd.DataFrame({
        'Feature': ['Age', 'Sex', 'Race', 'Income', 'Chronic Conditions'],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.barh(importance['Feature'], importance['Importance'])
    plt.xlabel("Importance")
    st.pyplot(fig)

# input form
st.sidebar.header("Patient Information")

age = st.sidebar.selectbox("Age", ["1", "2", "3"], 
    help="1: Young (0-24), 2: Adult (25-64), 3: Senior (65+)")

sex = st.sidebar.selectbox("Sex", ["1", "2"], 
    help="1: Male, 2: Female")

race = st.sidebar.selectbox("Race", ["1", "2", "3", "4", "5"],
    help="1: White, 2: Black, 3: Native American, 4: Asian, 5: Pacific Islander")

income = st.sidebar.selectbox("Income", ["1", "2", "3", "4", "5"],
    help="1: Poor, 2: Near Poor, 3: Low, 4: Middle, 5: High")

conditions = st.sidebar.selectbox("Chronic Conditions", ["1", "2", "3", "4", "5"],
    help="1: None, 2: Few, 3: Some, 4: High, 5: Many")

# prediction
if st.sidebar.button("Predict Cost"):
    # input
    input_data = pd.DataFrame({
        'CSP_AGE': [int(age)],
        'CSP_SEX': [int(sex) - 1],
        'CSP_RACE': [int(race) - 1],
        'CSP_INCOME': [int(income) - 1],
        'CSP_NCHRNCND': [int(conditions)]
    })
    
    # scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    # display results
    st.header("Predicted Healthcare Cost")
    st.metric("Yearly Cost", f"${prediction:,.2f}")
    
    # show selections
    st.write("### Patient Details")
    st.write(f"- Age: {'Young (0-24)' if age == '1' else 'Adult (25-64)' if age == '2' else 'Senior (65+)'}")
    st.write(f"- Sex: {'Male' if sex == '1' else 'Female'}")
    st.write(f"- Race: {['White', 'Black', 'Native American', 'Asian', 'Pacific Islander'][int(race)-1]}")
    st.write(f"- Income: {['Poor', 'Near Poor', 'Low', 'Middle', 'High'][int(income)-1]}")
    st.write(f"- Chronic Conditions: {['None', 'Few', 'Some', 'High', 'Many'][int(conditions)-1]}")
    
    st.info("This prediction includes medical visits, hospital stays, medications, and other healthcare costs.")
