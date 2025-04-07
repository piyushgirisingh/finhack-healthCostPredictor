
🏥 HealthCost AI Predictor

Welcome to HealthCost AI Predictor, a machine learning-powered web app built with Streamlit that estimates an individual's yearly healthcare cost based on simple personal and medical information.

🚀 Features

- 🧠 AI Model: Uses a trained Random Forest Regressor for accurate cost prediction.
- 📊 Feature Importance: See which factors most influence healthcare costs.
- 🧾 User Inputs: Enter patient information like age, sex, race, income, and chronic conditions.
- 📈 Live Prediction: Instantly receive a cost estimate on the screen.
- 💡 Explanations: Friendly descriptions of what each category means.

📂 Dataset

The model uses data from cspuf2022.csv, which must be placed in the data/ folder:

project/
│
├── data/
│   └── cspuf2022.csv
├── app.py
└── README.txt

📦 Installation

1. Clone the repository:

git clone https://github.com/your-username/healthcost-ai-predictor.git
cd healthcost-ai-predictor

2. Install dependencies:

We recommend using a virtual environment.

pip install -r requirements.txt

Or manually install the essentials:

pip install streamlit pandas numpy scikit-learn matplotlib seaborn

▶️ Run the App

streamlit run app.py

It will open automatically in your browser. If not, check the terminal for a link (usually http://localhost:8501).

🧪 How It Works

1. Data Loading: Reads and processes the healthcare dataset.
2. Preprocessing: Cleans numeric and categorical data, handles missing values, and scales features.
3. Model Training: Trains a Random Forest model on the selected features.
4. Prediction: Takes your inputs from the sidebar and shows a personalized cost prediction.
5. Insights: Visualizes which input features were most important in making the prediction.

🛠️ Tech Stack

- Python
- Streamlit – for interactive UI
- Pandas & NumPy – for data processing
- scikit-learn – for machine learning
- Matplotlib & Seaborn – for plots

🧑‍💻 Example Inputs

- Age Group: Young (0–24), Adult (25–64), Senior (65+)
- Sex: Male or Female
- Race: Includes White, Black, Asian, etc.
- Income Level: From Poor to High Income
- Chronic Conditions: Scale from 1 (few) to 5 (many)

⚠️ Note

Make sure your cspuf2022.csv file exists and is properly formatted. If not, the app will display an error message.

📜 License

This project is for educational and research use only. No real-world medical decisions should be made using this app.
