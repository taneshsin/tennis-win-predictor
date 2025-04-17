# 🎾 Tennis Match Win Predictor

A machine learning project that predicts win probabilities in ATP men’s tennis matches based on court type, match round, player rankings, and other metadata.

This project was built as part of the **IS8036 Final Project** to demonstrate the integration of DataOps, ModelOps, and a user-friendly Streamlit app interface for practical ML deployment.

---

## 🚀 Features

- ✅ Predict win probability between two players
- 🏟️ Inputs include surface, court type, series, round, and set format
- 🧠 XGBoost ML model trained on historical ATP data
- 📈 SHAP explainability to understand feature impact
- 📊 MLflow for experiment tracking and model versioning
- 🌐 Streamlit app for easy and intuitive user interaction

---

## 🛠 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/tennis-win-predictor.git
cd tennis-win-predictor


### 2. Install dependencies
```bash
pip install -r requirements.txt


### 3. Launch the Streamlit app
streamlit run streamlit_app.py
