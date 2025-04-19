# 🎾 Tennis Match Win Predictor

A machine learning project that predicts win probabilities in ATP men’s tennis matches using match conditions, player ranks, and court metadata.

---

## 🚀 Features

- 🎯 Predict win probability between two players
- 🧠 XGBoost model trained on historical ATP data
- 📊 Streamlit UI for easy usage
- 🔬 SHAP explainability for feature insights
- 📉 Data drift detection with Evidently
- 🧪 Experiment tracking with MLflow

## 🗣️ User Feedback Collection

The app includes a built-in feedback form that allows users to:

- ✅ Confirm if the prediction was correct  
- 💬 Share suggestions (e.g., "Player was injured", "UI felt unclear", etc.)  
- 📄 All feedback is stored in a connected Google Sheet for future model improvements

---

## 🛠 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/taneshsin/tennis-win-predictor.git
cd tennis-win-predictor
```

### 2. (Recommended) Create & Activate a Virtual Environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```


### 3. Install the Required Packages
```
pip install -r requirements.txt
```

### 4. If you don't have a requirements.txt, install manually:
```
pip install pandas numpy scikit-learn xgboost shap joblib streamlit mlflow evidently matplotlib
```
### 5. Train the Model (Optional – if you want to retrain)
```
python train_model.py
```

### 6. Launch the Streamlit App
```
streamlit run Streamlit_App_Group_4.py
```


📌 Project Status 
✅ MVP complete 
📈 Model deployed via Streamlit 
🧪 Drift detection integrated
⚙️ Further enhancements possible (live API, more models, CI/CD)
