import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import shap

# ============================ PAGE CONFIG ============================
st.set_page_config(
    page_title="🎾 Tennis Win Predictor",
    page_icon="🎯",
    layout="centered"
)

# ============================ LOAD MODEL & RANKS ============================
model = joblib.load("model.pkl")
player_rank_map = joblib.load("player_ranks.pkl")

# ============================ LOAD ENCODING MAPS ============================
surface_values = ['Carpet', 'Clay', 'Grass', 'Hard']
series_values = ['ATP250', 'ATP500', 'Grand Slam', 'International', 'International Gold', 'Masters', 'Masters 1000', 'Masters Cup']
round_values = ['1st Round', '2nd Round', '3rd Round', '4th Round', 'Quarterfinals', 'Round Robin', 'Semifinals', 'The Final']

surface_map = {name: i for i, name in enumerate(surface_values)}
series_map = {name: i for i, name in enumerate(series_values)}
round_map = {name: i for i, name in enumerate(round_values)}
court_map = {"Outdoor": 0, "Indoor": 1}

# ============================ HEADER ============================
st.markdown("<h1 style='text-align: center; color: #336699;'>🎾 ATP Tennis Match Win Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict the win probability of a player using match conditions and ATP rankings.</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================ SIDEBAR INPUTS ============================
st.sidebar.header("🎾 Match Setup")

surface = st.sidebar.selectbox("Surface", surface_values)
court = st.sidebar.selectbox("Court Type", list(court_map.keys()))
series = st.sidebar.selectbox("Series", series_values)
round_ = st.sidebar.selectbox("Round", round_values)
best_of = st.sidebar.selectbox("Best of Sets", [3, 5])

players = sorted([p.title() for p in player_rank_map.keys()])
player_1 = st.sidebar.selectbox("Player 1", players)
player_2 = st.sidebar.selectbox("Player 2", players)

if player_1 == player_2:
    st.sidebar.warning("Please select two different players.")

rank_1 = player_rank_map.get(player_1.title(), 1000)
rank_2 = player_rank_map.get(player_2.title(), 1000)
rank_diff = rank_2 - rank_1

# ============================ PREDICTION ============================
if st.button("🎯 Predict Win Probability") and player_1 != player_2:
    input_data = pd.DataFrame({
        'Surface_Code': [surface_map[surface]],
        'Court_Code': [court_map[court]],
        'Series_Code': [series_map[series]],
        'Round_Code': [round_map[round_]],
        'Best of': [best_of],
        'Rank_1': [rank_1],
        'Rank_2': [rank_2],
        'Rank_Diff': [rank_diff]
    })

    st.session_state['input_data'] = input_data
    st.session_state['prob'] = model.predict_proba(input_data)[0][1]
    st.session_state['player_1'] = player_1
    st.session_state['player_2'] = player_2
    st.session_state['rank_1'] = rank_1
    st.session_state['rank_2'] = rank_2

# ============================ SHOW RESULT ============================
if all(k in st.session_state for k in ['prob', 'input_data']):
    prob = st.session_state['prob']
    input_data = st.session_state['input_data']
    player_1 = st.session_state['player_1']
    player_2 = st.session_state['player_2']
    rank_1 = st.session_state['rank_1']
    rank_2 = st.session_state['rank_2']

    st.markdown("## 🌟 Predicted Win Probabilities")
    col1, col2 = st.columns(2)
    col1.metric(label=player_1, value=f"{prob:.2%}", help=f"ATP Rank: {rank_1}")
    col2.metric(label=player_2, value=f"{(1 - prob):.2%}", help=f"ATP Rank: {rank_2}")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [player_1, player_2],
        [prob, 1 - prob],
        color=['#00FF99', '#FF4C4C'],
        edgecolor='white',
        width=0.5
    )

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color='white')

    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Probability", fontsize=12, color='white')
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#111111')
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    st.pyplot(fig)

    # SHAP Explainability
    if st.checkbox("🔬 Show Advanced Insights (SHAP Explainability)"):
        st.markdown("## 🔍 Why Did the Model Predict This?")

        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)

        shap_df = pd.DataFrame({
            'Feature': input_data.columns,
            'SHAP Value': shap_values.values[0]
        }).sort_values(by='SHAP Value', key=abs, ascending=False)

        st.dataframe(shap_df)

        with st.expander("📊 View SHAP Waterfall Plot"):
            shap.plots.waterfall(shap_values[0], max_display=8, show=False)
            st.pyplot(bbox_inches='tight')

# ============================ FOOTER ============================
st.markdown("---")
st.markdown("<div style='text-align: center;'>Made with ❤️ by <strong>Group 4</strong> | Powered by <strong>XGBoost</strong> & <strong>Streamlit</strong></div>", unsafe_allow_html=True)