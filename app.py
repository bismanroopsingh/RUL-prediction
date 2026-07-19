import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from config import TEST_PATH
from src.data_loader import load_test_data

from predict_engine import PredictionEngine
from explain_engine import ExplainEngine
st.set_page_config(

    page_title="NASA Predictive Maintenance",

    page_icon="🚀",

    layout="wide",

)
st.title("🚀 NASA C-MAPSS Predictive Maintenance Dashboard")

st.markdown(
"""
Predict the Remaining Useful Life (RUL) of aircraft engines using
an optimized XGBoost model with SHAP Explainable AI.
"""
)

st.divider()
@st.cache_data
def load_dataset():

    return load_test_data(TEST_PATH)
@st.cache_resource
def load_models():

    predictor = PredictionEngine()

    explainer = ExplainEngine()

    return predictor, explainer
test_data = load_dataset()

predictor, explainer = load_models()
st.sidebar.title("Settings")

engine_list = sorted(
    test_data["engine_id"].unique()
)

selected_engine = st.sidebar.selectbox(

    "Select Engine",

    engine_list,

)

predict_button = st.sidebar.button(

    "Predict RUL",

    use_container_width=True,

)
############################################################
# Prediction
############################################################

if predict_button:

    ########################################################
    # Select Engine
    ########################################################

    engine_df = test_data[

        test_data["engine_id"] == selected_engine

    ].copy()

    ########################################################
    # Run Prediction
    ########################################################

    results = predictor.predict(engine_df)

    prediction_df = results["predictions"]

    X = results["processed_features"]

    prediction = prediction_df.iloc[0]
        ########################################################
    # Dashboard Cards
    ########################################################

    st.subheader("Prediction Summary")

    col1, col2, col3, col4 = st.columns(4)

    predicted_rul = max(0.0, predicted_rul := prediction["Predicted_RUL"])

    with col1:

        st.metric(

            "Predicted RUL",

            f"{predicted_rul:.1f} Cycles"

        )

    with col2:

        st.metric(

            "Estimated Failure",

            int(prediction["Failure_Cycle"])

        )

    with col3:

        st.metric(

            "Confidence",

            f"{prediction['Confidence']}%"

        )

    with col4:

        st.metric(

            "Health",

            prediction["Health"]

        )
    ########################################################
    # Health Gauge
    ########################################################

    st.subheader("Engine Health")

    health_score = predicted_rul/ 125

    # Keep value between 0 and 1
    health_score = max(0.0, min(1.0, health_score))

    st.progress(health_score)
            ########################################################
    # Prediction Table
    ########################################################

    st.subheader("Prediction Details")

    st.dataframe(

        prediction_df,

        use_container_width=True,

        hide_index=True,

    )
        ########################################################
    # SHAP Explainability
    ########################################################

    st.divider()

    st.header("Explainable AI")

    shap_result = explainer.explain(

        X,

        engine_index=0,

        top_n=10,

    )
    st.subheader("Top Sensors Affecting Prediction")

    importance = shap_result["importance"]

    st.dataframe(

        importance,

        use_container_width=True,

        hide_index=True,

    )
    st.subheader("Feature Importance")

    st.pyplot(

        shap_result["bar"],

        clear_figure=True,

    )
    st.subheader("Prediction Explanation")

    st.pyplot(

        shap_result["waterfall"],

        clear_figure=True,

    )
    st.divider()

    st.download_button(

        label="Download Prediction",

        data=prediction_df.to_csv(index=False),

        file_name=f"Engine_{selected_engine}_Prediction.csv",

        mime="text/csv",

    )
else:

    st.info(
        "Select an engine from the sidebar and click **Predict RUL**."
    )