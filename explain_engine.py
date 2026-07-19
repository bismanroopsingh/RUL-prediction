import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

from config import MODEL_DIR


class ExplainEngine:
    """
    SHAP backend for Streamlit application.
    """

    def __init__(self):

        self.model = joblib.load(
            MODEL_DIR / "xgboost.pkl"
        )

        self.explainer = shap.TreeExplainer(
            self.model
        )

    ###########################################################
    # Explain Single Engine
    ###########################################################

    def explain(
        self,
        X,
        engine_index=0,
        top_n=10,
    ):

        #######################################################
        # Compute SHAP
        #######################################################

        shap_values = self.explainer.shap_values(X)

        #######################################################
        # Selected Engine
        #######################################################

        sample = X.iloc[[engine_index]]

        sample_shap = shap_values[engine_index]

        #######################################################
        # Top Features
        #######################################################

        importance = pd.DataFrame({

            "Feature": X.columns,

            "SHAP": sample_shap,

            "ABS": abs(sample_shap)

        })

        importance = (

            importance

            .sort_values(
                "ABS",
                ascending=False
            )

            .head(top_n)

        )

        #######################################################
        # Waterfall Plot
        #######################################################

        fig_waterfall = plt.figure(
            figsize=(10,6)
        )

        shap.plots.waterfall(

            shap.Explanation(

                values=sample_shap,

                base_values=self.explainer.expected_value,

                data=sample.iloc[0],

                feature_names=X.columns,

            ),

            show=False

        )

        #######################################################
        # Bar Plot
        #######################################################

        fig_bar = plt.figure(
            figsize=(8,5)
        )

        shap.plots.bar(

            shap.Explanation(

                values=sample_shap,

                base_values=self.explainer.expected_value,

                data=sample.iloc[0],

                feature_names=X.columns,

            ),

            show=False

        )

        #######################################################
        # Return Everything
        #######################################################

        return {

            "importance": importance,

            "waterfall": fig_waterfall,

            "bar": fig_bar,

        }