import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.feature_selection import VarianceThreshold


class FeatureSelector:
    """
    Feature Selection Pipeline for NASA C-MAPSS

    Steps
    -----
    1. Remove low variance features
    2. Remove highly correlated features
    3. Rank remaining features using XGBoost
    4. Save everything required for inference
    """

    def __init__(
        self,
        variance_threshold=0.0,
        correlation_threshold=0.95,
        top_features=60,
    ):

        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.top_features = top_features

        self.variance_selector = None
        self.removed_columns = None
        self.selected_features = None

    ############################################################
    # Remove Low Variance Features
    ############################################################

    def remove_low_variance(self, X):

        print("Removing low variance features...")

        self.variance_selector = VarianceThreshold(
            threshold=self.variance_threshold
        )

        self.variance_selector.fit(X)

        X = X.loc[
            :,
            self.variance_selector.get_support()
        ]

        return X

    ############################################################
    # Remove Highly Correlated Features
    ############################################################

    def remove_correlated(self, X):

        print("Removing correlated features...")

        corr = X.corr().abs()

        upper = corr.where(

            np.triu(
                np.ones(corr.shape),
                k=1,
            ).astype(bool)

        )

        self.removed_columns = [

            column

            for column in upper.columns

            if any(
                upper[column]
                > self.correlation_threshold
            )

        ]

        X = X.drop(
            columns=self.removed_columns,
            errors="ignore",
        )

        return X

    ############################################################
    # XGBoost Feature Ranking
    ############################################################

    def select_by_xgboost(
        self,
        X,
        y,
    ):

        print("Ranking features using XGBoost...")

        model = xgb.XGBRegressor(

            objective="reg:squarederror",

            n_estimators=300,

            random_state=42,

            n_jobs=-1,

        )

        model.fit(X, y)

        importance = pd.DataFrame({

            "Feature": X.columns,

            "Importance": model.feature_importances_

        })

        importance = importance.sort_values(
            "Importance",
            ascending=False,
        )

        os.makedirs(
            "outputs",
            exist_ok=True,
        )

        importance.to_csv(
            "outputs/feature_importance_all.csv",
            index=False,
        )
        importance.head(self.top_features).to_csv(
        "outputs/selected_feature_importance.csv",
        index=False,
    )

        self.selected_features = (

            importance

            .head(self.top_features)

            ["Feature"]

            .tolist()

        )

        return X[self.selected_features]

    ############################################################
    # Save Pipeline
    ############################################################

    def save_features(self):

        os.makedirs(
            "models",
            exist_ok=True,
        )

        joblib.dump(

            self.selected_features,

            "models/selected_features.pkl",

        )

        joblib.dump(

            self.variance_selector,

            "models/variance_selector.pkl",

        )

        joblib.dump(

            self.removed_columns,

            "models/correlated_columns.pkl",

        )

    ############################################################
    # Load Pipeline
    ############################################################

    def load_features(self):

        self.selected_features = joblib.load(
            "models/selected_features.pkl"
        )

        self.variance_selector = joblib.load(
            "models/variance_selector.pkl"
        )

        self.removed_columns = joblib.load(
            "models/correlated_columns.pkl"
        )

    ############################################################
    # Complete Training Pipeline
    ############################################################

    def fit_transform(
        self,
        X,
        y,
    ):

        X = self.remove_low_variance(X)

        X = self.remove_correlated(X)

        X = self.select_by_xgboost(
            X,
            y,
        )

        self.save_features()

        print()

        print("=" * 60)
        print(
            f"Feature Selection Complete ({len(self.selected_features)} Features)"
        )
        print("=" * 60)

        return X

    ############################################################
    # Transform New Data
    ############################################################

    def transform(self, X):

        if self.variance_selector is None:

            raise ValueError(
                "Variance selector has not been loaded."
            )

        if self.selected_features is None:

            raise ValueError(
                "Selected features have not been loaded."
            )

        X = X.loc[
            :,
            self.variance_selector.get_support()
        ]

        X = X.drop(
            columns=self.removed_columns,
            errors="ignore",
        )

        missing = set(self.selected_features) - set(X.columns)

        if missing:
            raise ValueError(
                f"Missing features during prediction: {missing}"
            )

        X = X[self.selected_features]

        return X