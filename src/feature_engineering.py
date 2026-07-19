import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """
    Feature engineering pipeline for NASA C-MAPSS Remaining Useful Life prediction.
    """

    def __init__(
        self,
        selected_sensors=None,
        rolling_window=5,
        lag_steps=3,
        ema_span=5,
    ):

        self.rolling_window = rolling_window
        self.lag_steps = lag_steps
        self.ema_span = ema_span

        self.scaler = StandardScaler()

        # Sensors commonly used for FD001
        if selected_sensors is None:
            self.selected_sensors = [
                "sensor_2",
                "sensor_3",
                "sensor_4",
                "sensor_7",
                "sensor_8",
                "sensor_9",
                "sensor_11",
                "sensor_12",
                "sensor_13",
                "sensor_15",
                "sensor_17",
                "sensor_20",
                "sensor_21",
            ]
        else:
            self.selected_sensors = selected_sensors

    ##############################################################
    # Cycle Features
    ##############################################################

    def add_cycle_features(self, df):

        max_cycle = (
            df.groupby("engine_id")["cycle"]
            .transform("max")
        )
        df["cycle_squared"] = df["cycle"] ** 2
        df["cycle_cubed"] = df["cycle"] ** 3

        return df

    ##############################################################
    # Lag Features
    ##############################################################

    def add_lag_features(self, df):

        for sensor in self.selected_sensors:

            for lag in range(1, self.lag_steps + 1):

                df[f"{sensor}_lag_{lag}"] = (
                    df.groupby("engine_id")[sensor]
                    .shift(lag)
                )

        return df

    ##############################################################
    # Rolling Statistics
    ##############################################################

    def add_rolling_features(self, df):

        for sensor in self.selected_sensors:

            grouped = df.groupby("engine_id")[sensor]

            df[f"{sensor}_mean"] = grouped.transform(
                lambda x: x.rolling(
                    self.rolling_window,
                    min_periods=1
                ).mean()
            )

            df[f"{sensor}_std"] = grouped.transform(
                lambda x: x.rolling(
                    self.rolling_window,
                    min_periods=1
                ).std()
            )

            df[f"{sensor}_min"] = grouped.transform(
                lambda x: x.rolling(
                    self.rolling_window,
                    min_periods=1
                ).min()
            )

            df[f"{sensor}_max"] = grouped.transform(
                lambda x: x.rolling(
                    self.rolling_window,
                    min_periods=1
                ).max()
            )

        return df

    ##############################################################
    # Delta Features
    ##############################################################

    def add_delta_features(self, df):

        for sensor in self.selected_sensors:

            df[f"{sensor}_delta"] = (
                df.groupby("engine_id")[sensor]
                .diff()
            )

        return df

    ##############################################################
    # Exponential Moving Average
    ##############################################################

    def add_ema_features(self, df):

        for sensor in self.selected_sensors:

            df[f"{sensor}_ema"] = (
                df.groupby("engine_id")[sensor]
                .transform(
                    lambda x: x.ewm(
                        span=self.ema_span,
                        adjust=False,
                    ).mean()
                )
            )

        return df

    ##############################################################
    # Expanding Mean
    ##############################################################

    def add_expanding_features(self, df):

        for sensor in self.selected_sensors:

            df[f"{sensor}_expanding_mean"] = (
                df.groupby("engine_id")[sensor]
                .transform(
                    lambda x: x.expanding().mean()
                )
            )

        return df

    ##############################################################
    # Rolling Z-score
    ##############################################################

    def add_zscore_features(self, df):

        for sensor in self.selected_sensors:

            rolling_mean = (
                df.groupby("engine_id")[sensor]
                .transform(
                    lambda x: x.rolling(
                        self.rolling_window,
                        min_periods=1
                    ).mean()
                )
            )

            rolling_std = (
                df.groupby("engine_id")[sensor]
                .transform(
                    lambda x: x.rolling(
                        self.rolling_window,
                        min_periods=1
                    ).std()
                )
            )

            df[f"{sensor}_zscore"] = (
                (df[sensor] - rolling_mean)
                / (rolling_std + 1e-8)
            )

        return df

    ##############################################################
    # Sensor Interaction Features
    ##############################################################

    def add_interaction_features(self, df):

        df["sensor11_sensor15_ratio"] = (
            df["sensor_11"] /
            (df["sensor_15"] + 1e-6)
        )

        df["sensor20_sensor21_product"] = (
            df["sensor_20"] *
            df["sensor_21"]
        )

        df["sensor7_sensor9_difference"] = (
            df["sensor_7"] -
            df["sensor_9"]
        )

        return df

    ##############################################################
    # Missing Values
    ##############################################################

    def fill_missing(self, df):

        df = df.bfill()
        df = df.fillna(0)

        return df

    ##############################################################
    # Scaling
    ##############################################################

    def fit_scaler(self, X):

        self.scaler.fit(X)

        joblib.dump(
            self.scaler,
            "models/scaler.pkl"
        )
    ##############################################################
# Load Scaler
##############################################################

    def load_scaler(self):

        self.scaler = joblib.load(
            "models/scaler.pkl"
        )

    def transform(self, X):

        return self.scaler.transform(X)

    ##############################################################
    # Complete Pipeline
    ##############################################################

    def engineer(self, df):

        print("Creating cycle features...")
        df = self.add_cycle_features(df)

        print("Creating lag features...")
        df = self.add_lag_features(df)

        print("Creating rolling statistics...")
        df = self.add_rolling_features(df)

        print("Creating delta features...")
        df = self.add_delta_features(df)

        print("Creating EMA features...")
        df = self.add_ema_features(df)

        print("Creating expanding statistics...")
        df = self.add_expanding_features(df)

        print("Creating rolling z-score features...")
        df = self.add_zscore_features(df)

        print("Creating interaction features...")
        df = self.add_interaction_features(df)

        print("Handling missing values...")
        df = self.fill_missing(df)

        return df