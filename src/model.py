import joblib
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class XGBoostRUL:

    def __init__(self):

        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

    ##################################################

    def train(self, X_train, y_train):

        self.model.fit(X_train, y_train)

    ##################################################

    def predict(self, X):

        return self.model.predict(X)

    ##################################################

    def evaluate(self, X, y):

        pred = self.predict(X)

        mae = mean_absolute_error(y, pred)

        rmse = mean_squared_error(
            y,
            pred
        ) ** 0.5

        r2 = r2_score(
            y,
            pred
        )

        return {

            "MAE": mae,

            "RMSE": rmse,

            "R2": r2

        }

    ##################################################

    def save(self):

        joblib.dump(
            self.model,
            "models/xgboost.pkl"
        )

    ##################################################

    def load(self):

        self.model = joblib.load(
            "models/xgboost.pkl"
        )