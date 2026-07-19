import os
import joblib

from sklearn.model_selection import train_test_split

from src.data_loader import load_training_data
from src.preprocessing import add_rul
from src.feature_engineering import FeatureEngineer
from src.model import XGBoostRUL


# =====================================================
# Create folders
# =====================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================

print("=" * 60)
print("Loading Training Dataset...")

train = load_training_data(
    "data/raw/train_FD001.txt"
)

print(train.head())

# =====================================================
# Generate RUL
# =====================================================

print("=" * 60)
print("Generating Remaining Useful Life...")

train = add_rul(train)

# =====================================================
# Feature Engineering
# =====================================================

print("=" * 60)
print("Running Feature Engineering...")

fe = FeatureEngineer()

train = fe.engineer(train)

# =====================================================
# Prepare Features
# =====================================================

drop_columns = [
    "engine_id",
    "cycle",
    "RUL"
]

X = train.drop(columns=drop_columns)

y = train["RUL"]

feature_names = X.columns.tolist()

joblib.dump(
    feature_names,
    "models/feature_names.pkl"
)

# =====================================================
# Train-Test Split
# =====================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# Scaling
# =====================================================

print("=" * 60)
print("Scaling Features...")

fe.fit_scaler(X_train)

X_train = fe.transform(X_train)

X_val = fe.transform(X_val)

# =====================================================
# Train Model
# =====================================================

print("=" * 60)
print("Training XGBoost...")

model = XGBoostRUL()

model.train(
    X_train,
    y_train
)

# =====================================================
# Evaluate
# =====================================================

print("=" * 60)
print("Validation Results")

metrics = model.evaluate(
    X_val,
    y_val
)

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# =====================================================
# Save Model
# =====================================================

print("=" * 60)
print("Saving Model...")

model.save()

print("Model saved to models/xgboost.pkl")

print("=" * 60)
print("Training Completed Successfully.")