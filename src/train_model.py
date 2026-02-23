import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("📂 Loading data...")
df = pd.read_csv(os.path.join(BASE, "data", "Customer-Churn.csv"))
df = df.drop_duplicates()
print(f"   {len(df)} rows, {len(df.columns)} columns")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())

df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]
feature_columns = X.columns.tolist()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print("🔬 Training Logistic Regression...")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train, y_train)
y_pred_log = log_model.predict(x_test)

print("🌳 Training Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)

dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(x_train, y_train)
importance = pd.Series(dt_full.feature_importances_, index=feature_columns)
top_features = importance.sort_values(ascending=False).head(10)

def calc_metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

metrics = {
    "logistic_regression": calc_metrics(y_test, y_pred_log),
    "decision_tree": calc_metrics(y_test, y_pred_dt),
    "feature_importance": top_features.to_dict(),
}

for name, m in [("Logistic Regression", metrics["logistic_regression"]), ("Decision Tree", metrics["decision_tree"])]:
    print(f"   {name}: Acc={m['accuracy']}, Prec={m['precision']}, Rec={m['recall']}, F1={m['f1']}")

print("💾 Saving artifacts...")
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
joblib.dump(log_model, os.path.join(BASE, "models", "churn_log_model.joblib"))
joblib.dump(dt_model, os.path.join(BASE, "models", "churn_dt_model.joblib"))
joblib.dump(scaler, os.path.join(BASE, "models", "scaler.joblib"))
joblib.dump(feature_columns, os.path.join(BASE, "models", "feature_columns.joblib"))
joblib.dump(metrics, os.path.join(BASE, "models", "model_metrics.joblib"))
print("✅ Done!")
