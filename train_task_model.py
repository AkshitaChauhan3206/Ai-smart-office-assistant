# train_task_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle


# ==============================================
# 1. LOAD TASK WORKFLOW DATASET
# ==============================================

df = pd.read_csv("ai_workflow_optimization_dataset.csv")

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))


# ==============================================
# 2. CONVERT Priority_Level TO NUMBERS
# ==============================================

priority_col = "Priority_Level"

# Map: Low -> 0, Medium -> 1, High -> 2
priority_map = {
    "Low": 0,
    "low": 0,
    "Medium": 1,
    "medium": 1,
    "High": 2,
    "high": 2
}

df[priority_col] = df[priority_col].map(priority_map)

# Drop any rows where Priority_Level could not be mapped
df = df.dropna(subset=[priority_col])

y = df[priority_col].astype(int)
print("Mapped priority labels (0=Low, 1=Medium, 2=High):", y.unique())


# ==============================================
# 3. SELECT ONLY NUMERIC COLUMNS FOR X
# ==============================================

# Only these columns should be numeric
feature_cols = [
    "Estimated_Time_Minutes",
    "Actual_Time_Minutes",
    "Employee_Workload",
    "Approval_Level",
    "Cost_Per_Task"
]

X = df[feature_cols].copy()

# Convert to numeric, any non‑numeric → NaN → fill with median
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median(numeric_only=True))
X = X.astype(float)


# ==============================================
# 4. TRAIN-TEST SPLIT
# ==============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================================
# 5. SCALE & TRAIN MODEL
# ==============================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# ==============================================
# 6. EVALUATE
# ==============================================

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Priority‑model accuracy: {accuracy:.3f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))


# ==============================================
# 7. SAVE MODEL FOR app.py
# ==============================================

with open('task_priority_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'label_mapping': {0: 'Low', 1: 'Medium', 2: 'High'}
    }, f)

print("✅ Task‑priority model saved as task_priority_model.pkl")