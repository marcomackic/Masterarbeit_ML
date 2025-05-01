import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Pfade und Parameter
# -----------------------------
DATA_PATH = Path("data/processed/training_data_cycles_labeled.xlsx")

# -----------------------------
# Daten laden und vorbereiten
# -----------------------------
df = pd.read_excel(DATA_PATH)
LABEL_COLUMN = [col for col in df.columns if col.startswith("Wartung_in_")][0]

# Feature Engineering: delta_1h (Zuwachs pro Stunde), rolling_mean_3h, rolling_std_3h
df = df.sort_values(by=["EquipmentID", "Timestamp"])
df["delta_1h"] = df.groupby("EquipmentID")["Sensorwert"].diff()
df["rolling_mean_3h"] = df.groupby("EquipmentID")["Sensorwert"].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
df["rolling_std_3h"] = df.groupby("EquipmentID")["Sensorwert"].rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True)

features = df.drop(columns=["EquipmentID", "Timestamp", "Timestamp_hr", LABEL_COLUMN], errors="ignore")
X = features.select_dtypes(include=["float", "int"]).fillna(0)
y = df[LABEL_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -----------------------------
# Modelle definieren
# -----------------------------
models = {
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "SVM": SVC(class_weight="balanced", probability=True, random_state=42)
}

# -----------------------------
# Training & Evaluation
# -----------------------------
results = []
best_model = None

for name, model in models.items():
    print(f"\n🔍 Trainiere Modell: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Modell": name,
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1-Score": round(f1, 3)
    })

    if name == "Random Forest":
        best_model = model  # Speichere RF für Feature Importance

# -----------------------------
# Ergebnisse anzeigen
# -----------------------------
results_df = pd.DataFrame(results)
print("\n📊 Vergleich der Modelle:")
print(results_df.sort_values("F1-Score", ascending=False))

# -----------------------------
# Balkendiagramm erzeugen
# -----------------------------
results_df.set_index("Modell")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind="bar")
plt.title("Modellvergleich: Klassifikationsmetriken")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/plots/model_comparison.png")
plt.show()

# -----------------------------
# Feature Importance (nur für Random Forest)
# -----------------------------
if best_model is not None:
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45, ha="right")
    plt.ylabel("Wichtigkeit")
    plt.tight_layout()
    plt.savefig("outputs/plots/feature_importance_rf.png")
    plt.show()
