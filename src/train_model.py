import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Parameter definieren
# -----------------------------
DATA_PATH = Path("data/processed/training_data_cycles_labeled.xlsx")
MODEL_PATH = Path("models/rf_model.pkl")
CONF_MATRIX_PLOT = Path("outputs/plots/confusion_matrix.png")

# -----------------------------
# Daten laden
# -----------------------------
df = pd.read_excel(DATA_PATH)
LABEL_COLUMN = [col for col in df.columns if col.startswith("Wartung_in_")][0]
print(f"✅ Verwendete Zielspalte: {LABEL_COLUMN}")

# -----------------------------
# Features & Label vorbereiten
# -----------------------------
features = df.drop(columns=["EquipmentID", "Timestamp", "Timestamp_hr", LABEL_COLUMN], errors="ignore")
X = features.select_dtypes(include=["float", "int"])
y = df[LABEL_COLUMN]

# -----------------------------
# Train/Test-Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -----------------------------
# Modell trainieren
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Modell speichern
# -----------------------------
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Modell gespeichert unter: {MODEL_PATH.resolve()}")

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix plotten
# -----------------------------
CONF_MATRIX_PLOT.parent.mkdir(parents=True, exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(CONF_MATRIX_PLOT)
plt.show()
