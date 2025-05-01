import joblib
import pandas as pd

# Modell und Beispiel-Daten laden
model = joblib.load("models/rf_model.pkl")
df = pd.read_excel("data/processed/training_data_cycles_labeled.xlsx")

# Gleiche Feature-Vorbereitung wie im Training
X = df.drop(columns=["EquipmentID", "Timestamp", "Timestamp_hr", "Wartung_in_3Tagen"], errors="ignore")
X = X.select_dtypes(include=["float", "int"])

# Prognose durchführen
predictions = model.predict(X)

# Ausgabe mit Label
df["Prediction"] = predictions
print(df[["Timestamp", "EquipmentID", "Prediction"]].head())
