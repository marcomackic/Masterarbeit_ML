import pandas as pd
from datetime import timedelta
from pathlib import Path

# -----------------------------
# Parameter
# -----------------------------
LABEL_WINDOW_DAYS = 3  # Zeitfenster für Wartungslabel (z. B. 3 Tage)

# -----------------------------
# Pfade definieren
# -----------------------------
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_PATH = RAW_DATA_DIR / "measuring_points_full.XLSX"
ORDERS_PATH = RAW_DATA_DIR / "SAP_Orders_w_Sensors.XLSX"
OUTPUT_PATH = PROCESSED_DATA_DIR / "training_data_cycles_labeled.xlsx"

# -----------------------------
# Zyklen-Sensordaten laden & vorbereiten
# -----------------------------
sensor_df = pd.read_excel(SENSOR_PATH)

# Nur Einträge mit "Cycles" in der Bezeichnung des Messpunkts behalten
sensor_df = sensor_df[
    sensor_df["Bezeichnung des Meßpunktes"].str.contains("Cycles", case=False)
].copy()

# Zeitstempel erzeugen
sensor_df["Timestamp"] = pd.to_datetime(sensor_df["Datum"].astype(str) + " " + sensor_df["Meßzeitpunkt"].astype(str))
sensor_df = sensor_df.rename(columns={
    "Equipment": "EquipmentID",
    "Meßwert/GesamtzählSt.": "Sensorwert"
})
sensor_df["Sensorwert"] = sensor_df["Sensorwert"].astype(str).str.replace(",", ".")
sensor_df["Sensorwert"] = pd.to_numeric(sensor_df["Sensorwert"], errors="coerce")
sensor_df["Timestamp_hr"] = sensor_df["Timestamp"].dt.floor("H")

# Aggregation auf Stundenbasis
agg_sensor_df = (
    sensor_df.groupby(["EquipmentID", "Timestamp_hr"])
    .agg({"Sensorwert": "mean"})
    .reset_index()
    .rename(columns={"Timestamp_hr": "Timestamp"})
)

# -----------------------------
# Auftragsdaten vorbereiten
# -----------------------------
orders_df = pd.read_excel(ORDERS_PATH)
orders_df = orders_df.rename(columns={"Equipment": "EquipmentID"})
orders_df["Timestamp"] = pd.to_datetime(orders_df["Erfassungsdatum"])
orders_df = orders_df[orders_df["Timestamp"] >= pd.to_datetime("2024-09-04")].copy()
orders_df["Timestamp_hr"] = orders_df["Timestamp"].dt.floor("H")

# -----------------------------
# Label erzeugen: Wartung in X Tagen
# -----------------------------
label_window = timedelta(days=LABEL_WINDOW_DAYS)

def label_by_equipment(equipment_id, sensor_data, order_data):
    sensor_sub = sensor_data[sensor_data["EquipmentID"] == equipment_id].sort_values("Timestamp")
    order_sub = order_data[order_data["EquipmentID"] == equipment_id].sort_values("Timestamp_hr")

    merged = pd.merge_asof(
        sensor_sub,
        order_sub[["Timestamp_hr"]],
        left_on="Timestamp",
        right_on="Timestamp_hr",
        direction="forward",
        tolerance=label_window
    )
    merged["Wartung_in_{}Tagen".format(LABEL_WINDOW_DAYS)] = merged["Timestamp_hr"].notna().astype(int)
    return merged

# Auf alle Maschinen anwenden
equipment_ids = agg_sensor_df["EquipmentID"].unique()
labeled_dfs = [label_by_equipment(eid, agg_sensor_df, orders_df) for eid in equipment_ids]
final_df = pd.concat(labeled_dfs).reset_index(drop=True)

# -----------------------------
# Ergebnis speichern
# -----------------------------
final_df.to_excel(OUTPUT_PATH, index=False)
print(f"Trainingsdaten gespeichert unter: {OUTPUT_PATH}")
