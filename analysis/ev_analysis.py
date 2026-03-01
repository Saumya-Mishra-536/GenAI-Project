import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════
df = pd.read_csv("../data/ev_charging_patterns.csv")
print("Data loaded! Shape:", df.shape)

# ═══════════════════════════════════════════════════
# STEP 2: DATA CLEANING
# ═══════════════════════════════════════════════════
print("\n--- Missing Values Before Cleaning ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Fill missing values with median
df["Energy Consumed (kWh)"].fillna(df["Energy Consumed (kWh)"].median(), inplace=True)
df["Charging Rate (kW)"].fillna(df["Charging Rate (kW)"].median(), inplace=True)
df["Distance Driven (since last charge) (km)"].fillna(
    df["Distance Driven (since last charge) (km)"].median(), inplace=True)

print("\nMissing values after cleaning:", df.isnull().sum().sum(), "(none remaining)")

# ═══════════════════════════════════════════════════
# STEP 3: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════
print("\n--- Feature Engineering ---")

# Convert datetime
df["Charging Start Time"] = pd.to_datetime(df["Charging Start Time"])
df["Charging End Time"]   = pd.to_datetime(df["Charging End Time"])

# --- Time-series features ---
df["Start Hour"]      = df["Charging Start Time"].dt.hour
df["Start Month"]     = df["Charging Start Time"].dt.month
df["Day of Week Num"] = df["Charging Start Time"].dt.dayofweek   # 0 = Monday
df["Is Weekend"]      = (df["Day of Week Num"] >= 5).astype(int)
df["Quarter"]         = df["Charging Start Time"].dt.quarter

# --- Peak period label ---
def peak_flag(hour):
    if 7 <= hour <= 9:
        return "Morning Peak"
    elif 17 <= hour <= 20:
        return "Evening Peak"
    else:
        return "Off Peak"

df["Peak Period"] = df["Start Hour"].apply(peak_flag)

# --- Demand features ---
df["SoC Delta"]               = df["State of Charge (End %)"] - df["State of Charge (Start %)"]
df["Battery Utilisation (%)"] = (df["Energy Consumed (kWh)"] / df["Battery Capacity (kWh)"]) * 100
df["Cost per kWh"]            = df["Charging Cost (USD)"] / df["Energy Consumed (kWh)"]

# --- SoC start level ---
def charge_level(soc):
    if soc < 25:
        return "Critical"
    elif soc < 50:
        return "Low"
    elif soc < 75:
        return "Medium"
    else:
        return "High"

df["SoC Start Level"] = df["State of Charge (Start %)"].apply(charge_level)

# --- Location demand score (how busy each city is) ---
location_counts = df["Charging Station Location"].value_counts()
df["Location Demand Score"] = df["Charging Station Location"].map(location_counts)

print("Features created:")
new_cols = [
    "Start Hour", "Start Month", "Day of Week Num", "Is Weekend", "Quarter",
    "Peak Period", "SoC Delta", "Battery Utilisation (%)",
    "Cost per kWh", "SoC Start Level", "Location Demand Score"
]
for col in new_cols:
    print(f"  + {col}")

# ═══════════════════════════════════════════════════
# STEP 4: SUMMARY TABLES
# ═══════════════════════════════════════════════════
print("\n--- TABLE 1: Basic Statistics ---")
cols = ["Energy Consumed (kWh)", "Charging Duration (hours)",
        "Charging Cost (USD)", "SoC Delta", "Battery Utilisation (%)"]
print(df[cols].describe().round(2))

print("\n--- TABLE 2: Avg Metrics by Charger Type ---")
t2 = df.groupby("Charger Type").agg(
    Sessions        = ("User ID", "count"),
    Avg_kWh         = ("Energy Consumed (kWh)", "mean"),
    Avg_Duration_hr = ("Charging Duration (hours)", "mean"),
    Avg_Cost_USD    = ("Charging Cost (USD)", "mean")
).round(2)
print(t2)

print("\n--- TABLE 3: Demand by Location ---")
t3 = df.groupby("Charging Station Location").agg(
    Sessions = ("User ID", "count"),
    Avg_kWh  = ("Energy Consumed (kWh)", "mean"),
    Avg_Cost = ("Charging Cost (USD)", "mean")
).round(2).sort_values("Sessions", ascending=False)
print(t3)

print("\n--- TABLE 4: Demand by Peak Period ---")
t4 = df.groupby("Peak Period").agg(
    Sessions = ("User ID", "count"),
    Avg_kWh  = ("Energy Consumed (kWh)", "mean"),
    Avg_Cost = ("Charging Cost (USD)", "mean")
).round(2)
print(t4)

print("\n--- TABLE 5: SoC Start Level Summary ---")
t5 = df.groupby("SoC Start Level").agg(
    Sessions  = ("User ID", "count"),
    Avg_Delta = ("SoC Delta", "mean"),
    Avg_kWh   = ("Energy Consumed (kWh)", "mean")
).round(2)
print(t5)

# ═══════════════════════════════════════════════════
# STEP 5: CHARTS
# ═══════════════════════════════════════════════════
print("\n--- Generating Charts ---")

# Chart 1: Demand by Hour of Day
plt.figure(figsize=(10, 4))
hourly = df["Start Hour"].value_counts().sort_index()
plt.plot(hourly.index, hourly.values, marker="o", color="steelblue", linewidth=2)
plt.fill_between(hourly.index, hourly.values, alpha=0.2, color="steelblue")
plt.title("EV Charging Demand by Hour of Day")
plt.xlabel("Hour (0 = Midnight)")
plt.ylabel("Number of Sessions")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("../output/chart1_demand_by_hour.png")
plt.close()
print("Saved chart1_demand_by_hour.png")

# Chart 2: Sessions by Charger Type
plt.figure(figsize=(7, 4))
counts = df["Charger Type"].value_counts()
plt.bar(counts.index, counts.values, color=["steelblue", "orange", "green"])
plt.title("Sessions by Charger Type")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.savefig("../output/chart2_charger_type.png")
plt.close()
print("Saved chart2_charger_type.png")

# Chart 3: Energy Consumed Distribution
plt.figure(figsize=(7, 4))
plt.hist(df["Energy Consumed (kWh)"].dropna(), bins=25, color="steelblue", edgecolor="white")
mean_val = df["Energy Consumed (kWh)"].mean()
plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.1f} kWh")
plt.title("Distribution of Energy Consumed per Session")
plt.xlabel("kWh")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("../output/chart3_energy_dist.png")
plt.close()
print("Saved chart3_energy_dist.png")

# Chart 4: Avg Energy Demand by City
plt.figure(figsize=(8, 4))
loc_kwh = df.groupby("Charging Station Location")["Energy Consumed (kWh)"].mean().sort_values()
plt.barh(loc_kwh.index, loc_kwh.values, color="coral")
plt.title("Avg Energy Demand by City")
plt.xlabel("Avg kWh per Session")
plt.tight_layout()
plt.savefig("../output/chart4_demand_by_city.png")
plt.close()
print("Saved chart4_demand_by_city.png")

# Chart 5: Weekday vs Weekend Demand
plt.figure(figsize=(6, 4))
wk = df.groupby("Is Weekend")["Energy Consumed (kWh)"].mean()
wk.index = ["Weekday", "Weekend"]
plt.bar(wk.index, wk.values, color=["steelblue", "orange"], width=0.4)
plt.title("Avg Energy Demand: Weekday vs Weekend")
plt.ylabel("Avg kWh")
plt.tight_layout()
plt.savefig("../output/chart5_weekday_weekend.png")
plt.close()
print("Saved chart5_weekday_weekend.png")

# Chart 6: Demand by Peak Period
plt.figure(figsize=(7, 4))
peak_kwh = df.groupby("Peak Period")["Energy Consumed (kWh)"].mean().sort_values()
plt.bar(peak_kwh.index, peak_kwh.values, color=["steelblue", "orange", "green"])
plt.title("Avg Energy Demand by Peak Period")
plt.ylabel("Avg kWh")
plt.tight_layout()
plt.savefig("../output/chart6_peak_period.png")
plt.close()
print("Saved chart6_peak_period.png")

# ═══════════════════════════════════════════════════
# STEP 6: SAVE ENGINEERED DATASET
# ═══════════════════════════════════════════════════
df.to_csv("../output/ev_engineered.csv", index=False)
print("\nAll done! Saved to output/ev_engineered.csv")
print(f"Final dataset shape: {df.shape}")