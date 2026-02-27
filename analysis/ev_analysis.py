import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("../data/ev_charging_patterns.csv")
print("Data loaded! Shape:", df.shape)
print(df.head(3))

# ─────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────

# Convert date columns to datetime
df["Charging Start Time"] = pd.to_datetime(df["Charging Start Time"])
df["Charging End Time"]   = pd.to_datetime(df["Charging End Time"])

# Extract hour, month, day info
df["Start Hour"]  = df["Charging Start Time"].dt.hour
df["Start Month"] = df["Charging Start Time"].dt.month
df["Day of Week Num"] = df["Charging Start Time"].dt.dayofweek  # 0=Monday

# Weekend flag (1 = weekend, 0 = weekday)
df["Is Weekend"] = (df["Day of Week Num"] >= 5).astype(int)

# SoC Delta = how much battery was charged
df["SoC Delta"] = df["State of Charge (End %)"] - df["State of Charge (Start %)"]

# Cost per kWh
df["Cost per kWh"] = df["Charging Cost (USD)"] / df["Energy Consumed (kWh)"]

# Battery usage percentage
df["Battery Utilisation (%)"] = (df["Energy Consumed (kWh)"] / df["Battery Capacity (kWh)"]) * 100

# Label the charge level when session started
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

print("\nNew columns added:", ["Start Hour","Start Month","Is Weekend",
      "SoC Delta","Cost per kWh","Battery Utilisation (%)","SoC Start Level"])

# ─────────────────────────────────────────
# STEP 3: SUMMARY TABLES
# ─────────────────────────────────────────

print("\n--- TABLE 1: Basic Stats ---")
cols = ["Energy Consumed (kWh)", "Charging Duration (hours)",
        "Charging Cost (USD)", "SoC Delta", "Battery Utilisation (%)"]
print(df[cols].describe().round(2))

print("\n--- TABLE 2: Avg Metrics by Charger Type ---")
t2 = df.groupby("Charger Type")[["Energy Consumed (kWh)",
     "Charging Duration (hours)", "Charging Cost (USD)"]].mean().round(2)
print(t2)

print("\n--- TABLE 3: Sessions by User Type ---")
print(df["User Type"].value_counts())

print("\n--- TABLE 4: Avg Cost by Location ---")
t4 = df.groupby("Charging Station Location")["Charging Cost (USD)"].mean().round(2)
print(t4.sort_values(ascending=False))

print("\n--- TABLE 5: Sessions by SoC Start Level ---")
t5 = df.groupby("SoC Start Level")[["SoC Delta","Charging Cost (USD)"]].mean().round(2)
print(t5)

# ─────────────────────────────────────────
# STEP 4: CHARTS
# ─────────────────────────────────────────

# --- Chart 1: Sessions by Charger Type ---
plt.figure(figsize=(7, 4))
counts = df["Charger Type"].value_counts()
plt.bar(counts.index, counts.values, color=["steelblue","orange","green"])
plt.title("Sessions by Charger Type")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.savefig("../output/chart1_charger_type.png")
plt.show()
print("Saved chart1")

# --- Chart 2: Energy Consumed Distribution ---
plt.figure(figsize=(7, 4))
plt.hist(df["Energy Consumed (kWh)"].dropna(), bins=25, color="steelblue", edgecolor="white")
mean_val = df["Energy Consumed (kWh)"].mean()
plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.1f}")
plt.title("Energy Consumed Distribution")
plt.xlabel("kWh")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("../output/chart2_energy_dist.png")
plt.show()
print("Saved chart2")

# --- Chart 3: Sessions by Hour of Day ---
plt.figure(figsize=(9, 4))
hourly = df["Start Hour"].value_counts().sort_index()
plt.plot(hourly.index, hourly.values, marker="o", color="steelblue")
plt.fill_between(hourly.index, hourly.values, alpha=0.2, color="steelblue")
plt.title("Sessions by Hour of Day")
plt.xlabel("Hour (0 = midnight, 12 = noon)")
plt.ylabel("Sessions")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("../output/chart3_hourly.png")
plt.show()
print("Saved chart3")

# --- Chart 4: Avg Charging Cost by Location ---
plt.figure(figsize=(8, 4))
loc_cost = df.groupby("Charging Station Location")["Charging Cost (USD)"].mean().sort_values()
plt.barh(loc_cost.index, loc_cost.values, color="coral")
plt.title("Avg Charging Cost by City")
plt.xlabel("Avg Cost (USD)")
plt.tight_layout()
plt.savefig("../output/chart4_cost_by_city.png")
plt.show()
print("Saved chart4")

# --- Chart 5: SoC Delta by Start Level ---
plt.figure(figsize=(7, 4))
order = ["Critical", "Low", "Medium", "High"]
avg_delta = df.groupby("SoC Start Level")["SoC Delta"].mean().reindex(order)
plt.bar(avg_delta.index, avg_delta.values, color=["red","orange","gold","green"])
plt.title("Avg SoC Delta by Starting Charge Level")
plt.ylabel("Avg SoC Delta (%)")
plt.tight_layout()
plt.savefig("../output/chart5_soc_delta.png")
plt.show()
print("Saved chart5")

# --- Chart 6: Weekend vs Weekday Avg Cost ---
plt.figure(figsize=(6, 4))
wk = df.groupby("Is Weekend")["Charging Cost (USD)"].mean()
wk.index = ["Weekday", "Weekend"]
plt.bar(wk.index, wk.values, color=["steelblue", "orange"], width=0.4)
plt.title("Avg Charging Cost: Weekday vs Weekend")
plt.ylabel("Avg Cost (USD)")
plt.tight_layout()
plt.savefig("../output/chart6_weekend.png")
plt.show()
print("Saved chart6")

# ─────────────────────────────────────────
# STEP 5: SAVE ENGINEERED DATASET
# ─────────────────────────────────────────
df.to_csv("../output/ev_engineered.csv", index=False)
print("\nAll done! Engineered CSV saved to output/ev_engineered.csv")