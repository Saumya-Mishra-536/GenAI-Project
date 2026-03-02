# 🔋 Intelligent EV Charging Demand Prediction & Agentic Infrastructure Planning

> **Milestone 1 — Data Pipeline & ML Modeling**  
> Mid-Semester Submission | Applied Machine Learning

---

## 📌 Project Overview

This project builds an AI-driven analytics system for Electric Vehicle (EV) infrastructure planning.

- **Milestone 1 (Mid-Sem):** Classical ML pipeline to predict EV charging demand using historical station data  
- **Milestone 2 (End-Sem):** Agentic AI assistant using LangGraph + RAG to generate infrastructure recommendations

---

## 📂 Dataset

- 3 real-world California charging stations (A, B, C)
- **89,715 total records** after consolidation
- Features: Date/Time, Weather, Electricity Price, Grid Stability Index, Grid Availability, Number of EVs Charging
- Target: `EV Charging Demand (kW)`

---

## ⚙️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| ML | Scikit-Learn (Random Forest) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab |
| Model Export | Joblib |

---

## 🔄 Pipeline
```
Raw CSV (3 files) → Consolidation → Cleaning & Imputation → Feature Engineering → Model Training → Evaluation → Export (.pkl)
```

### Phase 1 — Data Loading
- Loaded 3 station CSVs and assigned `Station_ID`
- Unified `Date` + `Time` into a single `Datetime` column
- Consolidated and sorted 89,715 records

### Phase 2 — Data Cleaning
- Median imputation for missing values
- Binary encoding of `Grid Availability` (Available=1, Unavailable=0)
- EDA: heatmaps, boxplots, KDE histogram, pie charts

### Phase 3 — Feature Engineering
- Extracted `Hour` and `DayOfWeek` from datetime
- Created lag features: `Demand_Lag_1`, `Demand_Lag_2`
- Created `Rolling_Avg_3h` (3-step rolling mean, shifted to avoid leakage)

### Phase 4 — Model Training
- 80/20 train-test split (`random_state=42`)
- Trained `RandomForestRegressor` with 100 estimators

### Phase 5 — Evaluation
- Computed R² Score and MAE
- Plotted feature importances

---

## 📊 Results

| Metric | Value |
|--------|-------|
| R² Score | **88.65%** |
| Mean Absolute Error | **0.0192 kW** |
| Total Records | 89,715 |
| Algorithm | Random Forest (100 trees) |

---

## 🧠 Features Used

| # | Feature | Type |
|---|---------|------|
| 1 | `Hour` | Temporal |
| 2 | `DayOfWeek` | Temporal |
| 3 | `Demand_Lag_1` | Autoregressive |
| 4 | `Demand_Lag_2` | Autoregressive |
| 5 | `Rolling_Avg_3h` | Trend |
| 6 | `Electricity Price ($/kWh)` | Economic |
| 7 | `Grid Stability Index` | Grid State |
| 8 | `Number of EVs Charging` | Usage |

---

## 🚀 How to Run
```bash
# 1. Clone the repo
git clone https://github.com/your-repo/ev-charging-prediction.git

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# 3. Open notebook in Google Colab and upload CSV files when prompted
```

---

## 💾 Model Export

The trained model is saved as:
```
ev_demand_timeseries.pkl
```
Ready for integration into **Milestone 2** agentic planning system.

---

## 🔮 Milestone 2 Preview

- **Framework:** LangGraph
- **RAG:** Chroma / FAISS
- **LLM:** Open-source (free tier)
- **UI:** Streamlit / Gradio
- **Hosting:** Hugging Face Spaces / Streamlit Community Cloud

---

## 📋 Constraints

- ✅ Team Size: 4 Students
- ✅ API Budget: Free Tier Only
- ✅ No localhost-only demos (hosted deployment required for Milestone 2)

---

*Built with ❤️ for Applied ML — March 2026*
