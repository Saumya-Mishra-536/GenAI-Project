AI-Driven Analytics System for EV Infrastructure Planning

## 📌 Project Overview

This project builds an **AI-driven analytics system** for **Electric Vehicle (EV) infrastructure planning**. It combines classical machine learning with agentic AI to predict charging demand and generate smart infrastructure recommendations.

The project is divided into **2 Milestones:**

### 🏁 Milestone 1 – Predicting EV Charging Demand (Classical ML)
Using historical charging station data, time, and location features to **predict EV charging demand** with classical machine learning models.

### 🤖 Milestone 2 – Agentic AI Assistant (GenAI)
Evolving the system into an **agentic AI assistant** that:
- Reasons about charging demand patterns
- Retrieves infrastructure planning guidelines
- Generates structured recommendations for EV infrastructure

---

## 📁 Project Structure

```
GenAI-Project/
│
├── data/
│   └── ev_charging_patterns.csv       # Raw dataset (from Kaggle)
│
├── output/
│   ├── chart1_charger_type.png        # Sessions by charger type
│   ├── chart2_energy_dist.png         # Energy consumed distribution
│   ├── chart3_hourly.png              # Sessions by hour of day
│   ├── chart4_cost_by_city.png        # Avg charging cost by city
│   ├── chart5_soc_delta.png           # SoC delta by start level
│   ├── chart6_weekend.png             # Weekday vs weekend cost
│   └── ev_engineered.csv              # Dataset with engineered features
│
└── analysis/
    └── ev_analysis.py                 # Feature engineering + EDA script
```

---

## 📊 Dataset Information

- **Source:** [Kaggle – Electric Vehicle Charging Patterns](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns)
- **Size:** 1,320 rows × 20 columns
- **License:** Apache 2.0

### Key Columns

| Column | Description |
|--------|-------------|
| Vehicle Model | EV model (Tesla Model 3, Nissan Leaf, etc.) |
| Battery Capacity (kWh) | Total battery size |
| Charging Station Location | City (New York, LA, Houston, etc.) |
| Charging Start/End Time | Session timestamps |
| Energy Consumed (kWh) | Energy added during session |
| Charging Duration (hours) | Time taken to charge |
| Charging Cost (USD) | Total session cost |
| State of Charge Start/End % | Battery % at start and end |
| Distance Driven (km) | Distance since last charge |
| Temperature (°C) | Ambient temperature |
| Charger Type | Level 1 / Level 2 / DC Fast Charger |
| User Type | Commuter / Casual Driver / Long-Distance Traveler |

---

## 🏁 Milestone 1 – Feature Engineering & EDA

### What was done
- Cleaned and explored the raw dataset
- Engineered new features from raw data
- Generated visualizations to understand patterns

### New Features Created

| Feature | Description |
|---------|-------------|
| `Start Hour` | Hour the session started (0–23) |
| `Start Month` | Month of the session |
| `Is Weekend` | 1 = Weekend, 0 = Weekday |
| `SoC Delta` | Battery % gained (End − Start) |
| `Cost per kWh` | Charging cost efficiency |
| `Battery Utilisation (%)` | % of battery capacity used |
| `SoC Start Level` | Critical / Low / Medium / High |

### Charts Generated

| Chart | Description |
|-------|-------------|
| `chart1_charger_type.png` | Sessions per charger type |
| `chart2_energy_dist.png` | Energy consumed distribution |
| `chart3_hourly.png` | Sessions across hours of the day |
| `chart4_cost_by_city.png` | Avg charging cost by city |
| `chart5_soc_delta.png` | SoC gained by starting charge level |
| `chart6_weekend.png` | Weekday vs weekend cost comparison |

---

## 🤖 Milestone 2 – Agentic AI Assistant *(Coming Soon)*

The system will evolve into an agentic AI assistant that:
- Analyzes demand patterns from Milestone 1 predictions
- Retrieves EV infrastructure planning guidelines using RAG (Retrieval-Augmented Generation)
- Generates structured, location-aware recommendations for where and how to expand EV charging infrastructure

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/GenAI-Project.git
cd GenAI-Project
```

### 2. Install Dependencies
```bash
pip3 install pandas numpy matplotlib
```

### 3. Run the Script
```bash
cd analysis
python3 ev_analysis.py
```

### 4. View Output
- Charts saved to the `output/` folder as `.png` files
- Engineered dataset saved as `output/ev_engineered.csv`
- Summary tables printed in the terminal

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| Pandas | Data manipulation & feature engineering |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| scikit-learn *(Milestone 1)* | ML models for demand prediction |
| LangChain / OpenAI *(Milestone 2)* | Agentic AI assistant |

---

## 🌍 Real-World Impact

This project addresses a **real-world sustainability problem** — the need to plan EV charging infrastructure efficiently as EV adoption grows. By predicting demand and generating AI-driven recommendations, this system can help:
- City planners decide where to install new charging stations
- Energy providers forecast load on the grid
- Businesses optimize charging station placement

---

## 📝 Notes

- Dataset is for **learning and ML practice only** — not for academic research
- Run all commands from inside the `analysis/` folder
- Milestone 2 will be updated as the project progresses

---

## 👩‍💻 Author

**Saumya Mishra**
AI-Driven EV Infrastructure Planning — GenAI Project
