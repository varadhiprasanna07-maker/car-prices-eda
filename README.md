# 🚗 Car Prices Dataset — Python EDA Assignment

> **Pandas Data Analysis | HeroX Python Assignment**

Complete exploratory data analysis of 558,837 used car listings using Python & Pandas.

---

## 📁 Repository Structure

```
car-prices-eda/
├── Car_Prices_EDA_Assignment.ipynb   ← Main Jupyter Notebook (all tasks)
├── car_prices_analysis.py            ← Standalone Python script
├── car_prices.csv                    ← Dataset
├── plots/                            ← All generated visualizations
│   ├── 1_3a_missing_values_bar.png
│   ├── 1_3b_null_heatmap.png
│   ├── 3_1_correlation_heatmap.png
│   ├── 3_2_avg_price_by_year.png
│   ├── 3_3_avg_price_by_odometer.png
│   ├── 3_4_cars_by_state.png
│   ├── 3_5_avg_price_by_condition_bins5.png
│   ├── 3_6_count_by_condition_bins10.png
│   ├── 3_7a_boxplot_by_color_with_outliers.png
│   └── 3_7b_boxplot_by_color_no_outliers.png
└── README.md
```

---

## 📊 Tasks Completed

### Task 1 — Data Ingestion & Quality Profiling
- **1.1** Loaded dataset, displayed first 5 rows, data types, record count
- **1.2** Shape, column names, and data types inspection
- **1.3** Null detection (bar chart + heatmap), null resolution strategy, duplicate removal

### Task 2 — Dataframe Queries (12 queries)
- 2.1 Average / Min / Max selling price
- 2.2 Unique car colors
- 2.3 Unique brands & models
- 2.4 Cars priced > $165,000
- 2.5 Top 5 most sold models
- 2.6 Avg price by brand
- 2.7 Min price by interior
- 2.8 Highest odometer per year
- 2.9 Car age column
- 2.10 Condition ≥ 48 & odometer > 90,000
- 2.11 State with highest prices for newer cars
- 2.12 Value-for-money makes in top 20% condition

### Task 3 — Data Visualization (7 charts)
- 3.1 Correlation heatmap
- 3.2 Avg price by year (bar chart) — with insights
- 3.3 Avg price by odometer — with trend analysis
- 3.4 Cars sold per state — top 3 identified
- 3.5 Avg price by condition (bins=5) — with insights
- 3.6 Count by condition (bins=10) — with insights
- 3.7 Box plot price by color (with & without outliers)

---

## 🔑 Key Findings

| Finding | Value |
|---|---|
| Dataset size | 558,837 records, 16 columns |
| Price range | $1 – $230,000 (avg $13,611) |
| Top selling model | Nissan Altima (19,349) |
| Highest-priced brand | Rolls-Royce (~$153,488 avg) |
| Top state by volume | FL (82,945 cars) |
| Top state (newer cars) | OH ($28,020 avg for year > 2013) |
| Best value brand | Isuzu (excellent condition, lowest price) |
| Strongest price correlator | MMR (0.98 correlation) |

---

## ⚙️ Setup & Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn

# Run the analysis script
python car_prices_analysis.py

# Or open the Jupyter notebook
jupyter notebook Car_Prices_EDA_Assignment.ipynb
```

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib** — visualizations
- **Seaborn** — statistical plots

---
*Assignment submitted for HeroX Python Data Analysis Program.*
