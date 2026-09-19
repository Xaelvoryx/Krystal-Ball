# Video Walkthrough Presentation Outline — Krystal Ball

**Project Title:** Krystal Ball — Hotel Bar Inventory Forecasting & Par Level Recommendation System  
**Target Duration:** 3–5 Minutes (4:30 Target Screencast)  
**Presenter:** Senior Data Scientist & Inventory Optimization Specialist  
**Target Audience:** Hotel General Managers, Corporate Beverage Directors & Operations Reviewers  

---

## Slide & Screencast Section Outline

### 0:00 – 0:30 | Section 1: Business Context & Dual Inventory Bottlenecks
* **Visual Cue:** Screen display showing title slide with project title, followed by high-level schematic of hotel bar operations (Main Lounge, Rooftop Bar, Poolside Bar, Executive Lounge).
* **Script / Key Talking Points:**
  - *"Welcome! In hotel beverage operations, bar managers face two major challenges every week."*
  - *"First, stockouts of high-demand spirits—especially during Friday and Saturday night rushes—lead to lost beverage revenue, unsatisfied guests, and service disruptions."*
  - *"Second, overstocking slow-moving premium items ties up critical working capital, occupies limited backroom storage, and increases risk of bottle breakage or spoilage."*
  - *"Today, I'm presenting Krystal Ball: an end-to-end data-driven forecasting and dynamic par level recommendation system designed to optimize bar inventory."*

---

### 0:30 – 1:15 | Section 2: Data Preprocessing, Conservation Audit & Grid Padding
* **Visual Cue:** Transition to Jupyter Notebook showing Section 5–9 (Data Conservation Table and Complete Date Grid code). Highlight the 98.77% conservation validation rate and Cartesian product grid creation.
* **Script / Key Talking Points:**
  - *"We start with 5,040 transaction balance logs covering 180 days across 4 bars and 7 spirits brands."*
  - *"We first rigorously audit the physical conservation equation: `Closing Balance = Opening Balance + Purchase - Consumed`."*
  - *"Our automated audit validated 98.77% of rows within a tight 0.01 ml tolerance, highlighting 62 recording discrepancies for managerial review."*
  - *"Next, we aggregate raw serving timestamps into daily consumption totals and create a complete Cartesian date grid."*
  - *"Crucially, non-service days are padded as `Consumed = 0 ml` rather than omitted, preventing models from over-estimating demand on zero-sales days."*

---

### 1:15 – 2:00 | Section 3: Exploratory Data Analysis & Demand Forecasting Models
* **Visual Cue:** Show ABC Velocity Pareto Chart and Day-of-Week Seasonality Bar Chart, followed by the Model Evaluation Comparison Table (Baseline vs Holt-Winters vs Random Forest).
* **Script / Key Talking Points:**
  - *"Our ABC Pareto analysis classifies fast-moving revenue drivers like Grey Goose and Jack Daniel's as Class A items representing 70% of total volume."*
  - *"Day-of-week analysis confirms significant Friday and Saturday consumption spikes across lounge and rooftop bars."*
  - *"We built three forecasting architectures using a strict 80/20 chronological split to avoid forward-looking data leakage."*
  - *"Evaluating models on WAPE—which safely handles zero-demand days—Random Forest achieved the best performance with an overall WAPE of 0.2268, outperforming both the 7-day rolling mean baseline (0.3845) and Holt-Winters exponential smoothing (0.2982)."*

---

### 2:00 – 2:45 | Section 4: Dynamic Par Level & Safety Stock Formulation
* **Visual Cue:** Show mathematical formulation of dynamic par levels ($Par = Lead Time Demand + Safety Stock$) and the recommended par level summary table.
* **Script / Key Talking Points:**
  - *"Static fixed par levels fail because they ignore demand variance. We implement dynamic par level calculation:"*
  - *"Par Level equals Forecasted Demand during Lead Time plus Safety Stock: `Par = (d_daily * L) + Z * (sigma_daily * sqrt(L))`."*
  - *"Under a 2-day supplier lead time, we calculate dynamic safety stock at both 95% Service Level (`Z = 1.645`) and 99% Service Level (`Z = 2.326`)."*
  - *"This automatically expands buffer stock prior to high-volatility weekends and scales down stock for slow-moving Class C items."*

---

### 2:45 – 3:45 | Section 5: Discrete Daily Inventory Simulation & Policy Backtest
* **Visual Cue:** Show inventory trajectory plot comparing Fixed Baseline vs Dynamic 95% vs Dynamic 99% policies for Grey Goose Vodka at Main Lounge.
* **Script / Key Talking Points:**
  - *"To prove operational value, we built a discrete daily order-up-to inventory simulation backtesting policy performance over 36 days."*
  - *"Our backtest tracks pending orders, delivery arrivals, daily customer consumption, and stockouts."*
  - *"The fixed baseline policy resulted in 42 stockout days and 18.45 liters of lost sales across series."*
  - *"Our recommended Dynamic 95% policy reduced stockouts by 90.5% (down to 4 days) while reducing average holding stock by 11.2%."*
  - *"Upgrading Class A spirits to a 99% Service Level completely eliminated stockouts (0 days, 0 liters lost volume) with only a modest 15-20% increase in holding stock."*

---

### 3:45 – 4:30 | Section 6: Operational Recommendations & Conclusion
* **Visual Cue:** Transition to Executive Summary PDF displaying the 3 core managerial recommendations.
* **Script / Key Talking Points:**
  - *"In conclusion, we recommend bar operations implement three immediate steps:"*
  - *"1. Automate weekly order-up-to par level calculations using our Random Forest forecasting pipeline."*
  - *"2. Apply tiered service levels: 99% Service Level for Class A revenue drivers, and 95% for Class B and C spirits."*
  - *"3. Align liquor distributor order schedules to place purchase orders 2 days prior to peak weekend rushes."*
  - *"Thank you for reviewing Krystal Ball!"*
