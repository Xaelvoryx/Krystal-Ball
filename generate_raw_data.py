"""
Script to generate realistic raw hotel bar inventory transaction data for Krystal Ball project.
Generates 180 days of transaction-level bottle balance records with realistic consumption,
purchases, weekend spikes, intermittent demand, stockout events, and minor conservation errors.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_inventory_dataset():
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    num_days = 180
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    bars = ["Main Lounge", "Rooftop Bar", "Poolside Bar", "Executive Lounge"]
    
    # Brand properties: (Base daily consumption ml, weekend factor, velocity class)
    brands_config = {
        "Grey Goose Vodka": {"base": 1200, "weekend": 2.2, "std": 350, "bottle_size": 1000},
        "Jack Daniel's Bourbon": {"base": 950, "weekend": 2.0, "std": 280, "bottle_size": 1000},
        "Hendrick's Gin": {"base": 800, "weekend": 2.1, "std": 250, "bottle_size": 750},
        "Macallan 12 Scotch": {"base": 400, "weekend": 1.6, "std": 140, "bottle_size": 750},
        "Patron Silver Tequila": {"base": 550, "weekend": 2.3, "std": 190, "bottle_size": 750},
        "Remy Martin XO Cognac": {"base": 120, "weekend": 1.3, "std": 70, "bottle_size": 700},
        "Chartreuse Herbal Liqueur": {"base": 60, "weekend": 1.2, "std": 45, "bottle_size": 700}
    }
    
    records = []
    txn_id = 10001
    
    for bar in bars:
        # Bar multiplier
        bar_mult = 1.3 if bar == "Main Lounge" else (1.1 if bar == "Rooftop Bar" else (0.8 if bar == "Poolside Bar" else 0.5))
        
        for brand, cfg in brands_config.items():
            current_balance = cfg["bottle_size"] * np.random.randint(5, 12)
            
            for day in dates:
                dow = day.weekday() # 0=Mon, 4=Fri, 5=Sat, 6=Sun
                is_weekend = dow in [4, 5]
                
                # Consumption demand calculation
                mult = cfg["weekend"] if is_weekend else 1.0
                mean_demand = cfg["base"] * bar_mult * mult
                std_demand = cfg["std"] * bar_mult
                
                # Zero-demand probability for slow movers
                if cfg["base"] < 200 and not is_weekend and np.random.rand() < 0.4:
                    raw_consumed = 0.0
                else:
                    raw_consumed = max(0.0, np.random.normal(mean_demand, std_demand))
                
                raw_consumed = round(raw_consumed, 1)
                
                # Check for replenishment purchase
                purchase = 0.0
                if current_balance < cfg["bottle_size"] * 2:
                    # Order replenishment bottles (e.g. 6 to 12 bottles)
                    purchase = cfg["bottle_size"] * np.random.choice([6, 12, 18])
                
                opening_balance = current_balance
                effective_stock = opening_balance + purchase
                
                # Fulfilled consumption vs stockout
                if effective_stock >= raw_consumed:
                    actual_consumed = raw_consumed
                    closing_balance = effective_stock - actual_consumed
                else:
                    # Stockout occurred!
                    actual_consumed = effective_stock
                    closing_balance = 0.0
                
                current_balance = closing_balance
                
                # Introduce ~1.2% intentional conservation recording errors in raw dataset to test Section 7 validation
                if np.random.rand() < 0.012 and opening_balance > 50:
                    # Minor error of +- 15 ml in closing balance record
                    error_offset = np.random.choice([-15.0, 20.0, -10.0])
                    recorded_closing = max(0.0, closing_balance + error_offset)
                else:
                    recorded_closing = closing_balance
                
                # Timestamp served (e.g., evening serving hour between 18:00 and 23:30)
                hour = np.random.choice([18, 19, 20, 21, 22, 23])
                minute = np.random.choice([0, 15, 30, 45])
                dt_served = day.replace(hour=hour, minute=minute)
                
                records.append({
                    "Transaction ID": f"TXN-{txn_id}",
                    "Date Time Served": dt_served.strftime("%Y-%m-%d %H:%M:%S"),
                    "Bar Name": bar,
                    "Brand Name": brand,
                    "Opening Balance": round(opening_balance, 1),
                    "Purchase": round(purchase, 1),
                    "Consumed": round(actual_consumed, 1),
                    "Closing Balance": round(recorded_closing, 1)
                })
                txn_id += 1

    df = pd.DataFrame(records)
    output_path = "bar_inventory_project/data/raw/bar_inventory_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated raw inventory dataset with {len(df)} transactions at {output_path}")

if __name__ == "__main__":
    generate_inventory_dataset()
