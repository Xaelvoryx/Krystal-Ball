"""
Data Preprocessing Pipeline for Krystal Ball Project.
Executes timestamp parsing, inventory conservation validation, daily aggregation,
and continuous date grid construction.
"""
import os
import numpy as np
import pandas as pd

def process_inventory_data():
    raw_path = "bar_inventory_project/data/raw/bar_inventory_data.csv"
    processed_path = "bar_inventory_project/data/processed/daily_bar_consumption.csv"
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
        
    df = pd.read_csv(raw_path)
    
    # 1. Parse Timestamps
    df["Date Time Served"] = pd.to_datetime(df["Date Time Served"], errors="coerce")
    invalid_dates = df["Date Time Served"].isna().sum()
    print(f"[Timestamp Check] Total rows: {len(df)}, Invalid Datetimes: {invalid_dates}")
    
    df["Date"] = df["Date Time Served"].dt.floor("D")
    
    # 2. Inventory Conservation Validation
    # Closing Balance = Opening Balance + Purchase - Consumed
    df["Calculated_Closing"] = df["Opening Balance"] + df["Purchase"] - df["Consumed"]
    df["Conservation_Diff"] = (df["Closing Balance"] - df["Calculated_Closing"]).abs()
    
    tolerance = 0.01 # ml
    invalid_mask = df["Conservation_Diff"] > tolerance
    invalid_count = invalid_mask.sum()
    validation_pct = ((len(df) - invalid_count) / len(df)) * 100.0
    
    print(f"[Conservation Validation] Tolerance: {tolerance} ml")
    print(f"[Conservation Validation] Valid Rows: {len(df) - invalid_count} / {len(df)} ({validation_pct:.2f}%)")
    print(f"[Conservation Validation] Discrepant Rows: {invalid_count}")
    
    # 3. Daily Aggregation
    # Aggregating daily consumption per Bar Name and Brand Name
    daily_agg = df.groupby(["Date", "Bar Name", "Brand Name"])["Consumed"].sum().reset_index()
    daily_agg.rename(columns={"Consumed": "Consumed (ml)"}, inplace=True)
    
    # 4. Complete Date Grid Creation (Cartesian Product)
    bars = daily_agg["Bar Name"].unique()
    brands = daily_agg["Brand Name"].unique()
    dates = pd.date_range(start=daily_agg["Date"].min(), end=daily_agg["Date"].max(), freq="D")
    
    full_index = pd.MultiIndex.from_product([dates, bars, brands], names=["Date", "Bar Name", "Brand Name"])
    daily_ts = daily_agg.set_index(["Date", "Bar Name", "Brand Name"]).reindex(full_index, fill_value=0.0).reset_index()
    
    daily_ts = daily_ts.sort_values(["Bar Name", "Brand Name", "Date"]).reset_index(drop=True)
    
    # Save processed dataset
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    daily_ts.to_csv(processed_path, index=False)
    print(f"[Processed Output] Saved complete date grid dataset ({len(daily_ts)} rows) to {processed_path}")
    
    return df, daily_ts

if __name__ == "__main__":
    process_inventory_data()
