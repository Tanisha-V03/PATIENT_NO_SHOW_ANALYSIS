"""
patient_no_show_analysis.py
----------------------------
Patient No-Show Analysis: Data Cleaning, Data Manipulation, EDA & Visualization

Dataset: Modeled on the Kaggle "Medical Appointment No Shows" dataset
(patients from Vitoria, Brazil, 2016). Run generate_dataset.py first to
produce raw_patient_appointments.csv, or point RAW_FILE below at your own
downloaded KaggleV2-May-2016.csv.

Sections:
    1. Load data
    2. Data Cleaning
    3. Data Manipulation / Feature Engineering
    4. Exploratory Data Analysis (EDA)
    5. Data Visualization (saved to ./output_charts/)

Run:
    python patient_no_show_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless/server environments
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

RAW_FILE = r"C:\Users\hp\Documents\patient no show.csv"
OUTPUT_DIR = "output_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
NAVY = "#1F3B57"
ORANGE = "#E8622C"
PALETTE = [NAVY, ORANGE]


# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(r"C:\Users\hp\Documents\patient no show.csv")
    print(f"\n[LOAD] Loaded '{r"C:\Users\hp\Documents\patient no show.csv"}' -> {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ============================================================
# 2. DATA CLEANING
# ============================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("\n[CLEAN] Starting data cleaning...")

    # --- 2.1 Remove exact duplicate rows ---
    before = len(df)
    df = df.drop_duplicates()
    print(f"  - Removed {before - len(df)} duplicate rows")

    # --- 2.2 Standardize text fields (strip whitespace, fix casing) ---
    for col in ["Gender", "Neighbourhood", "No-show"]:
        df[col] = df[col].astype(str).str.strip()
    df["Gender"] = df["Gender"].str.upper()
    df["Neighbourhood"] = df["Neighbourhood"].str.upper()
    df["No-show"] = df["No-show"].str.upper()
    print("  - Standardized text casing/whitespace in Gender, Neighbourhood, No-show")

    # --- 2.3 Parse dates ---
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], errors="coerce")
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], errors="coerce")
    print("  - Parsed ScheduledDay / AppointmentDay to datetime")

    # --- 2.4 Handle invalid / missing Age ---
    invalid_age_mask = (df["Age"] < 0) | (df["Age"] > 115)
    print(f"  - Found {invalid_age_mask.sum()} invalid Age values (negative or >115); setting to NaN")
    df.loc[invalid_age_mask, "Age"] = np.nan

    missing_age = df["Age"].isna().sum()
    median_age = df["Age"].median()
    df["Age"] = df["Age"].fillna(median_age)
    print(f"  - Imputed {missing_age} missing Age values with the median ({median_age:.0f})")
    df["Age"] = df["Age"].astype(int)

    # --- 2.5 Handle missing Neighbourhood ---
    missing_neigh = (df["Neighbourhood"].isna() | (df["Neighbourhood"] == "NAN")).sum()
    df["Neighbourhood"] = df["Neighbourhood"].replace("NAN", np.nan)
    df["Neighbourhood"] = df["Neighbourhood"].fillna("UNKNOWN")
    print(f"  - Filled {missing_neigh} missing Neighbourhood values with 'UNKNOWN'")

    # --- 2.6 Drop rows with unusable dates (couldn't be parsed) ---
    before = len(df)
    df = df.dropna(subset=["ScheduledDay", "AppointmentDay"])
    print(f"  - Dropped {before - len(df)} rows with unparseable dates")

    # --- 2.7 Ensure binary/flag columns are proper integers ---
    binary_cols = ["Scholarship", "Hipertension", "Diabetes", "Alcoholism", "SMS_received", "Handcap"]
    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    print(f"  - Normalized dtype for binary/flag columns: {binary_cols}")

    print(f"[CLEAN] Done. Shape after cleaning: {df.shape}")
    return df


# ============================================================
# 3. DATA MANIPULATION / FEATURE ENGINEERING
# ============================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("\n[MANIPULATE] Engineering features...")

    # --- 3.1 Lead time in days (ScheduledDay -> AppointmentDay) ---
    df["WaitingDays"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days
    df["WaitingDays"] = df["WaitingDays"].clip(lower=0)  # negative values are data errors
    print("  - Created WaitingDays = AppointmentDay - ScheduledDay")

    # --- 3.2 Day of week appointment falls on ---
    df["DayOfWeek"] = df["AppointmentDay"].dt.day_name().str[:3]
    print("  - Extracted DayOfWeek from AppointmentDay")

    # --- 3.3 Age groups (bucketing a skewed continuous variable) ---
    bins = [-1, 17, 35, 60, 200]
    labels = ["0-17", "18-35", "36-60", "60+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
    print("  - Binned Age into AgeGroup: 0-17, 18-35, 36-60, 60+")

    # --- 3.4 Lead-time buckets (for trend analysis) ---
    lead_bins = [-1, 0, 2, 7, 14, 30, 60, 1000]
    lead_labels = ["Same day", "1-2d", "3-7d", "8-14d", "15-30d", "31-60d", "60+d"]
    df["LeadBucket"] = pd.cut(df["WaitingDays"], bins=lead_bins, labels=lead_labels)
    print("  - Binned WaitingDays into LeadBucket for trend analysis")

    # --- 3.5 Consolidated chronic-condition flag ---
    df["ChronicConditionFlag"] = (
        (df["Hipertension"] == 1) | (df["Diabetes"] == 1) | (df["Alcoholism"] == 1)
    ).astype(int)
    print("  - Created ChronicConditionFlag (Hipertension OR Diabetes OR Alcoholism)")

    # --- 3.6 Prior no-show count per patient (behavioral history feature) ---
    df = df.sort_values("ScheduledDay")
    df["NoShowBin"] = (df["No-show"] == "YES").astype(int)
    df["PriorNoShowCount"] = df.groupby("PatientId")["NoShowBin"].cumsum().shift(1).fillna(0).astype(int)
    print("  - Created PriorNoShowCount (cumulative prior no-shows per PatientId)")

    print(f"[MANIPULATE] Done. Shape after feature engineering: {df.shape}")
    return df


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
def run_eda(df: pd.DataFrame) -> None:
    print("\n[EDA] ==================== Summary Statistics ====================")
    print(df[["Age", "WaitingDays"]].describe().round(2))

    overall_rate = df["NoShowBin"].mean() * 100
    print(f"\nOverall no-show rate: {overall_rate:.2f}%")

    print("\nNo-show rate by Gender:")
    print((df.groupby("Gender")["NoShowBin"].mean() * 100).round(2))

    print("\nNo-show rate by AgeGroup:")
    print((df.groupby("AgeGroup", observed=True)["NoShowBin"].mean() * 100).round(2))

    print("\nNo-show rate by SMS_received (0 = no reminder, 1 = reminder sent):")
    print((df.groupby("SMS_received")["NoShowBin"].mean() * 100).round(2))

    print("\nNo-show rate by LeadBucket:")
    print((df.groupby("LeadBucket", observed=True)["NoShowBin"].mean() * 100).round(2))

    print("\nTop 5 neighbourhoods by no-show rate (min 30 appointments):")
    neigh_stats = df.groupby("Neighbourhood")["NoShowBin"].agg(["mean", "count"])
    neigh_stats = neigh_stats[neigh_stats["count"] >= 30].sort_values("mean", ascending=False)
    neigh_stats["mean"] = (neigh_stats["mean"] * 100).round(2)
    print(neigh_stats.head(5))

    print("\nCorrelation matrix (numeric/binary fields vs. NoShowBin):")
    corr_cols = ["Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism",
                 "SMS_received", "WaitingDays", "PriorNoShowCount", "NoShowBin"]
    print(df[corr_cols].corr().round(2)["NoShowBin"].sort_values(ascending=False))
    print("[EDA] =============================================================")


# ============================================================
# 5. DATA VISUALIZATION
# ============================================================
def make_visualizations(df: pd.DataFrame) -> None:
    print(f"\n[VISUALIZE] Saving charts to ./{OUTPUT_DIR}/ ...")

    # 5.1 Age distribution
    plt.figure(figsize=(7, 4.2))
    sns.histplot(df["Age"], bins=30, color=NAVY, kde=True)
    plt.title("Age Distribution of Patients")
    plt.xlabel("Age (years)")
    plt.ylabel("Number of Appointments")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_age_distribution.png", dpi=150)
    plt.close()

    # 5.2 No-show rate by day of week
    plt.figure(figsize=(7, 4.2))
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rate = (df.groupby("DayOfWeek")["NoShowBin"].mean() * 100).reindex(order).dropna()
    sns.barplot(x=rate.index, y=rate.values, color=ORANGE)
    plt.title("No-Show Rate by Day of Appointment")
    plt.xlabel("Day of Week")
    plt.ylabel("No-Show Rate (%)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_noshow_by_day.png", dpi=150)
    plt.close()

    # 5.3 Waiting days boxplot by outcome
    plt.figure(figsize=(7, 4.2))
    sns.boxplot(x="No-show", y="WaitingDays", data=df, hue="No-show",
                palette=PALETTE, legend=False)
    plt.title("Lead Time (Waiting Days) vs. Appointment Outcome")
    plt.xlabel("No-show")
    plt.ylabel("Waiting Days")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_waitingdays_boxplot.png", dpi=150)
    plt.close()

    # 5.4 Correlation heatmap
    plt.figure(figsize=(6.5, 5.5))
    corr_cols = ["Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism",
                 "SMS_received", "WaitingDays", "PriorNoShowCount", "NoShowBin"]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-0.4, vmax=0.4, cbar_kws={"label": "Correlation"})
    plt.title("Correlation Matrix of Key Variables")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_correlation_heatmap.png", dpi=150)
    plt.close()

    # 5.5 Scatter: Age vs waiting days by outcome
    plt.figure(figsize=(7, 4.5))
    sample = df.sample(min(1000, len(df)), random_state=1)
    sns.scatterplot(data=sample, x="Age", y="WaitingDays", hue="No-show",
                     palette=PALETTE, alpha=0.6, s=28)
    plt.title("Age vs. Waiting Days, by Appointment Outcome")
    plt.xlabel("Age (years)")
    plt.ylabel("Waiting Days")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_scatter_age_waitingdays.png", dpi=150)
    plt.close()

    # 5.6 SMS reminder effect
    plt.figure(figsize=(7, 4.2))
    sms_rate = df.groupby("SMS_received")["NoShowBin"].mean() * 100
    sns.barplot(x=["No SMS Reminder", "SMS Reminder Sent"], y=sms_rate.values, color=NAVY)
    plt.title("Effect of SMS Reminders on No-Show Rate")
    plt.ylabel("No-Show Rate (%)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_sms_effect.png", dpi=150)
    plt.close()

    # 5.7 Trend line: no-show rate across lead-time buckets
    plt.figure(figsize=(7.5, 4.2))
    order = ["Same day", "1-2d", "3-7d", "8-14d", "15-30d", "31-60d", "60+d"]
    trend = (df.groupby("LeadBucket", observed=True)["NoShowBin"].mean() * 100).reindex(order).dropna()
    plt.plot(trend.index, trend.values, marker="o", color=ORANGE, linewidth=2)
    plt.title("No-Show Rate Trend Across Scheduling Lead Time")
    plt.xlabel("Lead Time")
    plt.ylabel("No-Show Rate (%)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_leadtime_trend.png", dpi=150)
    plt.close()

    # 5.8 No-show rate by neighbourhood (top locations by volume)
    plt.figure(figsize=(7.5, 4.5))
    neigh_stats = df.groupby("Neighbourhood")["NoShowBin"].agg(["mean", "count"])
    neigh_stats = neigh_stats[neigh_stats["count"] >= 30].sort_values("mean", ascending=False).head(8)
    sns.barplot(x=(neigh_stats["mean"] * 100).values, y=neigh_stats.index, color=NAVY)
    plt.title("No-Show Rate by Neighbourhood (Top Locations)")
    plt.xlabel("No-Show Rate (%)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_neighbourhood_noshow_rate.png", dpi=150)
    plt.close()

    print(f"[VISUALIZE] Saved 8 charts to ./{OUTPUT_DIR}/")


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(
            f"'{RAW_FILE}' not found. Run generate_dataset.py first, "
            f"or place your own KaggleV2-May-2016.csv here and update RAW_FILE."
        )

    df_raw = load_data(RAW_FILE)
    df_clean = clean_data(df_raw)
    df_final = engineer_features(df_clean)
    run_eda(df_final)
    make_visualizations(df_final)

    df_final.to_csv("cleaned_patient_appointments.csv", index=False)
    print("\n[DONE] Cleaned & feature-engineered dataset saved to 'cleaned_patient_appointments.csv'")


if __name__ == "__main__":
    main()