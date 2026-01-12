# app/processing.py
import pandas as pd
import numpy as np
import tempfile
from typing import BinaryIO

def process_attendance(file_obj: BinaryIO) -> str:
    # ===============================
    # STEP 1: LOAD DATA
    # ===============================
    df = pd.read_csv(file_obj, skiprows=2)

    # ===============================
    # STEP 2: CLEAN & STANDARDIZE COLUMNS
    # ===============================
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r'[^\x00-\x7F]+', '', regex=True)
    )

    df = df.rename(columns={
        "SR+A3:R24 NO": "sr_no",
        "ALPHA EMP CODE": "employee_id",
        "EMP FULL NAME": "employee_name",
        "DESIG NAME": "designation",
        "ON DATE": "date",
        "SHIFT START TIME": "shift_start",
        "SHIFT END TIME": "shift_end",
        "ACTUAL IN TIME": "punch_in_raw",
        "ACTUAL OUT TIME": "punch_out_raw",
        "DURATION": "worked_duration",
        "AB LEAVE": "attendance_status"
    })

    # ===============================
    # STEP 3: NORMALIZE STATUS
    # ===============================
    df["attendance_status"] = (
        df["attendance_status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ===============================
    # STEP 4: DATE HANDLING
    # ===============================
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["date"] = df.groupby("employee_id")["date"].ffill()

    # ===============================
    # STEP 4a: CUSTOM MONTH CYCLE (25th→25th)
    # ===============================
    df['month_cycle'] = (df['date'] - pd.DateOffset(days=24)).dt.to_period('M')

    # ===============================
    # STEP 5: SAFE TIME EXTRACTION
    # ===============================
    def extract_time(val):
        if pd.isna(val):
            return pd.NaT
        try:
            return pd.to_datetime(val, errors="coerce").time()
        except Exception:
            return pd.NaT

    df["punch_in_time"] = df["punch_in_raw"].apply(extract_time)
    df["punch_out_time"] = df["punch_out_raw"].apply(extract_time)

    # ===============================
    # STEP 6: BUILD REAL DATETIME
    # ===============================
    df["punch_in"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["punch_in_time"].astype(str),
        errors="coerce"
    )
    df["punch_out"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["punch_out_time"].astype(str),
        errors="coerce"
    )

    # Overnight shift fix
    df.loc[df["punch_out"] < df["punch_in"], "punch_out"] += pd.Timedelta(days=1)

    # ===============================
    # STEP 7: WORKED HOURS
    # ===============================
    df["working_hours"] = (df["punch_out"] - df["punch_in"]).dt.total_seconds() / 3600

    worked_hours_fallback = pd.to_timedelta(df["worked_duration"], errors="coerce").dt.total_seconds() / 3600
    df["working_hours"] = df["working_hours"].fillna(worked_hours_fallback)
    df["working_hours"] = df["working_hours"].replace([np.nan, np.inf, -np.inf], 0)

    # ===============================
    # STEP 8: MONTH / WORKING DAY
    # ===============================
    df["month"] = df["date"].dt.to_period("M")
    df["working_day"] = df["attendance_status"] == "P"
    df["working_day"] = df["working_day"].astype(bool)

    # ===============================
    # STEP 9: OFFICE TIME RULES
    # ===============================
    SHIFT_START = pd.to_datetime("10:00 AM").time()
    GRACE_LIMIT = pd.to_datetime("10:15 AM").time()
    FLEX_LIMIT = pd.to_datetime("11:00 AM").time()

    def is_within_grace(punch, working_day):
        return bool(working_day and punch and SHIFT_START < punch <= GRACE_LIMIT)

    def is_late_beyond_grace(punch, working_day):
        return bool(working_day and punch and punch > GRACE_LIMIT)

    def is_flex_late(punch, working_day):
        return bool(working_day and punch and GRACE_LIMIT < punch <= FLEX_LIMIT)

    df["within_grace"] = df.apply(lambda row: is_within_grace(row["punch_in_time"], row["working_day"]), axis=1)
    df["late_beyond_grace"] = df.apply(lambda row: is_late_beyond_grace(row["punch_in_time"], row["working_day"]), axis=1)
    df["flex_late"] = df.apply(lambda row: is_flex_late(row["punch_in_time"], row["working_day"]), axis=1)

    # ===============================
    # STEP 10: GRACE & FLEX COUNTS
    # ===============================
    df["grace_count"] = df.groupby(["employee_id", "month_cycle"])["within_grace"].cumsum()
    df["grace_violation"] = df["grace_count"] > 4

    df["flex_count"] = df.groupby(["employee_id", "month_cycle"])["flex_late"].cumsum()
    df["flex_violation"] = df["flex_count"] > 5

    # ===============================
    # STEP 11: DEDUCTION ENGINE
    # ===============================
    df["half_day"] = 0.0
    df["full_day"] = 0.0

    mask_full = df["working_day"] & df["late_beyond_grace"] & (df["grace_count"] > 4) & (df["working_hours"] < 9)
    df.loc[mask_full, "full_day"] = 1.0

    mask_half = df["working_day"] & df["late_beyond_grace"] & (df["grace_count"] > 4) & (~df["flex_late"] | df["flex_violation"])
    df.loc[mask_half, "half_day"] = 0.5

    mask_full2 = df["working_day"] & (~df["late_beyond_grace"]) & (df["working_hours"] < 8)
    df.loc[mask_full2, "full_day"] = 1.0

    mask_half2 = df["working_day"] & (~df["late_beyond_grace"]) & (df["working_hours"] >= 8) & (df["working_hours"] < 9)
    df.loc[mask_half2, "half_day"] = 0.5

    df["day_deduction"] = df[["half_day", "full_day"]].max(axis=1)

    # ===============================
    # STEP 11a: APPLY AVERAGE WORK HOURS >9.5 FLEX RULE
    # ===============================
    def apply_average_rule(sub_df):
        avg_hours = sub_df.loc[sub_df['working_day'], 'working_hours'].mean()
        if avg_hours > 9.5:
            allowed = sub_df[(sub_df['flex_late'])].head(5).index
            sub_df.loc[allowed, ['day_deduction', 'half_day', 'full_day']] = 0
        return sub_df

    df = df.groupby('employee_id', group_keys=False).apply(apply_average_rule).reset_index(drop=True)

    # ===============================
    # STEP 12: PAYABLE DAY
    # ===============================
    df["payable_day"] = 0.0
    df.loc[df["attendance_status"] == "WO", "payable_day"] = 1.0
    df.loc[df["attendance_status"] == "A", "payable_day"] = 0.0
    df.loc[df["attendance_status"] == "P", "payable_day"] = 1.0 - df["day_deduction"]
    df["payable_day"] = df["payable_day"].clip(0, 1)

    # ===============================
    # STEP 13: CLEAN OUTPUT FORMAT
    # ===============================
    df["date"] = df["date"].dt.date
    df["punch_in"] = df["punch_in"].fillna("").astype(str)
    df["punch_out"] = df["punch_out"].fillna("").astype(str)

    df.insert(0, "sr_no_fixed", range(1, len(df) + 1))

    # ===============================
    # STEP 14: EXPORT BOTH SHEETS
    # ===============================
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    output_file = tmp.name

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Full_Attendance', index=False)
        df[df['day_deduction'] > 0].to_excel(writer, sheet_name='With_Deductions', index=False)

    return output_file