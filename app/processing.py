
import pandas as pd
import numpy as np
import io
from typing import BinaryIO

def process_attendance(file_obj: BinaryIO) -> bytes:
    # STEP 1: LOAD DATA
    df = pd.read_csv(file_obj, skiprows=2)

    # STEP 2: CLEAN & STANDARDIZE COLUMNS
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

    # STEP 3: NORMALIZE STATUS
    df["attendance_status"] = (
        df["attendance_status"].astype(str).str.strip().str.upper()
    )

    # STEP 4: DATE HANDLING
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["date"] = df.groupby("employee_id")["date"].ffill()
    df["month"] = df["date"].dt.to_period("M")

    # STEP 5: SAFE TIME EXTRACTION
    def extract_time(val):
        if pd.isna(val):
            return pd.NaT
        try:
            return pd.to_datetime(val, errors="coerce").time()
        except Exception:
            return pd.NaT

    df["punch_in_time"] = df["punch_in_raw"].apply(extract_time)
    df["punch_out_time"] = df["punch_out_raw"].apply(extract_time)

    # STEP 6: BUILD REAL DATETIME
    df["punch_in"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["punch_in_time"].astype(str),
        errors="coerce"
    )
    df["punch_out"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["punch_out_time"].astype(str),
        errors="coerce"
    )
    df.loc[df["punch_out"] < df["punch_in"], "punch_out"] += pd.Timedelta(days=1)

    # STEP 7: WORKED HOURS
    df["working_hours"] = (df["punch_out"] - df["punch_in"]).dt.total_seconds() / 3600
    worked_hours_fallback = pd.to_timedelta(df["worked_duration"], errors="coerce").dt.total_seconds() / 3600
    df["working_hours"] = df["working_hours"].fillna(worked_hours_fallback)
    df["working_hours"] = df["working_hours"].replace([np.nan, np.inf, -np.inf], 0)

    # STEP 8: WORKING DAY FLAG
    df["working_day"] = df["attendance_status"] == "P"

    # STEP 9: OFFICE TIME RULES
    SHIFT_START = pd.to_datetime("10:00 AM").time()
    GRACE_LIMIT = pd.to_datetime("10:15 AM").time()

    def is_within_grace(punch, working_day):
        return bool(working_day and punch and SHIFT_START < punch <= GRACE_LIMIT)

    def is_late_beyond_grace(punch, working_day):
        return bool(working_day and punch and punch > GRACE_LIMIT)

    df["within_grace"] = df.apply(lambda r: is_within_grace(r["punch_in_time"], r["working_day"]), axis=1)
    df["late_beyond_grace"] = df.apply(lambda r: is_late_beyond_grace(r["punch_in_time"], r["working_day"]), axis=1)

    # STEP 10: GRACE COUNTS
    df["grace_count"] = df.groupby(["employee_id", "month"])["within_grace"].cumsum()
    df["grace_violation"] = df["grace_count"] > 4

    # STEP 11: DEDUCTION ENGINE (sequential logic)
    def calculate_deduction(row):
        if not row["working_day"]:
            return 0.0

        # Hours rule always applies
        if row["working_hours"] < 8:
            base_deduction = 1.0
        elif 8 <= row["working_hours"] < 9:
            base_deduction = 0.5
        else:
            base_deduction = 0.0

        # Grace violation adds stricter rules
        if row["late_beyond_grace"] and row["grace_violation"]:
            if row["working_hours"] < 9:
                return 1.0
            else:
                return max(base_deduction, 0.5)

        return base_deduction

    df["day_deduction"] = df.apply(calculate_deduction, axis=1)

    # STEP 11a: Average rule override (removed flex logic)
    # No flex privilege logic for now

    # STEP 12: PAYABLE DAY
    df["payable_day"] = 0.0
    df.loc[df["attendance_status"] == "WO", "payable_day"] = 1.0
    df.loc[df["attendance_status"] == "A", "payable_day"] = 0.0
    df.loc[df["attendance_status"] == "P", "payable_day"] = 1.0 - df["day_deduction"]
    df["payable_day"] = df["payable_day"].clip(0, 1)

    # STEP 13: CLEAN OUTPUT FORMAT
    df["date"] = df["date"].dt.date
    df["punch_in"] = df["punch_in"].fillna("").astype(str)
    df["punch_out"] = df["punch_out"].fillna("").astype(str)
    df.insert(0, "sr_no_fixed", range(1, len(df) + 1))

    def get_reason(row):
        reasons = []
        if row["day_deduction"] > 0:
            if row["late_beyond_grace"]:
                reasons.append("Late beyond grace")
            if row["working_hours"] < 8:
                reasons.append("Working hours < 8")
            elif 8 <= row["working_hours"] < 9:
                reasons.append("Working hours between 8–9")
            if row["grace_violation"]:
                reasons.append("Grace violation > 4")
        return ", ".join(reasons) if reasons else ""

    df["deduction_reason"] = df.apply(get_reason, axis=1)

    df["full_day"] = (df["day_deduction"] == 1.0).astype(float)
    df["half_day"] = (df["day_deduction"] == 0.5).astype(float)

    # STEP 14: EMPLOYEE SUMMARY
    deduction_rows = df[df['day_deduction'] > 0].copy()
    if deduction_rows.empty:
        summary_df = pd.DataFrame(columns=[
            'sr_no', 'employee_id', 'employee_name', 'designation',
            'deduction_dates',
            'total_full_day_deductions', 'total_half_day_deductions', 'total_deductions',
            'grace_violation_count', 'working_hours_less8_count',
            'late_beyond_grace_count', 'working_hours_between8to9_count'
        ])
    else:
        deduction_rows['date_formatted'] = pd.to_datetime(
            deduction_rows['date'], errors='coerce'
        ).dt.strftime('%d/%m/%Y')

        summary_df = (
            deduction_rows
            .groupby(['employee_id', 'employee_name', 'designation'])
            .agg({
                'date_formatted': lambda x: ', '.join(sorted(set(x.dropna()))),
                'full_day': 'sum',
                'half_day': 'sum',
                'day_deduction': 'sum',
                'grace_violation': 'sum',
                'working_hours': [
                    lambda x: (x < 8).sum(),              # count <8 hours
                    lambda x: ((x >= 8) & (x < 9)).sum()  # count between 8–9 hours
                ],
                'late_beyond_grace': 'sum'
            })
            .round(1)
            .reset_index()
        )

        # Flatten MultiIndex columns created by multiple lambdas
        summary_df.columns = [
            'employee_id', 'employee_name', 'designation',
            'deduction_dates',
            'total_full_day_deductions',
            'total_half_day_deductions',
            'total_deductions',
            'grace_violation_count',
            'working_hours_less8_count',
            'working_hours_between8to9_count',
            'late_beyond_grace_count'
        ]

        summary_df.insert(0, "sr_no", range(1, len(summary_df) + 1))

    # STEP 15: EXPORT TO EXCEL
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Full_Attendance', index=False)
        with_deductions = df[df['day_deduction'] > 0]
        if not with_deductions.empty:
            with_deductions.to_excel(writer, sheet_name='With_Deductions', index=False)
        summary_df.to_excel(writer, sheet_name='Employee_Summary', index=False)

    output.seek(0)
    return output.getvalue()

