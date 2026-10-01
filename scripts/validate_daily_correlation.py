"""Recompute a descriptive daily correlation from the tracked raw CSV snapshot."""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr


def main():
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    counts = []
    total_rows = 0
    valid_2024_rows = 0
    for path in sorted(raw.glob("nypd_*.csv")):
        for chunk in pd.read_csv(
            path, usecols=["cmplnt_fr_dt", "cmplnt_fr_tm"], chunksize=100000
        ):
            total_rows += len(chunk)
            dates = pd.to_datetime(
                chunk["cmplnt_fr_dt"].astype(str) + " " + chunk["cmplnt_fr_tm"].astype(str),
                errors="coerce", format="mixed",
            ).dt.normalize()
            dates = dates[dates.dt.year == 2024]
            valid_2024_rows += len(dates)
            counts.append(dates.value_counts())
    if not counts:
        raise ValueError("No NYPD CSV parts were found")
    daily = pd.concat(counts, axis=1).fillna(0).sum(axis=1).rename("crime_count")
    daily.index.name = "date"
    weather = pd.read_csv(raw / "noaa_ghcnd_2024.csv", usecols=["DATE", "TMAX", "TMIN"])
    weather["date"] = pd.to_datetime(weather["DATE"], errors="coerce")
    weather["temp_avg"] = (weather["TMAX"] + weather["TMIN"]) / 2
    weather = weather[weather["date"].dt.year == 2024].dropna(subset=["date", "temp_avg"])
    merged = daily.reset_index().merge(
        weather[["date", "temp_avg"]], on="date", validate="one_to_one"
    )
    r, p = pearsonr(merged["crime_count"], merged["temp_avg"])
    print(json.dumps({
        "source": "tracked raw CSV snapshot",
        "year": 2024,
        "raw_crime_rows": total_rows,
        "valid_crime_rows_in_2024": valid_2024_rows,
        "matched_days": len(merged),
        "pearson_r": float(r),
        "naive_independent_observations_p_value": float(p),
        "limitations": [
            "Complaint rows are counted; this is not a measure of all actual crime.",
            "Temperature is the midpoint of daily maximum/minimum, not a measured daily mean.",
            "The nominal p-value assumes independent observations and does not adjust for seasonality or temporal autocorrelation.",
            "This is descriptive correlation; no causal or policing recommendation follows."
        ]
    }, indent=2))


if __name__ == "__main__":
    main()
