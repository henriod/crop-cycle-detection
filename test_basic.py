"""
Basic smoke test — runs crop_cycle_model on the bundled sample data
and verifies that at least one season is detected.

Usage:
    python test_basic.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from src.crop_cycle import crop_cycle_model


def test_sample_data():
    csv_path = os.path.join(os.path.dirname(__file__), "sample_data", "example_ndvi.csv")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    assert not df.empty, "Sample CSV is empty"

    report, ndvi_processed = crop_cycle_model(df)

    assert not report.empty, (
        "crop_cycle_model returned an empty report — "
        "expected at least one season in the sample data."
    )
    assert "ndvi_peak_date" in report.columns
    assert "approx_season_start" in report.columns
    assert "approx_season_end" in report.columns
    assert not ndvi_processed.empty

    print(f"PASS — {len(report)} season(s) detected.")
    print(report[["season_number", "approx_season_start", "approx_season_end", "ndvi_peak_date"]].to_string(index=False))


def test_weak_season_not_dropped():
    # Bimodal field in western Kenya: the 2025-26 short rains peaked at ~0.66,
    # well below the other seasons (~0.75). It must still be detected.
    csv_path = os.path.join(os.path.dirname(__file__), "sample_data", "bimodal_kenya_ndvi.csv")
    df = pd.read_csv(csv_path, parse_dates=["date"])

    report, _ = crop_cycle_model(df)

    assert len(report) == 7, f"expected 7 seasons, got {len(report)}"
    weak = report[report["ndvi_peak_date"] == "2025-12-30"]
    assert len(weak) == 1, "2025-26 short-rains season missing"
    assert weak["approx_season_start"].iloc[0] < pd.Timestamp("2025-11-15"), (
        "2025-26 season split by a small wobble; start too late"
    )
    print(f"PASS — weak season detected ({len(report)} seasons).")


def test_imports():
    from src.crop_cycle import crop_cycle_model  # noqa: F401
    from src.ndvi_preprocessing import whittaker_smooth, local_regression  # noqa: F401
    print("PASS — all imports resolved.")


if __name__ == "__main__":
    print("Running import test ...")
    test_imports()
    print("\nRunning sample data test ...")
    test_sample_data()
    print("\nRunning weak season test ...")
    test_weak_season_not_dropped()
    print("\nAll tests passed.")
