import pandas as pd
import numpy as np
import os

from fx_regime_classifier import classify_fx_regimes


def compute_fx_risk_score():
    # ----------------------------------
    # Step 1: Get full FX dataframe
    # ----------------------------------
    df = classify_fx_regimes()

    # ----------------------------------
    # Step 2: Volatility percentile (per corridor)
    # ----------------------------------
    df["vol_percentile"] = (
        df.groupby("corridor")["garch_volatility"]
        .rank(pct=True)
    )

    # ----------------------------------
    # Step 3: FX slippage (bps)
    # ----------------------------------
    df["fx_slippage_bps"] = df["vol_percentile"] * 30

    # ----------------------------------
    # Step 4: FX risk score (0–100)
    # ----------------------------------
    df["fx_risk_score"] = (df["vol_percentile"] * 100).round(2)

    # ----------------------------------
    # Step 5: SAVE FULL ROW-LEVEL CSV ✅
    # ----------------------------------
    output_dir = "phase2_fx_stress"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "phase2_fx_transaction_level.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nPHASE-2 TRANSACTION LEVEL FX OUTPUT SAVED")
    print("-----------------------------------------")
    print("Shape:", df.shape)
    print("\nSample rows:")
    print(
        df[
            [
                "date",
                "corridor",
                "garch_volatility",
                "rolling_vol_20",
                "max_shock_5",
                "fx_stress_regime",
                "fx_risk_score",
                "fx_slippage_bps",
            ]
        ].sample(10)
    )

    return df


if __name__ == "__main__":
    compute_fx_risk_score()
