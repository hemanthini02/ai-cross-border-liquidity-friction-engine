import pandas as pd
import numpy as np
from garch_volatility import compute_garch_volatility


def classify_fx_regimes():
    """
    PHASE-2B: FX STRESS REGIME CLASSIFICATION (FIXED)

    Uses percentile-based regimes per corridor
    instead of raw volatility clustering.
    """

    df = compute_garch_volatility()

    # -------------------------------
    # Compute volatility percentile PER corridor
    # -------------------------------
    df["vol_percentile"] = (
        df.groupby("corridor")["garch_volatility"]
        .rank(pct=True)
    )

    # -------------------------------
    # Define FX stress regimes
    # -------------------------------
    def assign_regime(p):
        if p <= 0.30:
            return "LOW"
        elif p <= 0.70:
            return "MEDIUM"
        else:
            return "HIGH"

    df["fx_stress_regime"] = df["vol_percentile"].apply(assign_regime)

    # Numeric encoding (used downstream)
    regime_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    df["fx_regime_num"] = df["fx_stress_regime"].map(regime_map)

    # -------------------------------
    # Debug checks (IMPORTANT)
    # -------------------------------
    print("\nFX REGIME DISTRIBUTION (PER CORRIDOR)")
    print("------------------------------------")
    print(
        df.groupby(["corridor", "fx_stress_regime"])
          .size()
          .unstack(fill_value=0)
    )

    print("\nFX Stress Regime Sample:")
    print(
        df[
            ["date", "corridor", "garch_volatility",
             "vol_percentile", "fx_stress_regime"]
        ].head(10)
    )

    return df


if __name__ == "__main__":
    classify_fx_regimes()
