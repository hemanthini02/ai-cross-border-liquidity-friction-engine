import pandas as pd
from fx_regime_classifier import classify_fx_regimes

def debug_fx_regimes():
    # Load fully processed FX regime data
    df = classify_fx_regimes()

    print("\n==============================")
    print("DATASET SHAPE")
    print("==============================")
    print(df.shape)

    print("\n==============================")
    print("UNIQUE CORRIDORS")
    print("==============================")
    print(df["corridor"].unique())

    print("\n==============================")
    print("FX REGIME COUNTS (GLOBAL)")
    print("==============================")
    print(df["fx_stress_regime"].value_counts())

    print("\n==============================")
    print("FX REGIME DISTRIBUTION PER CORRIDOR")
    print("==============================")
    print(
        df.groupby("corridor")["fx_stress_regime"]
          .value_counts()
          .unstack()
          .fillna(0)
    )

    print("\n==============================")
    print("SAMPLE: HIGH STRESS ROWS")
    print("==============================")
    print(
        df[df["fx_stress_regime"] == "HIGH"]
        [["date", "corridor", "garch_volatility"]]
        .head(10)
    )

    print("\n==============================")
    print("RANDOM SAMPLE (FULL VARIABILITY CHECK)")
    print("==============================")
    print(
        df.sample(10)
        [["date", "corridor", "fx_stress_regime", "garch_volatility"]]
    )

if __name__ == "__main__":
    debug_fx_regimes()
