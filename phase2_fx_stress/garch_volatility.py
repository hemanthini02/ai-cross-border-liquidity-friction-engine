import pandas as pd
import numpy as np
from arch import arch_model
from fx_features import build_fx_features


def compute_garch_volatility():
    df = build_fx_features()

    garch_results = []

    for corridor in df["corridor"].unique():
        df_c = df[df["corridor"] == corridor].copy()

        # GARCH needs returns in %
        returns = df_c["log_return"] * 100

        # GARCH(1,1) model
        model = arch_model(
            returns,
            vol="Garch",
            p=1,
            q=1,
            mean="Zero",
            rescale=False
        )

        res = model.fit(disp="off")

        # Conditional volatility
        df_c["garch_volatility"] = res.conditional_volatility

        garch_results.append(df_c)

        print(f"GARCH fitted for corridor: {corridor}")

    garch_df = pd.concat(garch_results).reset_index(drop=True)

    print("\nGARCH volatility sample:")
    print(garch_df[
        ["date", "corridor", "garch_volatility"]
    ].head())

    return garch_df


if __name__ == "__main__":
    compute_garch_volatility()
