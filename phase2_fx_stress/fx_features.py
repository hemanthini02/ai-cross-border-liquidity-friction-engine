import pandas as pd
import numpy as np
from load_fx_data import load_fx_data


def build_fx_features():
    df = load_fx_data()

    # Rolling volatility (simple, before GARCH)
    df["rolling_vol_5"] = df.groupby("corridor")["log_return"].rolling(5).std().reset_index(0, drop=True)
    df["rolling_vol_20"] = df.groupby("corridor")["log_return"].rolling(20).std().reset_index(0, drop=True)

    # Absolute returns (shock indicator)
    df["abs_return"] = df["log_return"].abs()

    # Rolling max shock
    df["max_shock_5"] = df.groupby("corridor")["abs_return"].rolling(5).max().reset_index(0, drop=True)

    df = df.dropna().reset_index(drop=True)

    print("FX feature dataset created")
    print("----------------------------------")
    print(df.head())

    return df


if __name__ == "__main__":
    build_fx_features()
