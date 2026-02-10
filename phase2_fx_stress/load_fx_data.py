import pandas as pd
import numpy as np


def load_fx_data():
    df = pd.read_csv("data/all_corridors_train_2016.csv")

    print("Raw FX dataset loaded")
    print("----------------------------------")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Rename DATE column safely
    df = df.rename(columns={"DATE": "date"})

    # Drop unnamed junk columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Melt wide FX data into long format
    df_long = df.melt(
        id_vars=["date"],
        var_name="corridor",
        value_name="rate"
    )

    # Convert date
    df_long["date"] = pd.to_datetime(df_long["date"])

    # Sort
    df_long = df_long.sort_values(["corridor", "date"])

    # Calculate log returns
    df_long["log_return"] = (
        np.log(df_long["rate"]) -
        np.log(df_long.groupby("corridor")["rate"].shift(1))
    )

    df_long = df_long.dropna().reset_index(drop=True)

    print("\nFX dataset after reshaping + returns:")
    print(df_long.head())

    return df_long


if __name__ == "__main__":
    load_fx_data()
