import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode_survival_features():
    # Load survival dataset
    df = pd.read_excel("data/dataset_2_cleaning_work.xlsx")

    # Convert delay to numeric
    df["settlement_delay_minutes"] = pd.to_numeric(
        df["settlement_delay_minutes"], errors="coerce"
    )
    df = df.dropna(subset=["settlement_delay_minutes"])
    df = df[df["settlement_delay_minutes"] > 0]

    # Create event column
    df["event"] = 1

    # Separate survival variables
    y_time = df["settlement_delay_minutes"]
    y_event = df["event"]

    # Categorical and numeric features
    categorical_cols = [
        "corridor",
        "firm",
        "firm_type",
        "payment instrument",
        "access point",
        "receiving network coverage",
        "pickup method",
        "transparent"
    ]

    numeric_cols = [
        "cc1 fx margin",
        "cc1 lcu fee",
        "cc1 total cost %"
    ]

    X_cat = df[categorical_cols]
    X_num = df[numeric_cols]

    # One-Hot Encoding
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_cat_encoded = encoder.fit_transform(X_cat)

    # Combine encoded categorical + numeric
    X = pd.concat(
        [
            pd.DataFrame(X_cat_encoded),
            X_num.reset_index(drop=True)
        ],
        axis=1
    )

    print("Feature encoding completed")
    print("----------------------------------------")
    print("Feature matrix shape:", X.shape)
    print("Number of encoded features:", X_cat_encoded.shape[1])

    return X, y_time, y_event, encoder


if __name__ == "__main__":
    encode_survival_features()
