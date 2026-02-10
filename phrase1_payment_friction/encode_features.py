import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from survival_format import create_survival_dataset


def encode_survival_features():
    df = create_survival_dataset()

    # Survival target
    y = np.array(
        list(zip(df["event"].astype(bool), df["settlement_delay_minutes"])),
        dtype=[("event", bool), ("time", float)]
    )

    corridor = df["corridor"].reset_index(drop=True)

    X = df.drop(
        columns=["settlement_delay_minutes", "event", "corridor"]
    )

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = encoder.fit_transform(X[cat_cols])
    X_num = X[num_cols].values

    feature_names = (
        list(num_cols) +
        list(encoder.get_feature_names_out(cat_cols))
    )

    X_final = pd.DataFrame(
        np.hstack([X_num, X_cat]),
        columns=feature_names
    )

    # SAVE encoder for future Digital Twin use
    joblib.dump(encoder, "models/phase1_encoder.pkl")

    X_train, X_test, y_train, y_test, corridor_train, corridor_test = train_test_split(
        X_final, y, corridor, test_size=0.2, random_state=42
    )

    print("Phase-1 feature encoding complete")
    print("Shape:", X_final.shape)

    return X_train, X_test, y_train, y_test, corridor_train, corridor_test


if __name__ == "__main__":
    encode_survival_features()
