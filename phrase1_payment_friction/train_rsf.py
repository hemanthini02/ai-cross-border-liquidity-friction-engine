from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from encode_features import encode_survival_features


def train_rsf_and_return():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        corridor_train,
        corridor_test
    ) = encode_survival_features()

    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    rsf.fit(X_train, y_train)

    preds = rsf.predict(X_test)

    c_index = concordance_index_censored(
        y_test["event"],
        y_test["time"],
        preds
    )[0]

    print("RSF trained successfully")
    print("C-index:", round(c_index, 4))

    return rsf, X_test, y_test, corridor_test


if __name__ == "__main__":
    train_rsf_and_return()
