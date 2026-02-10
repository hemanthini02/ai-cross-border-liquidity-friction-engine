import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import os


def run_friction_ml_engine():

    print("\nLoading Phase-3 feature data...")
    df = pd.read_csv("phase3_friction_decomposition/phase3_features.csv")

    # -------------------------------------------------
    # 1. Feature selection for ML
    # -------------------------------------------------
    features = ["settlement_risk", "fx_risk"]
    X = df[features].copy()

    # -------------------------------------------------
    # 2. Scaling (important for ML)
    # -------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------------------------------
    # 3. TRUE ML – friction regime discovery
    # -------------------------------------------------
    print("\nRunning KMeans + GMM friction discovery...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    df["friction_cluster"] = kmeans.fit_predict(X_scaled)

    gmm = GaussianMixture(n_components=3, random_state=42)
    df["friction_regime_ml"] = gmm.fit_predict(X_scaled)
    df["regime_confidence"] = np.max(gmm.predict_proba(X_scaled), axis=1)

    # -------------------------------------------------
    # 4. Cluster financial profiling
    # -------------------------------------------------
    profile = df.groupby("friction_regime_ml")[features].mean()
    profile["gap"] = profile["fx_risk"] - profile["settlement_risk"]

    # Rank-based regime mapping
    sorted_clusters = profile.sort_values("gap")

    low = sorted_clusters.index[0]    # most settlement dominant
    mid = sorted_clusters.index[1]    # balanced
    high = sorted_clusters.index[2]   # most FX dominant

    regime_map = {
        low: "BANK_ROUTING",
        mid: "MIXED",
        high: "FX_DRIVEN"
    }

    df["friction_type_ml"] = df["friction_regime_ml"].map(regime_map)

    # -------------------------------------------------
    # 5. Save full ML output
    # -------------------------------------------------
    out_path = os.path.join(
        "phase3_friction_decomposition",
        "phase3_friction_ml_output.csv"
    )
    df.to_csv(out_path, index=False)

    # -------------------------------------------------
    # 6. Console summary
    # -------------------------------------------------
    print("\nPHASE-3 TRUE ML FRICTION ENGINE COMPLETE")
    print("--------------------------------------")
    print("\nFriction regime counts:")
    print(df["friction_type_ml"].value_counts())

    print("\nCluster financial profile:")
    print(profile)

    print("\nCluster → Regime mapping:")
    print(regime_map)

    # -------------------------------------------------
    # 7. RANDOM 20 TRANSACTIONS DISPLAY
    # -------------------------------------------------
    print("\nRANDOM 20 TRANSACTIONS WITH ML FRICTION TYPES")
    print("------------------------------------------------")

    sample_df = df.sample(20, random_state=42)[
        [
            "settlement_corridor",
            "fx_corridor",
            "settlement_risk",
            "fx_risk",
            "friction_type_ml",
            "regime_confidence"
        ]
    ]

    print(sample_df.to_string(index=False))

    # -------------------------------------------------
    # 8. Save random 20 transactions to separate CSV
    # -------------------------------------------------
    sample_out = os.path.join(
        "phase3_friction_decomposition",
        "phase3_random20_ml_samples.csv"
    )
    sample_df.to_csv(sample_out, index=False)

    print("\nRandom 20 transactions saved to:")
    print(sample_out)

    return df


if __name__ == "__main__":
    run_friction_ml_engine()
