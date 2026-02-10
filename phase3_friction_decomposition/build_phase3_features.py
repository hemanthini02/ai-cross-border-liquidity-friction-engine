import pandas as pd
import os


def build_phase3_features():

    # -------------------------------
    # Load Phase-1 (corridor intelligence)
    # -------------------------------
    phase1 = pd.read_csv(
        "phrase1_payment_friction/phase1_corridor_intelligence.csv"
    )

    phase1 = phase1[
        ["corridor", "settlement_risk_score", "p50_delay_min", "p90_delay_min"]
    ]

    # -------------------------------
    # Load Phase-2 (FX transaction-level output)
    # -------------------------------
    phase2 = pd.read_csv(
        "phase2_fx_stress/phase2_fx_transaction_level.csv"
    )

    phase2 = phase2[
        [
            "date",
            "corridor",
            "fx_risk_score",
            "fx_slippage_bps",
            "fx_stress_regime"
        ]
    ]

    # Encode FX regime
    fx_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    phase2["fx_regime_num"] = phase2["fx_stress_regime"].map(fx_map)

    # Map settlement corridor → FX corridor
    corridor_fx_map = {
        "AREIND": "INR_AED",
        "DEUIND": "USD_EUR",
        "GBRIND": "INR_GBP",
        "USAIND": "USD_INR"
    }

    phase1["fx_corridor"] = phase1["corridor"].map(corridor_fx_map)

    # -------------------------------
    # Merge
    # -------------------------------
    df = phase2.merge(
        phase1,
        left_on="corridor",
        right_on="fx_corridor",
        how="inner"
    )

    df.rename(columns={
        "corridor_x": "fx_corridor",
        "corridor_y": "settlement_corridor",
        "settlement_risk_score": "settlement_risk",
        "fx_risk_score": "fx_risk",
        "p90_delay_min": "extreme_delay_min"
    }, inplace=True)

    df = df[
        [
            "date",
            "settlement_corridor",
            "fx_corridor",
            "settlement_risk",
            "p50_delay_min",
            "extreme_delay_min",
            "fx_risk",
            "fx_slippage_bps",
            "fx_regime_num"
        ]
    ]

    os.makedirs("phase3_friction_decomposition", exist_ok=True)
    out = "phase3_friction_decomposition/phase3_features.csv"
    df.to_csv(out, index=False)

    print("Phase-3 features created")
    print("Shape:", df.shape)

    return df


if __name__ == "__main__":
    build_phase3_features()
