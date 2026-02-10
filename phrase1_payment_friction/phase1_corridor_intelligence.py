import pandas as pd
import numpy as np
import os
from train_rsf import train_rsf_and_return

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def build_corridor_intelligence():
    # Train RSF and get test data
    rsf, X_test, y_test, corridor_test = train_rsf_and_return()
    surv_funcs = rsf.predict_survival_function(X_test)

    records = []

    for i, fn in enumerate(surv_funcs):
        times = fn.x
        surv_probs = fn.y

        def percentile_delay(p):
            idx = np.where(surv_probs <= (1 - p))[0]
            return times[idx[0]] if len(idx) > 0 else times[-1]

        records.append({
            "corridor": corridor_test.iloc[i],
            "p50_delay_min": percentile_delay(0.50),
            "p75_delay_min": percentile_delay(0.75),
            "p90_delay_min": percentile_delay(0.90),
            "p_delay_gt_1day": surv_probs[times >= 1440][0] if any(times >= 1440) else 0,
            "p_delay_gt_2day": surv_probs[times >= 2880][0] if any(times >= 2880) else 0
        })

    df = pd.DataFrame(records)

    # Corridor-level aggregation
    corridor_summary = df.groupby("corridor").agg(
        p50_delay_min=("p50_delay_min", "median"),
        p75_delay_min=("p75_delay_min", "median"),
        p90_delay_min=("p90_delay_min", "median"),
        avg_p_delay_gt_1day=("p_delay_gt_1day", "mean"),
        avg_p_delay_gt_2day=("p_delay_gt_2day", "mean")
    ).reset_index()

    # Convert minutes to hours
    corridor_summary["p50_delay_hrs"] = corridor_summary["p50_delay_min"] / 60
    corridor_summary["p75_delay_hrs"] = corridor_summary["p75_delay_min"] / 60
    corridor_summary["p90_delay_hrs"] = corridor_summary["p90_delay_min"] / 60

    # Congestion regime classification
    def regime(p90):
        if p90 <= 24:
            return "LOW"
        elif p90 <= 48:
            return "MEDIUM"
        else:
            return "HIGH"

    corridor_summary["congestion_regime"] = corridor_summary["p90_delay_hrs"].apply(regime)

    # Settlement risk score (0–100)
    corridor_summary["settlement_risk_score"] = (
        corridor_summary["avg_p_delay_gt_1day"] * 50 +
        corridor_summary["avg_p_delay_gt_2day"] * 50
    ).round(0)

    print("\nPHASE-1 FINAL CORRIDOR INTELLIGENCE")
    print("----------------------------------")
    print(corridor_summary)

    # Ensure output directory exists
    output_dir = "phrase1_payment_friction"
    os.makedirs(output_dir, exist_ok=True)

    # Save Phase-1 output for downstream phases
    output_path = os.path.join(
        output_dir, "phase1_corridor_intelligence.csv"
    )
    corridor_summary.to_csv(output_path, index=False)

    print("\nPhase-1 output saved to CSV:")
    print(output_path)

    return corridor_summary


if __name__ == "__main__":
    build_corridor_intelligence()
