import pandas as pd
import numpy as np
from train_rsf import train_rsf_and_return


def generate_phase1_outputs():
    rsf, X_test, y_test, corridor_test = train_rsf_and_return()

    surv_funcs = rsf.predict_survival_function(X_test)

    outputs = []

    for i, fn in enumerate(surv_funcs):
        times = fn.x
        surv_probs = fn.y

        expected_delay = np.trapz(surv_probs, times)

        p_gt_1day = surv_probs[times >= 1440][0] if any(times >= 1440) else 0
        p_gt_2day = surv_probs[times >= 2880][0] if any(times >= 2880) else 0

        outputs.append({
            "corridor": corridor_test.iloc[i],
            "expected_delay_minutes": expected_delay,
            "p_delay_gt_1day": p_gt_1day,
            "p_delay_gt_2day": p_gt_2day
        })

    df_out = pd.DataFrame(outputs)

    print("\nPHASE-1 OUTPUT SUMMARY")
    print(df_out.describe())

    return df_out


if __name__ == "__main__":
    generate_phase1_outputs()
