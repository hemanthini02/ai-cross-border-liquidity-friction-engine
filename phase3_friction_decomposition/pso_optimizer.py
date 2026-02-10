import pandas as pd
import numpy as np
import os

# -------------------------------
# Particle for constrained PSO
# -------------------------------
class Particle:
    def __init__(self):
        self.position = np.random.uniform(0.2, 0.8, 2)
        self.position = self.position / self.position.sum()
        self.velocity = np.random.uniform(-0.1, 0.1, 2)
        self.best_position = self.position.copy()
        self.best_error = float("inf")


def objective_function(weights, settlement_risk, fx_risk, target_delay):
    fused = weights[0] * settlement_risk + weights[1] * fx_risk
    return np.mean((fused - target_delay) ** 2)


def run_pso(settlement_risk, fx_risk, target_delay,
            n_particles=30, n_iterations=50):

    swarm = [Particle() for _ in range(n_particles)]
    global_best_position = None
    global_best_error = float("inf")

    for _ in range(n_iterations):
        for p in swarm:
            error = objective_function(
                p.position,
                settlement_risk,
                fx_risk,
                target_delay
            )

            if error < p.best_error:
                p.best_error = error
                p.best_position = p.position.copy()

            if error < global_best_error:
                global_best_error = error
                global_best_position = p.position.copy()

        for p in swarm:
            inertia = 0.5
            cognitive = 1.5 * np.random.rand() * (p.best_position - p.position)
            social = 1.5 * np.random.rand() * (global_best_position - p.position)

            p.velocity = inertia * p.velocity + cognitive + social
            p.position += p.velocity

            # ---- CONSTRAINTS ----
            p.position = np.clip(p.position, 0.2, None)
            p.position = p.position / p.position.sum()

    return global_best_position


def optimize_phase3_weights():

    df = pd.read_csv(
        "phase3_friction_decomposition/phase3_features.csv"
    )

    results = []

    for corridor, grp in df.groupby("settlement_corridor"):
        weights = run_pso(
            grp["settlement_risk"].values,
            grp["fx_risk"].values,
            grp["extreme_delay_min"].values
        )

        results.append({
            "corridor": corridor,
            "settlement_weight": round(weights[0], 3),
            "fx_weight": round(weights[1], 3)
        })

    out = pd.DataFrame(results)

    path = "phase3_friction_decomposition/pso_optimized_weights.csv"
    out.to_csv(path, index=False)

    print("\nCONSTRAINED PSO OPTIMIZED WEIGHTS")
    print("--------------------------------")
    print(out)

    return out


if __name__ == "__main__":
    optimize_phase3_weights()
