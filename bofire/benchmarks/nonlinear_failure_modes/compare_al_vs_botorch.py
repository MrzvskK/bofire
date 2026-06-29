"""Head-to-head: Augmented Lagrangian vs BoTorch thin-band for circle equality.

Compares:
  - Method A (current): optimize_acqf with ±eq_tol slab encoding
  - Method B (proposed): augmented_lagrangian.optimize_acqf_al

Both methods are given the same acquisition function (fitted on the same seed
experiments) and the same n_restarts budget.

Metrics:
  - Wall-clock time for a single ask()
  - Final |f(x*)| (constraint violation)
  - Final acqf value
  - Whether the candidate passes BoFire validation

Run:
    python -m bofire.benchmarks.nonlinear_failure_modes.compare_al_vs_botorch
"""

from __future__ import annotations

import time

import torch

from bofire.benchmarks.nonlinear_failure_modes.augmented_lagrangian import (
    optimize_acqf_al,
)
from bofire.benchmarks.nonlinear_failure_modes.scenarios import (
    fm3_circle_equality_string,
)
from bofire.data_models.constraints.api import NonlinearEqualityConstraint
from bofire.data_models.strategies.api import SoboStrategy as SoboDataModel
from bofire.strategies.api import SoboStrategy
from bofire.strategies.predictives.acqf_optimization import AcquisitionOptimizer
from bofire.utils.torch_tools import (
    _nonlinear_constraint_feature_indices,
    evaluate_nonlinear_constraint_on_tensor,
    get_torch_bounds_from_domain,
)


SEED = 42
N_INITIAL = 8
N_RESTARTS = 5  # keep same for fair comparison


def setup_scenario():
    """Return (domain, seed_df) for the circle equality scenario."""
    spec = fm3_circle_equality_string(dim=2, radius=0.5)
    domain = spec.make_domain()
    seed_df = spec.make_seed_experiments(domain, n=N_INITIAL, seed=SEED)
    return domain, seed_df


def build_strategy_with_model(domain, seed_df):
    """Build a SoboStrategy and fit the surrogate on seed_df."""
    data_model = SoboDataModel(domain=domain)
    strategy = SoboStrategy(data_model=data_model)
    strategy.tell(seed_df)
    return strategy


def compute_constraint_violation(x_flat, domain):
    """Return max |f_i(x)| over all equality constraints."""
    eq_constraints = domain.constraints.get(NonlinearEqualityConstraint)
    max_viol = 0.0
    for c in eq_constraints:
        feat_idx = _nonlinear_constraint_feature_indices(c, domain)
        x_2d = x_flat.reshape(1, -1)
        val = evaluate_nonlinear_constraint_on_tensor(c, x_2d, feat_idx)
        max_viol = max(max_viol, abs(val.item()))
    return max_viol


def method_a_botorch(strategy, domain, n_restarts):
    """Method A: Standard BoFire ask() — uses BoTorch ±eq_tol slab encoding."""
    print("\n--- Method A: BoTorch (current approach) ---")
    print(f"  n_restarts={n_restarts}  (BoTorch default via strategy)")
    t0 = time.time()
    try:
        candidates = strategy.ask(candidate_count=1)
        elapsed = time.time() - t0
        print(f"  ok=True  elapsed={elapsed:.2f}s")
        x_vals = candidates[domain.inputs.get_keys()].values[0]
        print(f"  candidates: {x_vals}")
        x_tensor = torch.tensor(x_vals, dtype=torch.float64)
        viol = compute_constraint_violation(x_tensor, domain)
        print(f"  |f(x*)|={viol:.2e}")
        # Evaluate the actual acquisition value for this candidate
        acqfs_for_eval = strategy._get_acqfs(n=1)
        input_preprocessing_specs = AcquisitionOptimizer._input_preprocessing_specs(
            domain
        )
        bounds_eval = get_torch_bounds_from_domain(domain, input_preprocessing_specs)
        n_dims = bounds_eval.shape[-1]
        with torch.no_grad():
            acqf_val = acqfs_for_eval[0](x_tensor.reshape(1, 1, n_dims)).item()
        print(f"  acqf_val={acqf_val:.4f}")
        try:
            domain.validate_candidates(
                candidates[domain.inputs.get_keys()],
                only_inputs=True,
                raise_validation_error=True,
            )
            valid = True
        except Exception as e:
            valid = False
            print(f"  validation FAILED: {e}")
        print(f"  BoFire validation: {'PASS' if valid else 'FAIL'}")
        return {
            "ok": True,
            "elapsed": elapsed,
            "violation": viol,
            "valid": valid,
            "acqf_val": acqf_val,
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ok=False  elapsed={elapsed:.2f}s  error={type(e).__name__}: {e}")
        return {"ok": False, "elapsed": elapsed, "violation": None, "valid": False}


def method_b_al(strategy, domain, n_restarts):
    """Method B: Augmented Lagrangian on top of the same fitted acquisition function."""
    print("\n--- Method B: Augmented Lagrangian ---")
    print(f"  n_restarts={n_restarts}")

    # Obtain the fitted acquisition function from the strategy
    acqfs = strategy._get_acqfs(n=1)
    input_preprocessing_specs = AcquisitionOptimizer._input_preprocessing_specs(domain)
    bounds = get_torch_bounds_from_domain(domain, input_preprocessing_specs)

    t0 = time.time()
    try:
        candidates, acqf_val = optimize_acqf_al(
            acqf=acqfs[0],
            bounds=bounds,
            domain=domain,
            n_restarts=n_restarts,
            q=1,
            n_outer=8,
            n_inner=100,
            verbose=True,
        )
        elapsed = time.time() - t0
        print(f"  ok=True  elapsed={elapsed:.2f}s  acqf_val={acqf_val.item():.4f}")
        x_flat = candidates.reshape(-1)
        print(f"  candidates: {x_flat.tolist()}")
        viol = compute_constraint_violation(x_flat, domain)
        print(f"  |f(x*)|={viol:.2e}")

        # Convert to dataframe and validate (only inputs + constraints, not output columns)
        import pandas as pd

        keys = domain.inputs.get_keys()
        df = pd.DataFrame([x_flat.tolist()], columns=keys)
        try:
            domain.validate_candidates(
                df, only_inputs=True, raise_validation_error=True
            )
            valid = True
        except Exception as e:
            valid = False
            print(f"  validation FAILED: {e}")
        print(f"  BoFire validation: {'PASS' if valid else 'FAIL'}")
        return {
            "ok": True,
            "elapsed": elapsed,
            "violation": viol,
            "valid": valid,
            "acqf_val": acqf_val.item(),
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ok=False  elapsed={elapsed:.2f}s  error={type(e).__name__}: {e}")
        return {"ok": False, "elapsed": elapsed, "violation": None, "valid": False}


def print_summary(result_a, result_b):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'Method A (BoTorch)':<22} {'Method B (AL)'}")
    print("-" * 60)
    for key in ("ok", "elapsed", "violation", "valid", "acqf_val"):
        va = result_a.get(key, "—")
        vb = result_b.get(key, "—")
        if key == "elapsed" and isinstance(va, float):
            va = f"{va:.2f}s"
        if key == "elapsed" and isinstance(vb, float):
            vb = f"{vb:.2f}s"
        if key == "violation" and isinstance(va, float):
            va = f"{va:.2e}"
        if key == "violation" and isinstance(vb, float):
            vb = f"{vb:.2e}"
        if key == "acqf_val" and isinstance(va, float):
            va = f"{va:.4f}"
        if key == "acqf_val" and isinstance(vb, float):
            vb = f"{vb:.4f}"
        print(f"  {key:<23} {str(va):<22} {str(vb)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--n-restarts", type=int, default=N_RESTARTS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    all_results_a = []
    all_results_b = []

    for seed in args.seeds:
        print(f"\n{'=' * 60}")
        print(f"SEED {seed}")
        print(f"{'=' * 60}")

        domain, seed_df = setup_scenario()
        # Rebuild scenario with different seed
        spec = fm3_circle_equality_string(dim=2, radius=0.5)
        domain = spec.make_domain()
        seed_df = spec.make_seed_experiments(domain, n=N_INITIAL, seed=seed)
        print(f"  seed experiments: {len(seed_df)} rows")

        strategy = build_strategy_with_model(domain, seed_df)
        print("  surrogate fitted.")

        result_a = method_a_botorch(strategy, domain, n_restarts=args.n_restarts)
        result_b = method_b_al(strategy, domain, n_restarts=args.n_restarts)

        all_results_a.append(result_a)
        all_results_b.append(result_b)

        print_summary(result_a, result_b)

    if len(args.seeds) > 1:
        print(f"\n{'=' * 60}")
        print("AGGREGATE ACROSS ALL SEEDS")
        print(f"{'=' * 60}")
        ok_a = sum(r["ok"] for r in all_results_a)
        ok_b = sum(r["ok"] for r in all_results_b)
        valid_a = sum(r.get("valid", False) for r in all_results_a)
        valid_b = sum(r.get("valid", False) for r in all_results_b)
        times_a = [r["elapsed"] for r in all_results_a if r["ok"]]
        times_b = [r["elapsed"] for r in all_results_b if r["ok"]]
        viols_a = [
            r["violation"] for r in all_results_a if r.get("violation") is not None
        ]
        viols_b = [
            r["violation"] for r in all_results_b if r.get("violation") is not None
        ]
        n = len(args.seeds)
        print(f"  ok_rate:      A={ok_a}/{n}  B={ok_b}/{n}")
        print(f"  valid_rate:   A={valid_a}/{n}  B={valid_b}/{n}")
        if times_a:
            print(
                f"  mean_time:    A={sum(times_a) / len(times_a):.2f}s  B={sum(times_b) / len(times_b):.2f}s"
            )
        if viols_a:
            mean_a = sum(viols_a) / len(viols_a)
            mean_b = sum(viols_b) / len(viols_b)
            print(f"  mean_|f|:     A={mean_a:.2e}  B={mean_b:.2e}")
