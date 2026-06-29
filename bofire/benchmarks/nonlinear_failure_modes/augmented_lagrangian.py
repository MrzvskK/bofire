"""Augmented Lagrangian optimizer for equality-constrained acquisition functions.

Research prototype — not yet integrated into the main BoFire strategy path.

The standard BoFire approach encodes `f(x) = 0` as two BoTorch inequalities:
    c1(x) = eq_tol - f(x) >= 0
    c2(x) = eq_tol + f(x) >= 0
forming a thin band of width 2*eq_tol. scipy SLSQP struggles with curved bands
(circle, sphere) because the feasible region has a position-dependent Jacobian.

This module implements the Augmented Lagrangian (AL) / Method of Multipliers:

    maximize   acqf(x) - lambda * f(x) - (mu/2) * f(x)^2
    subject to x in [lower, upper]   (box constraints only)

The inner problem is unconstrained in the nonlinear sense — only box bounds remain.
scipy L-BFGS-B (or torch LBFGS) handles this efficiently regardless of constraint
curvature. The outer loop updates lambda and escalates mu until |f(x)| < tol.

Key properties:
- No curved slab → no linesearch failures from scipy.
- Converges to a constrained optimum as mu increases (exact penalty theorem).
- Works for any differentiable f(x) (string or callable expression).

Usage (research):
    from bofire.benchmarks.nonlinear_failure_modes.augmented_lagrangian import (
        optimize_acqf_al,
    )
    candidates, acqf_val = optimize_acqf_al(
        acqf=acqf,
        bounds=bounds,
        domain=domain,
        n_restarts=5,
        q=1,
    )
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from botorch.acquisition import AcquisitionFunction
from torch import Tensor

from bofire.data_models.constraints.api import NonlinearEqualityConstraint
from bofire.data_models.domain.api import Domain
from bofire.utils.torch_tools import (
    _nonlinear_constraint_feature_indices,
    evaluate_nonlinear_constraint_on_tensor,
)


def _al_objective(
    x_flat: Tensor,
    acqf: AcquisitionFunction,
    eq_info: list,
    lambdas: Tensor,
    mu: float,
    q: int,
) -> Tensor:
    """Augmented Lagrangian objective for a single candidate x_flat (shape [n_dims]).

    Returns the scalar AL objective:
        -acqf(x) + sum_i [ lambda_i * f_i(x) + (mu/2) * f_i(x)^2 ]

    Negative acqf because we minimize (acqf is a maximization objective).
    """
    n_dims = x_flat.shape[-1]
    X = x_flat.reshape(1, q, n_dims)  # [1, q, n_dims] for acqf
    acqf_val = acqf(X)  # scalar or [1]
    loss = -acqf_val.squeeze()

    for i, (c, feat_idx) in enumerate(eq_info):
        f_val = evaluate_nonlinear_constraint_on_tensor(
            c, x_flat.unsqueeze(0), feat_idx
        )
        f_scalar = f_val.squeeze()
        loss = loss + lambdas[i] * f_scalar + (mu / 2) * f_scalar**2

    return loss


def _al_inner_optimize(
    x_init: Tensor,
    acqf: AcquisitionFunction,
    eq_info: list,
    lambdas: Tensor,
    mu: float,
    bounds: Tensor,
    q: int,
    n_inner: int = 100,
    lr: float = 0.01,
) -> Tensor:
    """Minimize AL objective from x_init using torch LBFGS with box projection.

    Returns the optimized x (shape [n_dims]).
    """
    x = x_init.detach().clone().requires_grad_(True)
    lower = bounds[0]
    upper = bounds[1]

    opt = torch.optim.LBFGS(
        [x],
        lr=lr,
        max_iter=20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = _al_objective(x, acqf, eq_info, lambdas, mu, q)
        loss.backward()
        return loss

    for _ in range(n_inner // 20):  # each LBFGS call does up to 20 iterations
        opt.step(closure)
        with torch.no_grad():
            x.clamp_(lower, upper)

    return x.detach()


def _generate_al_initial_conditions(
    bounds: Tensor,
    n_restarts: int,
    eq_info: list,
    max_iter: int = 200,
    lr: float = 0.05,
    tol: float = 1e-5,
) -> Tensor:
    """Generate n_restarts initial conditions near the equality manifold.

    Uses gradient-descent projection (same as feasible_ic_generator in
    acqf_optimization.py) to snap random box samples onto the equality manifold.
    """
    n_dims = bounds.shape[-1]
    lower = bounds[0]
    upper = bounds[1]
    n_candidates = max(n_restarts * 20, 128)

    X = lower + (upper - lower) * torch.rand(
        n_candidates, n_dims, device=bounds.device, dtype=bounds.dtype
    )
    X_proj = X.clone().requires_grad_(True)
    opt = torch.optim.Adam([X_proj], lr=lr)

    for _ in range(max_iter):
        opt.zero_grad()
        total = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        for c, feat_idx in eq_info:
            f = evaluate_nonlinear_constraint_on_tensor(c, X_proj, feat_idx)
            total = total + (f**2).sum()
        if total.item() < tol * n_candidates:
            break
        total.backward()
        opt.step()
        with torch.no_grad():
            X_proj.clamp_(bounds[0], bounds[1])

    X_on_manifold = X_proj.detach()

    # Select n_restarts with most spread (just take first n_restarts for now)
    return X_on_manifold[:n_restarts]


def optimize_acqf_al(
    acqf: AcquisitionFunction,
    bounds: Tensor,
    domain: Domain,
    n_restarts: int = 5,
    q: int = 1,
    n_outer: int = 8,
    n_inner: int = 100,
    mu_init: float = 1.0,
    mu_scale: float = 4.0,
    mu_max: float = 1e5,
    eq_tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Augmented Lagrangian optimizer for equality-constrained acquisition maximization.

    Args:
        acqf: BoTorch acquisition function (maximized).
        bounds: [2, n_dims] box bounds tensor.
        domain: BoFire domain (used to extract equality constraints).
        n_restarts: Number of independent restarts.
        q: Candidate batch size.
        n_outer: Maximum AL outer iterations per restart.
        n_inner: Maximum inner optimization iterations per outer step.
        mu_init: Initial penalty parameter.
        mu_scale: Penalty escalation factor per outer iteration.
        mu_max: Maximum penalty parameter.
        eq_tol: Convergence tolerance for |f(x)|.
        verbose: Print per-restart progress.

    Returns:
        (candidates, acqf_vals): best candidate [1, q, n_dims] and its value [1].
    """
    eq_constraints = domain.constraints.get(NonlinearEqualityConstraint)
    if len(eq_constraints) == 0:
        raise ValueError("optimize_acqf_al requires at least one equality constraint.")

    eq_info = [
        (c, _nonlinear_constraint_feature_indices(c, domain)) for c in eq_constraints
    ]
    n_eq = len(eq_constraints)

    # Generate initial conditions on the equality manifold
    X_init = _generate_al_initial_conditions(bounds, n_restarts, eq_info)

    best_acqf_val = float("-inf")
    best_x = X_init[0].clone()

    for restart in range(n_restarts):
        x = X_init[restart].clone()
        lambdas = torch.zeros(n_eq, dtype=bounds.dtype, device=bounds.device)
        mu = mu_init

        for outer in range(n_outer):
            x = _al_inner_optimize(
                x_init=x,
                acqf=acqf,
                eq_info=eq_info,
                lambdas=lambdas,
                mu=mu,
                bounds=bounds,
                q=q,
                n_inner=n_inner,
            )

            # Compute constraint residuals at current x
            residuals = []
            with torch.no_grad():
                for c, feat_idx in eq_info:
                    f = evaluate_nonlinear_constraint_on_tensor(
                        c, x.unsqueeze(0), feat_idx
                    )
                    residuals.append(f.squeeze().item())

            max_viol = max(abs(r) for r in residuals)

            if verbose:
                with torch.no_grad():
                    acqf_val = acqf(x.reshape(1, q, -1)).item()
                print(
                    f"  restart={restart} outer={outer} mu={mu:.1e} "
                    f"|f|={max_viol:.2e} acqf={acqf_val:.4f}",
                    flush=True,
                )

            # Update Lagrange multipliers: λ += μ * f(x*)
            with torch.no_grad():
                for i, res in enumerate(residuals):
                    lambdas[i] += mu * res

            # Escalate penalty
            mu = min(mu * mu_scale, mu_max)

            if max_viol < eq_tol:
                if verbose:
                    print(
                        f"  restart={restart}: converged at outer={outer}", flush=True
                    )
                break

        # Only consider converged restarts — those with |f| < eq_tol — as valid
        # candidates. Unconverged restarts may have higher acqf values but they
        # sit off the manifold and will fail BoFire validation.
        if max_viol >= eq_tol:
            if verbose:
                print(
                    f"  restart={restart}: did not converge (|f|={max_viol:.2e} > {eq_tol:.0e}), skipping.",
                    flush=True,
                )
            continue

        with torch.no_grad():
            val = acqf(x.reshape(1, q, -1)).squeeze().item()
            if math.isnan(val):
                continue

        if val > best_acqf_val:
            best_acqf_val = val
            best_x = x.clone()

    n_dims = bounds.shape[-1]
    return best_x.reshape(1, q, n_dims), torch.tensor(
        [best_acqf_val], dtype=bounds.dtype, device=bounds.device
    )
