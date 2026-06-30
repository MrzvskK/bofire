"""Manifold-aware acquisition optimisers for equality-constrained domains.

Two approaches implemented here, both solving the competing-gradient problem
identified in Section 10 of RESEARCH.md:

## Option 2 — Null-space parametrisation (circle/sphere case)

For a circle equality f(x) = x₀²+x₁²−r² = 0, reparametrise:
    x(θ) = [r·cos(θ), r·sin(θ)]

The optimisation becomes 1D in θ. The equality is exact by construction —
no penalty, no multiplier, no retraction needed. Inequality constraints become
nonlinear constraints on θ (1D), handled via a simple quadratic penalty.

The key insight: the equality gradient cannot compete with the inequality
gradient because they live in completely disjoint spaces — equality is in the
construction of x(θ), inequality is in the objective over θ.

Generalises to n-sphere: spherical coordinates (n−1 angles).

## Option 1 — Riemannian gradient descent with retraction (general manifolds)

For an arbitrary smooth equality manifold f(x) = 0:
1. Project the acquisition+inequality gradient onto the tangent space of f:
       ∇^tangent L = ∇L − (∇L · n̂) n̂,  n̂ = ∇f / |∇f|
2. Step in the tangent direction (stays approximately on manifold)
3. Retract: project back onto manifold by minimising f(x)² for a few steps
4. Repeat

No equality term in the objective — only inequalities and acquisition.
The equality is enforced geometrically, not via penalty.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from botorch.acquisition import AcquisitionFunction
from torch import Tensor

from bofire.benchmarks.nonlinear_failure_modes.augmented_lagrangian import (
    _generate_al_initial_conditions,
)
from bofire.data_models.constraints.api import (
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.utils.torch_tools import (
    _nonlinear_constraint_feature_indices,
    evaluate_nonlinear_constraint_on_tensor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_ineq(x_2d: Tensor, ineq_info: list) -> list[float]:
    """Return list of inequality residuals g_j(x) (feasible when <= 0)."""
    vals = []
    with torch.no_grad():
        for c, feat_idx in ineq_info:
            g = evaluate_nonlinear_constraint_on_tensor(c, x_2d, feat_idx)
            vals.append(g.squeeze().item())
    return vals


def _ineq_penalty(x_2d: Tensor, ineq_info: list, mu: float) -> Tensor:
    """Quadratic penalty for inequality violations: (mu/2) * max(0, g)^2."""
    total = torch.tensor(0.0, dtype=x_2d.dtype, device=x_2d.device)
    for c, feat_idx in ineq_info:
        g = evaluate_nonlinear_constraint_on_tensor(c, x_2d, feat_idx).squeeze()
        total = total + (mu / 2) * torch.clamp(g, min=0.0) ** 2
    return total


# ---------------------------------------------------------------------------
# Option 2: Null-space parametrisation for circle / n-sphere
# ---------------------------------------------------------------------------


def _parse_sphere_radius(domain: Domain) -> Optional[float]:
    """Extract radius from a single sphere equality f = ||x||^2 - r^2 = 0.

    Returns None if the constraint doesn't match the sphere pattern.
    The expression must be exactly  "x0**2 + x1**2 + ... - <r^2>"
    """
    eq_constraints = domain.constraints.get(NonlinearEqualityConstraint)
    if len(eq_constraints) != 1:
        return None
    expr = eq_constraints[0].expression.strip().replace(" ", "")
    # Expect pattern: x0**2+x1**2+...-<number>
    features = eq_constraints[0].features
    expected_prefix = "+".join(f"{f}**2" for f in features) + "-"
    if not expr.startswith(expected_prefix):
        return None
    try:
        r2 = float(expr[len(expected_prefix) :])
        return math.sqrt(r2)
    except ValueError:
        return None


def optimize_acqf_null_space_sphere(
    acqf: AcquisitionFunction,
    bounds: Tensor,
    domain: Domain,
    n_restarts: int = 5,
    q: int = 1,
    n_steps: int = 200,
    lr: float = 0.1,
    ineq_mu: float = 50.0,
    eq_tol: float = 1e-8,
    ineq_tol: float = 1e-4,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Null-space parametrisation for sphere equality constraints.

    Works for dim=2 (circle) and dim=3 (sphere) equality constraints of the
    form  x0²+x1²+...+x(n-1)² = r².

    For dim=2: x(θ) = [r·cos(θ), r·sin(θ)], optimise over θ ∈ ℝ.
    For dim=3: x(θ,φ) = [r·sin(φ)cos(θ), r·sin(φ)sin(θ), r·cos(φ)].

    Equalities are exact by construction. Inequalities are penalised in θ-space.

    Args:
        acqf: BoTorch acquisition function (maximised).
        bounds: [2, n_dims] box bounds.
        domain: BoFire domain.
        n_restarts: Number of independent restarts in θ-space.
        q: Candidate batch size.
        n_steps: Gradient descent steps per restart.
        lr: Learning rate for Adam optimiser in θ-space.
        ineq_mu: Quadratic penalty coefficient for inequality constraints.
        eq_tol: Tolerance for equality check on returned candidates.
        ineq_tol: Tolerance for inequality check.
        verbose: Print per-restart progress.

    Returns:
        (candidates [1, q, n_dims], acqf_value [1])
    """
    r = _parse_sphere_radius(domain)
    if r is None:
        raise ValueError(
            "optimize_acqf_null_space_sphere requires a single sphere equality "
            "constraint of the form x0**2 + x1**2 + ... - r**2 = 0."
        )

    n_dims = bounds.shape[-1]
    if n_dims not in (2, 3):
        raise ValueError(
            f"Null-space sphere parametrisation supports dim=2 or dim=3, got {n_dims}."
        )

    eq_info = [
        (c, _nonlinear_constraint_feature_indices(c, domain))
        for c in domain.constraints.get(NonlinearEqualityConstraint)
    ]
    ineq_info = [
        (c, _nonlinear_constraint_feature_indices(c, domain))
        for c in domain.constraints.get(NonlinearInequalityConstraint)
    ]

    # Generate ICs on the manifold, convert to angular coordinates
    X_init = _generate_al_initial_conditions(bounds, n_restarts, eq_info, ineq_info)

    def angles_from_cartesian(x: Tensor) -> Tensor:
        """Convert Cartesian point on sphere to angular params."""
        if n_dims == 2:
            return torch.atan2(x[1], x[0]).unsqueeze(0)  # [1] — θ
        else:
            # φ = polar angle from z-axis, θ = azimuth
            phi = torch.acos(torch.clamp(x[2] / r, -1.0, 1.0))
            theta = torch.atan2(x[1], x[0])
            return torch.stack([theta, phi])  # [2]

    def cartesian_from_angles(angles: Tensor) -> Tensor:
        """Convert angular params to Cartesian point on sphere."""
        if n_dims == 2:
            theta = angles[0]
            return torch.stack([r * torch.cos(theta), r * torch.sin(theta)])
        else:
            theta, phi = angles[0], angles[1]
            return torch.stack(
                [
                    r * torch.sin(phi) * torch.cos(theta),
                    r * torch.sin(phi) * torch.sin(theta),
                    r * torch.cos(phi),
                ]
            )

    best_acqf_val = float("-inf")
    best_x = X_init[0].clone()
    fallback_viol, fallback_x = float("inf"), X_init[0].clone()

    for restart in range(n_restarts):
        angles = angles_from_cartesian(X_init[restart]).clone().requires_grad_(True)
        opt = torch.optim.Adam([angles], lr=lr)

        for _ in range(n_steps):
            opt.zero_grad()
            x_cart = cartesian_from_angles(angles)
            X = x_cart.reshape(1, q, n_dims)
            loss = -acqf(X).squeeze()
            # Inequality penalty in θ-space (via the Cartesian mapping)
            loss = loss + _ineq_penalty(x_cart.unsqueeze(0), ineq_info, ineq_mu)
            loss.backward()
            opt.step()

        # Evaluate final candidate
        with torch.no_grad():
            x_final = cartesian_from_angles(angles).detach()
            eq_viol = abs((x_final**2).sum().item() - r**2)
            ineq_residuals = _eval_ineq(x_final.unsqueeze(0), ineq_info)
            ineq_viol = max((max(0.0, g) for g in ineq_residuals), default=0.0)
            max_viol = max(eq_viol, ineq_viol)

            if verbose:
                val = acqf(x_final.reshape(1, q, n_dims)).item()
                print(
                    f"  restart={restart}  |eq|={eq_viol:.2e}"
                    f"  ineq_viol={ineq_viol:.2e}  acqf={val:.4f}",
                    flush=True,
                )

        if max_viol < fallback_viol:
            fallback_viol = max_viol
            fallback_x = x_final.clone()

        if eq_viol >= eq_tol or ineq_viol >= ineq_tol:
            if verbose:
                print(f"  restart={restart}: not feasible, skipping.", flush=True)
            continue

        with torch.no_grad():
            val = acqf(x_final.reshape(1, q, n_dims)).squeeze().item()
            if math.isnan(val):
                continue
        if val > best_acqf_val:
            best_acqf_val = val
            best_x = x_final.clone()

    if best_acqf_val == float("-inf"):
        best_x = fallback_x

    return best_x.reshape(1, q, n_dims), torch.tensor(
        [best_acqf_val], dtype=bounds.dtype, device=bounds.device
    )


# ---------------------------------------------------------------------------
# Option 1: Riemannian gradient descent with retraction (general manifolds)
# ---------------------------------------------------------------------------


def _project_gradient_to_tangent(grad: Tensor, x: Tensor, eq_info: list) -> Tensor:
    """Project gradient onto tangent space of the equality manifold at x.

    For each equality constraint f_i, compute the unit normal n̂_i = ∇f_i/|∇f_i|
    and subtract the component of grad along n̂_i:
        grad_tangent = grad - Σ_i (grad · n̂_i) n̂_i

    For multiple constraints, this is a sequential orthogonalisation (Gram-Schmidt).
    Exact when constraints are orthogonal; approximate otherwise.
    """
    g = grad.clone()
    x_tmp = x.detach().clone().requires_grad_(True)
    for c, feat_idx in eq_info:
        if x_tmp.grad is not None:
            x_tmp.grad.zero_()
        f = evaluate_nonlinear_constraint_on_tensor(c, x_tmp.unsqueeze(0), feat_idx)
        f.squeeze().backward()
        if x_tmp.grad is None:
            continue
        n = x_tmp.grad.clone()
        n_norm = n.norm()
        if n_norm < 1e-12:
            continue
        n_hat = n / n_norm
        g = g - (g @ n_hat) * n_hat
        x_tmp = x_tmp.detach().clone().requires_grad_(True)
    return g


def _retract_to_manifold(
    x: Tensor,
    eq_info: list,
    bounds: Tensor,
    n_steps: int = 50,
    tol: float = 1e-10,
) -> Tensor:
    """Project x back onto the equality manifold by minimising Σ f_i(x)².

    Uses LBFGS with strong Wolfe line search for fast, tight convergence.
    """
    x_r = x.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS(
        [x_r],
        max_iter=n_steps,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
    )

    def closure():
        opt.zero_grad()
        total = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        for c, feat_idx in eq_info:
            f = evaluate_nonlinear_constraint_on_tensor(c, x_r.unsqueeze(0), feat_idx)
            total = total + f.squeeze() ** 2
        total.backward()
        return total

    opt.step(closure)
    with torch.no_grad():
        x_r.clamp_(bounds[0], bounds[1])
    return x_r.detach()


def optimize_acqf_riemannian(
    acqf: AcquisitionFunction,
    bounds: Tensor,
    domain: Domain,
    n_restarts: int = 5,
    q: int = 1,
    n_steps: int = 200,
    lr: float = 0.01,
    retract_every: int = 10,
    retract_steps: int = 20,
    ineq_mu: float = 50.0,
    eq_tol: float = 1e-6,
    ineq_tol: float = 1e-4,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Riemannian gradient descent for arbitrary equality + inequality domains.

    Optimises the acquisition function by moving along the equality manifold:
    1. Compute gradient of (acqf + inequality penalties) w.r.t. x.
    2. Project gradient onto tangent space of the equality manifold (removes
       normal component — the source of competing gradients in plain AL).
    3. Step in the projected (tangential) direction.
    4. Every `retract_every` steps: retract x back onto the manifold by
       minimising Σ f_i(x)² (a few Adam steps).
    5. Repeat for n_steps.

    The equality is NOT in the objective — only in the retraction. Inequalities
    are penalised in the objective. Because inequality gradients are projected
    onto the tangent space before each step, their normal component is removed,
    eliminating the competing-gradient problem.

    Args:
        acqf: BoTorch acquisition function (maximised).
        bounds: [2, n_dims] box bounds.
        domain: BoFire domain.
        n_restarts: Independent restarts.
        q: Candidate batch size.
        n_steps: Gradient steps per restart.
        lr: Step size (applied to tangential gradient).
        retract_every: Frequency of retraction steps.
        retract_steps: Adam steps per retraction.
        ineq_mu: Quadratic penalty for inequality violations.
        eq_tol: Equality feasibility tolerance for acceptance.
        ineq_tol: Inequality feasibility tolerance for acceptance.
        verbose: Print per-restart progress.

    Returns:
        (candidates [1, q, n_dims], acqf_value [1])
    """
    eq_constraints = domain.constraints.get(NonlinearEqualityConstraint)
    ineq_constraints = domain.constraints.get(NonlinearInequalityConstraint)

    if len(eq_constraints) == 0:
        raise ValueError(
            "optimize_acqf_riemannian requires at least one equality constraint."
        )

    eq_info = [
        (c, _nonlinear_constraint_feature_indices(c, domain)) for c in eq_constraints
    ]
    ineq_info = [
        (c, _nonlinear_constraint_feature_indices(c, domain)) for c in ineq_constraints
    ]

    X_init = _generate_al_initial_conditions(bounds, n_restarts, eq_info, ineq_info)

    best_acqf_val = float("-inf")
    best_x = X_init[0].clone()
    fallback_viol, fallback_x = float("inf"), X_init[0].clone()

    for restart in range(n_restarts):
        x = X_init[restart].clone()

        for step in range(n_steps):
            x_req = x.detach().clone().requires_grad_(True)

            # Compute gradient of (acqf + ineq_penalty) w.r.t. x
            X = x_req.reshape(1, q, -1)
            loss = -acqf(X).squeeze() + _ineq_penalty(
                x_req.unsqueeze(0), ineq_info, ineq_mu
            )
            loss.backward()

            grad = x_req.grad.clone()
            # Project gradient onto tangent space of equality manifold
            # (must run outside no_grad — requires backward through constraint Jacobian)
            grad_tangent = _project_gradient_to_tangent(grad, x_req.detach(), eq_info)

            with torch.no_grad():
                # Gradient ascent on acqf → gradient descent on -acqf
                # We minimise loss, so step in -grad direction (already negated acqf)
                x = x - lr * grad_tangent
                x = x.clamp(bounds[0], bounds[1])

            # Periodically retract back onto manifold
            if (step + 1) % retract_every == 0:
                x = _retract_to_manifold(
                    x, eq_info, bounds, n_steps=retract_steps, tol=1e-8
                )

        # Final retraction for tight equality
        x = _retract_to_manifold(x, eq_info, bounds, n_steps=100, tol=1e-12)

        # Evaluate
        with torch.no_grad():
            eq_residuals = []
            for c, feat_idx in eq_info:
                f = evaluate_nonlinear_constraint_on_tensor(c, x.unsqueeze(0), feat_idx)
                eq_residuals.append(f.squeeze().item())
            ineq_residuals = _eval_ineq(x.unsqueeze(0), ineq_info)

            eq_viol = max((abs(r) for r in eq_residuals), default=0.0)
            ineq_viol = max((max(0.0, g) for g in ineq_residuals), default=0.0)
            max_viol = max(eq_viol, ineq_viol)

            val = acqf(x.reshape(1, q, -1)).squeeze().item()

            if verbose:
                print(
                    f"  restart={restart}  |eq|={eq_viol:.2e}"
                    f"  ineq_viol={ineq_viol:.2e}  acqf={val:.4f}",
                    flush=True,
                )

        if max_viol < fallback_viol:
            fallback_viol = max_viol
            fallback_x = x.clone()

        if eq_viol >= eq_tol or ineq_viol >= ineq_tol:
            if verbose:
                print(f"  restart={restart}: not feasible, skipping.", flush=True)
            continue

        if math.isnan(val):
            continue
        if val > best_acqf_val:
            best_acqf_val = val
            best_x = x.clone()

    if best_acqf_val == float("-inf"):
        best_x = fallback_x

    n_dims = bounds.shape[-1]
    return best_x.reshape(1, q, n_dims), torch.tensor(
        [best_acqf_val], dtype=bounds.dtype, device=bounds.device
    )
