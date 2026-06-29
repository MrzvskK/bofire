"""Profile a single FM-3 circle equality ask() to find the bottleneck."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from bofire.data_models.constraints.api import NonlinearEqualityConstraint
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MinimizeObjective
from bofire.data_models.strategies.api import SoboStrategy as SoboDataModel
from bofire.strategies.api import SoboStrategy
from bofire.utils.torch_tools import get_nonlinear_constraints


print("1. Imports done", flush=True)
print("2. BoFire imports done", flush=True)

# Build FM-3 domain: x0^2 + x1^2 - 0.25 = 0  (circle radius=0.5 in [-1,1]^2)
dim, radius = 2, 0.5
domain = Domain(
    inputs=Inputs(
        features=[ContinuousInput(key=f"x{i}", bounds=(-1, 1)) for i in range(dim)]
    ),
    outputs=Outputs(
        features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
    ),
    constraints=Constraints(
        constraints=[
            NonlinearEqualityConstraint(
                features=[f"x{i}" for i in range(dim)],
                expression=" + ".join(f"x{i}**2" for i in range(dim))
                + f" - {radius**2}",
            )
        ]
    ),
)
print("3. Domain built", flush=True)

# Generate seed experiments ON the circle (analytic)
rng = np.random.default_rng(0)
n_initial = 8
z = rng.normal(size=(n_initial, dim))
z /= np.linalg.norm(z, axis=1, keepdims=True)
x = z * radius
experiments = pd.DataFrame(x, columns=[f"x{i}" for i in range(dim)])
experiments["y"] = 0.0
experiments["valid_y"] = 1
print(
    f"4. Seed data: {n_initial} points on circle, e.g. x0={x[0,0]:.4f}, x1={x[0,1]:.4f}",
    flush=True,
)

# Build strategy
t0 = time.time()
data_model = SoboDataModel(domain=domain)
strategy = SoboStrategy(data_model=data_model)
strategy._validation_tol = 1e-3
print(f"5. Strategy created in {time.time()-t0:.2f}s", flush=True)

# Tell
t0 = time.time()
strategy.tell(experiments)
print(f"6. tell() done in {time.time()-t0:.2f}s", flush=True)

# Check constraint setup
nl = get_nonlinear_constraints(domain)
print(
    f"7. nonlinear_constraints count: {len(nl)} (equality represented as 2 inequalities)",
    flush=True,
)
print(f"   constraint fns: {[type(fn).__name__ for fn, _ in nl]}", flush=True)

# Ask — the main bottleneck
print("8. Starting ask(1) ...", flush=True)
t0 = time.time()
try:
    candidates = strategy.ask(candidate_count=1)
    print(
        f"9. ask() done in {time.time()-t0:.2f}s  ok=True  candidates:\n{candidates}",
        flush=True,
    )
except Exception as e:
    print(
        f"9. ask() FAILED in {time.time()-t0:.2f}s  error={type(e).__name__}: {e}",
        flush=True,
    )
