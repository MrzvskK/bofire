# Nonlinear constraint failure modes — lab note

Branch: `research/nonlinear-constraint-failure-modes` (local exploration; **not** merged with `nonlinearC` PR).

**Next steps / program map:** see [`RESEARCH_DIRECTIONS.md`](RESEARCH_DIRECTIONS.md) (tolerance-gate design, smooth BO for equalities, verification checklist — not only library integration).

**LaTeX report:** [`report/RESEARCH_REPORT.tex`](report/RESEARCH_REPORT.tex) — build via `report/generate_figures.py` + `report/build_report.sh`.

## Question

Are nonlinear **equality** constraints handled consistently across pandas validation and BoTorch callables, and do **strategies** behave reliably on equality domains?

## Setup

Harness: `python -m bofire.benchmarks.nonlinear_failure_modes.run`

- **FM-3:** circle equality `x₀² + x₁² − r² = 0`, probe = points near the circle (`ball_boundary_shell`).
- **FM-2 shell:** ball inequality `‖x‖² − r² ≤ 0`, same boundary-style probe (strategy baseline).

Strategy protocol: `n_initial=32`, seeds `0–19`, `ask_n ∈ {1, 4}`. Completed strategy runs used **`_validation_tol` / `equality_tolerance` = 1e-3** (nonlinear-equality / BoTorch patch). **Most of BoFire otherwise uses 1e-6** (`validate_candidates`, `is_fulfilled` defaults). See tolerance table in [`RESEARCH_DIRECTIONS.md`](RESEARCH_DIRECTIONS.md).

---

## 1. Backend contract (constraints-only)

### 1a. Mismatched tolerances (pandas `tol` swept, torch `equality_tolerance=1e-3` fixed)

| tol (pandas) | disagree (FM-3 d2 r0.5, mean over 20 seeds) | Direction |
|--------------|-----------------------------------------------|-----------|
| 1e-12 | ~50% | **torch-only** (`th_only`) |
| 1e-08 | ~41% | **torch-only** |
| 1e-06 | ~25% | **torch-only** |
| 1e-04 | ~9% | **torch-only** |
| 1e-03 | ~2.5% | **torch-only** |

`feas_th` ~0.91–0.92 flat across pandas `tol`; `feas_pd` rises with `tol`.
**Conclusion:** Apparent “backend mismatch” is largely **comparing different tolerance bands**, not inconsistent geometry.

### 1b. Aligned tolerances (`--match-equality-tol`, torch slack 0)

| tol = eq_tol | disagree (mean) | Direction | Status |
|--------------|-----------------|-----------|--------|
| 1e-12 | ~25% | **pandas-only** | `is_fulfilled` effective band ~1e-9 vs torch 1e-12 |
| 1e-09 | ~2.5% | **pandas-only** | pandas `tol+eps` ≈ 2e-9 vs torch 1e-9 |
| 1e-10 | ~9% | **pandas-only** | (earlier sweep) |
| 1e-08 | &lt;0.4% | **pandas-only** | (earlier sweep) |
| **1e-06** | **0.01%** (max 0.02%) | **pd_only** trace | **BoFire-nominal — Thread A closed** |
| 1e-03 | ~0% | — | (earlier sweep) |

**Verified (2025-05, seeds 0–19, FM-3 shell):** `--tols 1e-6 --match-equality-tol` → `feas_pd` = `feas_th` per seed, `disagree_mean` ≈ 0.0001.

**Conclusion (Thread A):** At **τ = 1e-6** (rest of BoFire), pandas and torch **agree**. Tighter tol sweeps measure **gate definition** (`eps` floor, band width), not BoTorch eval failure. **No further backend forensics required** before Thread B.

**Forensics (`--forensics`, FM-3 matched @ 1e-12, boundary shell):** ~24% `pd_only`; torch margins ~10⁻¹¹ on mismatch points — edge of gate, not O(10⁻³) optimizer slack.

### 1c. Inequalities

FM-2 ball (uniform box and boundary shell): **0%** pandas vs torch disagree across seeds/tols.

### 1d. FM-5 disconnected / nonconvex (constraints-only)

`--scenario-set fm5`, seeds `0–4`, `tol ∈ {1e-12, 1e-8, 1e-6}`, default `equality_tolerance=1e-3`:

| Scenario | Probe | disagree |
|----------|-------|----------|
| `fm5_disjoint_two_balls_callable_d2_s0.6_r0.25` | uniform box | **0%** |
| `fm5_annulus_string_d2_ri0.35_ro0.85` | outer shell | **0%** |

- **Two-ball union** uses `torch.minimum` on two ball violations (callable; plain Python `min` breaks on tensor batches).
- **Annulus** uses two string inequalities (outer + inner); nonconvex boundary but **connected**.

**Conclusion:** Nonconvex / disconnected **inequality** geometries do not introduce pandas vs torch disagreement beyond the equality-tolerance contract already studied on FM-3.

---

## 2. Thread B — strategy / smooth equalities (active)

**Goal:** Make nonlinear **equalities** usable in BO at **BoFire-nominal τ = 1e-6**, without relying on the **1e-3 patch** (`equality_tolerance` / `_validation_tol` in acqf).

**Baseline (completed, τ = 1e-3 patch):**

| Scenario | ok / 40 | Notes |
|----------|---------|--------|
| FM-3 circle equality | 14 / 40 (35%) | `ConstraintNotFulfilledError` after `ask` |
| FM-2 ball inequality shell | 40 / 40 | control |

**B0 result (SOBO FM-3, τ=1e-6 unified, seeds 0–19, `stepB0_fm3_sobo_1e6.csv`):**

| Metric | τ = 1e-6 (B0) | τ = 1e-3 patch (baseline) |
|--------|---------------|---------------------------|
| **ok / 40** | **9 (22.5%)** | 14 (35%) |
| ask_n=1 | 3/20 (15%) | — |
| ask_n=4 | 6/20 (30%) | — |

**Failure modes (τ=1e-6):**

| error_type | count | Meaning |
|------------|-------|---------|
| `ValueError` | **14** | BoTorch: `batch_initial_conditions` must satisfy nonlinear inequality constraints — **IC infeasible** at tight band |
| `ConstraintNotFulfilledError` | **17** | Candidates returned but fail `validate_candidates` |
| (none) | 9 | ok |

**Interpretation:**

1. **H5 confirmed:** Nominal **1e-6 is stricter than the 1e-3 patch** → fewer successful asks (22.5% vs 35%). Widening τ was a **workaround**, not fixing geometry.
2. **New dominant failure at 1e-6:** **14/31 failures** are **before** validation — BoTorch cannot find feasible initial conditions in the ±1e-6 tube. Thread B should target **on-manifold IC** (B2), not tighter pandas/torch alignment.
3. **When ok:** `max_violation_mean` ~ **10⁻¹¹–10⁻¹⁰** ≪ 1e-6 — optimizer can hit the manifold when the run completes; problem is **getting there**, not the validation band width alone.
4. **Harness caveat:** `feas_rate` uses inequality-style violation on raw \(f(x)\); for equalities many `ok=True` rows show `feas_rate=0`. Trust **`ok`** and `max_viol` until metrics use \(|f|\).

**Next (B1+):** projection / on-manifold IC at **τ=1e-6**; optional B0 control on FM-2 inequality @ 1e-6.

```bash
# B0 control (optional)
python -m bofire.benchmarks.nonlinear_failure_modes.run \
  --mode strategy --scenario fm2_boundary_ball_shell_d2_r0.5 \
  --seeds-range 0-19 --strategy sobo --ask-ns 1,4 --n-initial 32 \
  --validation-tol 1e-6 \
  --out bofire/benchmarks/nonlinear_failure_modes/_runs/stepB0_fm2_sobo_1e6.csv
```

**B1 result (SOBO FM-3, τ=1e-6, `--circle-projection`, `stepB1_fm3_sobo_1e6_proj.csv`):**

| Metric | B0 (no proj) | B1 (proj + circle IC) | Baseline τ=1e-3 patch |
|--------|--------------|------------------------|------------------------|
| **ok / 40** | 9 (22.5%) | **40 (100%)** | 14 (35%) |
| ask_n=1 | 15% | **100%** | — |
| ask_n=4 | 30% | **100%** | — |
| IC `ValueError` | 14 | **0** | — |

**Interpretation:**

- **H3 supported (harness):** Geometry-aware **on-manifold IC** + **pre-validation projection** removes both failure modes at BoFire-nominal τ=1e-6. This is stronger than widening to 1e-3 (35% ok without projection).
- **`feas_rate` in CSV** measures **raw** candidates from BoTorch (often off-manifold, \(|f|\sim 10^{-9}\)). **`ok=True`** means projection ran inside the patched `validate_candidates` before the gate. Product would need to **return projected** candidates, not only validate them.
- BoTorch still logs scipy line-search warnings; failures are recovered via retries / IC — no hard abort.

| ID | Intervention | Status |
|----|--------------|--------|
| B1 | Projection + on-manifold IC | **Done** — 100% ok @ 1e-6 in harness |
| B3 | Violation histogram / return projected x | Planned |

See [`RESEARCH_DIRECTIONS.md`](RESEARCH_DIRECTIONS.md) § Thread B for full program.

---

## 2b. Strategy behavior (SOBO, τ = 1e-3 patch — completed)

| Scenario | ok / 40 runs | Notes |
|----------|--------------|--------|
| **FM-3** circle equality | **14 / 40 (35%)** | 26× `ConstraintNotFulfilledError` after `ask` |
| **FM-2** ball inequality shell | **40 / 40 (100%)** | `max_viol = 0` |

When FM-3 `ok=yes`, candidates sit at **~1e-3** violation (equality band edge).

**Conclusion:** Under the **current 1e-3 contract**, SOBO **often fails validation on equalities** but **not** on this inequality baseline. This is user-visible (not just a benchmark artifact).

---

## 3. Strategy behavior (MOBO, repeat)

`MoboStrategy` requires **≥2 objectives**. Single-output FM-3/FM-2 scenarios use dedicated **mobo** ids (same constraints, `y1`/`y2` minimize):

| Scenario | Role |
|----------|------|
| `fm3_circle_equality_mobo_d2_r0.5` | equality + MOBO |
| `fm2_boundary_ball_shell_mobo_d2_r0.5` | inequality baseline + MOBO |

```bash
python -m bofire.benchmarks.nonlinear_failure_modes.run \
  --mode strategy --scenario fm3_circle_equality_mobo_d2_r0.5 \
  --seeds-range 0-19 --strategy mobo --ask-ns 1,4 --n-initial 32 \
  --out bofire/benchmarks/nonlinear_failure_modes/_runs/step4_fm3_mobo.csv

python -m bofire.benchmarks.nonlinear_failure_modes.run \
  --mode strategy --scenario fm2_boundary_ball_shell_mobo_d2_r0.5 \
  --seeds-range 0-19 --strategy mobo --ask-ns 1,4 --n-initial 32 \
  --out bofire/benchmarks/nonlinear_failure_modes/_runs/step4_fm2_mobo.csv
```

**Note:** Running MOBO on single-output `fm3_circle_equality_string_d2_r0.5` fails immediately with `ValidationError` (expected).

### MOBO results (FM-3 mobo scenario, completed)

| Strategy | Scenario | ok / 40 | ask_n=1 | ask_n=4 |
|----------|----------|---------|---------|---------|
| SOBO | `fm3_circle_equality_string_d2_r0.5` | **14 / 40 (35%)** | — | — |
| MOBO | `fm3_circle_equality_mobo_d2_r0.5` | **14 / 40 (35%)** | 45% | 25% |

Same **ok rate** as SOBO on equality in this protocol; failures are still mostly `ConstraintNotFulfilledError`. Successful MOBO runs often show `feas_rate=1.0` or `0.0` depending on whether candidates land inside the ±1e-3 band under harness scoring (not a strategy crash).

CSV: `_runs/step4_fm3_mobo.csv`

### MOBO inequality baseline (`fm2_boundary_ball_shell_mobo_d2_r0.5`)

| | ok / 40 | ask_n=1 | ask_n=4 |
|---|---------|---------|---------|
| SOBO | **40/40 (100%)** | — | — |
| MOBO | **39/40 (97.5%)** | 20/20 | 19/20 |

One failure: seed 9, `ask_n=4`, `ConstraintNotFulfilledError`. CSV: `_runs/step4_fm2_mobo.csv` (~21 min).

### Strategy comparison (equality vs inequality)

| Constraint | SOBO ok | MOBO ok |
|------------|---------|---------|
| Circle **equality** | 35% | 35% |
| Ball **inequality** shell | 100% | 97.5% |

**Conclusion:** Failure mode is tied to **equality**, not to SOBO vs MOBO or MOBO being generally broken.

---

## 4. Intended contract (for a future PR — not `nonlinearC` now)

| Knob | Typical value | Role |
|------|---------------|------|
| `is_fulfilled(..., tol)` | user / `_validation_tol` | pandas: **\|f\| ≤ tol** (equalities) |
| `get_nonlinear_constraints(..., equality_tolerance)` | **1e-3** | torch: ±band as inequality pair |
| `Strategy._validation_tol` | **1e-3** if nonlinear equality | post-`ask` validation |

**Recommendation:** Single source of truth — `equality_tolerance` should follow `_validation_tol` everywhere.

---

## 5. What we are *not* claiming

- Callable vs string inequality mismatch (parity holds when callable is valid).
- Generic “all nonlinear constraints are broken.”
- Ready-to-merge product fix (research only).

---

## Artifacts

| File | Content |
|------|---------|
| `step2a_mismatched.csv` | FM-3 constraints, fixed eq_tol |
| `step2b_matched.csv` | FM-3 constraints, matched eq_tol |
| `step4_strategy_fm3.csv` | SOBO equality (optional) |
| `step4_fm3_mobo.csv` | MOBO equality |
| `step4_fm2_mobo.csv` | MOBO inequality baseline |

---

## Pipeline: what we ran vs next

| Phase | Goal | Status | Notes |
|-------|------|--------|--------|
| **A — 1a** | Mismatched tol → apparent backend bug | **Done** | ~50% `th_only` |
| **A — 1b** | Matched tol; nominal **1e-6** | **Done** | disagree &lt; 0.02%; Thread **closed** |
| **A — forensics** | `pd_only` at 1e-12 / 1e-9 | **Done** | gate semantics |
| **A — FM-5** | Nonconvex / disconnected constraints-only | **Done** | 0% disagree |
| **B — baseline** | Strategy @ 1e-3 patch | **Done** | 35% / 100% equality vs inequality |
| **B — B0** | Strategy @ **1e-6** unified (`--validation-tol`) | **Done** | 22.5% ok; 14× IC `ValueError` |
| **B — B1** | + `--circle-projection` @ 1e-6 | **Done** | **100%** ok; 0 IC errors |
| **B — B1+** | Projection / on-manifold IC | **Planned** | harness prototypes |
| **4** | MOBO @ 1e-3 | **Done** | repeat after B0 at 1e-6 optional |

If “step 2” meant something else (e.g. only FM-2 constraints, or a write-up before fixes), clarify in the issue tracker — numerically, **aligned-tolerance backend work is complete**.

---

## Results synthesis (beyond “they fail”)

### Two different failure layers

1. **Backend consistency (constraints-only)**
   - **Problem:** Comparing `is_fulfilled(tol)` to torch with a **fixed** `equality_tolerance=1e-3` looks like a 50% backend bug.
   - **Reality:** After alignment, backends **agree at operational tolerances**. Inequalities never showed a gap.
   - **Implication:** No urgent “rewrite nonlinear `__call__`” for balls/circles; fix the **contract** and documentation first.

2. **Strategy / optimizer (user-visible)**
   - **Problem:** ~65% of `ask` cycles on **circle equality** end in `ConstraintNotFulfilledError` after `validate_candidates`.
   - **Reality:** Optimizer targets **±1e-3 band**; validation uses the same band; candidates often land **just outside** (or batch q>1 is harder). Inequalities are easy (interior feasible set).
   - **Implication:** “Bad performance” is **thin feasible set + acquisition**, not pandas/torch disagreeing on the same point.

### What is *not* the main issue

- SOBO vs MOBO (same ~35% on equality).
- String vs callable (when callable is valid).
- Random box sampling for inequalities (0% backend disagree).

### What *is* the main issue

- **Equality as a near-zero-measure set** in continuous space, optimized with **soft ±tol inequalities** in BoTorch and checked with **`validate_candidates`** after numerical optimization.
- **Tolerance knobs duplicated** (`equality_tolerance` in `get_nonlinear_constraints` / `acqf_optimization` vs `_validation_tol` in `Strategy`) — confusing for users and for benchmarks unless kept in sync.

---

## How constraint behavior could be improved (future work, not `nonlinearC` now)

Ordered by impact vs effort. These are design options for a **later PR**, informed by this harness.

### A. Contract / API (high impact, moderate effort)

1. **Single tolerance source**
   Pass `Strategy._validation_tol` into `get_nonlinear_constraints(domain, equality_tolerance=...)` and `acqf_optimization` so optimizer and validator use the **same** band.

2. **Explicit equality API**
   Document: equalities are implemented as **two inequalities** in torch; pandas uses **\|f\| ≤ tol**. Users should set one `tol`, not assume 1e-6 validation with 1e-3 optimization.

3. **`validate_candidates(..., raise_validation_error=False)` policy**
   Today failures are hard errors. For equalities, consider **warn + clip/project** or **retry ask** when violation is O(tol) — product decision, not only numerics.

### B. Optimization / acquisition (high impact for equalities, higher effort)

4. **Manifold-aware initialization**
   Seeds already lie on the circle; **initial candidates for acqf** should be sampled **on** (or very near) the equality manifold, not uniform in the box.

5. **Post-optimize projection**
   After `gen_candidates`, project each candidate onto {x : |f(x)| ≤ tol} (e.g. radial scaling for circle). Cheap for some geometries; documents “feasible output” guarantee.

6. **Tighter torch check during optimization**
   Harness found **double slack** if `torch_slack_tol` equals pandas `tol` on top of built-in equality band — optimizer path should use **≥0 on callables only**, matching BoTorch docs.

7. **Batch ask (q>1)**
   Failures more frequent for `ask_n=4` than `ask_n=1` on equalities — investigate **joint** feasibility in BoTorch batch optimization vs per-point validation.

### C. Validation semantics (medium impact)

8. **Harmonize `NonlinearEqualityConstraint.is_fulfilled` with torch pair**
   Residual `pd_only` at 1e-12 under `--match-equality-tol` suggests small **eval** differences (pandas `eval` vs torch loop). Unify eval path or accept documented epsilon.

9. **Report validation margin in `ask` errors**
   When `ConstraintNotFulfilledError` fires, attach **max \|f\|** and **which constraint** — makes 1e-3 edge failures diagnosable without rerunning.

### D. Verification (ongoing)

10. **Keep this harness** for regression:
    - constraints: `--match-equality-tol` on FM-3 variants (dim, radius).
    - strategy: equality vs inequality smoke on CI with small seeds (optional, slow).

11. **Deferred phase 2 forensics** (if tightening to 1e-12 matters):
    Export histogram of \|f(x)\| for `pd_only` points at tol=1e-12 to see if gap is O(1e-15) or O(1e-9).

### Suggested order for a future fix PR

1. Wire **one tolerance** through strategy → torch constraints → validation.
2. Add **regression test** (unit or harness): matched backends &lt;1% disagree at tol=1e-3 on FM-3 shell.
3. Prototype **on-manifold IC** or **projection** for equality domains; re-run strategy grid — target **ok rate** ≫ 35% before tuning GP hyperparams.

---

## Open questions

- Do real user domains use **multiple nonlinear equalities** (harder than single circle)?
- Is **1e-3** intentionally loose for noisy simulators, or legacy from BoTorch examples?
- Should equalities be **first-class** in docs (“not recommended for naive BO”) until manifold samplers exist?

---

## 6. Optimizer profiling session (2026-06-29)

**Setup:** Single-run timing probe on FM-3 circle equality (`x0²+x1²=0.25`, `n_initial=8` seed points exactly on circle, `n_restarts=20` default SOBO optimizer, `_validation_tol=1e-3`).

**Result — wall-clock breakdown:**

| Phase | Time |
|-------|------|
| Imports | ~5s |
| `strategy.tell(8 experiments)` | **0.09s** |
| `strategy.ask(1)` | **433s (7.2 min)** |
| Outcome | `ConstraintNotFulfilledError` |

**Returned candidate:** `x0=-0.013232, x1=0.500824`
→ `|f(x)| = |x0²+x1²-0.25| = |0.251-0.25| = 0.001` — right at the 1e-3 gate, floating-point over.

**Key finding: the bottleneck is entirely inside `optimize_acqf`, not IC generation.**

Tell and IC generation are fast. The 7-minute delay comes from BoTorch running 20 scipy SLSQP restarts on the curved ±1e-3 equality slab. Many restarts fail with **"Positive directional derivative for linesearch" (scipy status 8)**, triggering BoTorch's internal retry loop which calls the IC generator again and re-attempts.

**Why our tests pass quickly:** `tests/bofire/strategies/test_nonlinear_constraints.py` uses `NONLINEAR_BOTORCH_OPTIMIZER` with `n_restarts=2` (10x fewer). Real users hitting the **default `n_restarts=20`** would see 7-minute `ask()` calls on circle-style equality constraints.

**Why linear equality (`x+y=1.5`) is fast:** Constant Jacobian `[1, 1]` — scipy SLSQP handles flat slabs efficiently. Circle slab has position-dependent Jacobian `[2x0, 2x1]` that rotates around the manifold — linesearch cannot stay in the curved band.

**Two distinct failure modes on FM-3:**

| Failure mode | When | Root cause |
|---|---|---|
| **Slow `ask()` (7+ min)** | Default n_restarts=20 | scipy SLSQP inefficient on curved equality slab; retry loop amplifies cost |
| **`ConstraintNotFulfilledError`** on nearly-feasible point | After optimization completes | Optimizer converges just outside ±1e-3 band; hard validation gate rejects it |

**Comparison with B1:** B1 (100% ok) used a circle-specific post-optimization projection (radial snap) before `validate_candidates`. The `nonlinearC` PR's gradient-descent projection handles IC generation but does **not** project candidates after optimization. Without post-optimization snapping, boundary numerical noise causes validation failures.

**What `nonlinearC` PR does and does not solve:**

| Aspect | Status after `nonlinearC` |
|---|---|
| IC generation for equality constraints | Fixed (gradient-descent projection) |
| IC generation for tight inequality constraints | Fixed (tiling fallback) |
| BoTorch validation of ICs (>= 0 threshold) | Fixed |
| Linear equality (x+y=c) — full BO loop | Works fast (~17s) |
| Speed of `ask()` with default n_restarts=20 | Not fixed — 7+ min for circle equality |
| Post-optimization snap to manifold | Not implemented — boundary candidates fail |

---

## 7. Research directions (updated after profiling)

Profiling reveals a **two-layer problem** for BO under nonlinear equality constraints.

### Layer 1 — Optimizer performance (the hard problem)

The ±tol slab representation is a structural liability for second-order optimizers on curved equalities:

- **Augmented Lagrangian (AL):** Encode `f(x)=0` as `min acqf(x) + λf(x) + μf(x)²` and update multipliers. No finite-width tube needed. Standard for equality-constrained optimization.
- **Riemannian / manifold-aware acquisition optimization:** Parameterize optimization directly on the constraint manifold (tangent space or coordinate chart). Gradient steps stay on the manifold by construction. Potential research contribution.
- **Projected gradient inner loop:** Project the iterate onto the manifold after each scipy step. Simple prototype of a projected gradient method.

### Layer 2 — Post-optimization snapping (the easy fix)

Project each returned candidate onto the manifold before `validate_candidates`. For circles/spheres: O(1) radial scaling. For general nonlinear equalities: a few Newton steps. Directly eliminates the "boundary miss" `ConstraintNotFulfilledError`.

**Suggested next steps (ordered by effort vs impact):**

| Step | Effort | Impact | Notes |
|------|--------|--------|-------|
| Post-optimization projection before validation | Low | High | Eliminates boundary ConstraintNotFulfilledError |
| Adaptive n_restarts for equality domains | Low | Medium | Default 20 -> e.g. 5 for equality; reduces wall time |
| Augmented Lagrangian optimizer for equality domains | High | High | Replaces ±tol slab representation |
| Riemannian acquisition optimization | Very high | Very high | Fundamental improvement; potential research contribution |

---

## Open questions (updated)

- Do real user domains use **multiple nonlinear equalities** (harder than single circle)?
- Is **1e-3** intentionally loose for noisy simulators, or legacy from BoTorch examples?
- Should equalities be **first-class** in docs until manifold samplers exist?
- Is 7-minute `ask()` acceptable for equality-constrained campaigns (long wet-lab cycles), or must it be reduced?
- Does post-optimization projection affect acquisition value consistency (are projected candidates still "good" suggestions)?

---

## 8. Evaluation plan — Track 2 options

Branch: `research/post-opt-snap` (forked from `nonlinearC`).

### What we are measuring

All experiments use FM-3 circle equality `x0²+x1²=0.25` in `[-1,1]²` as the primary stress scenario, and FM-2 ball inequality `||x||²≤0.25` as a control that must not regress.

**Primary metric:** `ok_rate` = fraction of `(seed, ask_n)` pairs where `strategy.ask()` completes without error and the returned candidate satisfies `|f(x)| ≤ tol`.

**Secondary metrics:**
- `ask_s` — wall-clock time for `strategy.ask()` per call
- `max_violation_mean` — mean `|f(x)|` for returned candidates
- `acqf_delta` — acquisition function value before vs after any post-processing (measures candidate quality impact)

**Protocol:** seeds 0–4, `ask_ns = [1, 4]`, `n_initial = 8`, `_validation_tol = 1e-3`. Small seed count intentional — each FM-3 run takes ~7 min with default settings. FM-2 control runs fast (<5s each).

---

### Option A — Post-optimization snap

**Hypothesis:** The optimizer already reaches `|f(x)| ≈ O(1e-3)` in ~7 min. A single Newton step or radial projection at the end would snap `|f(x)|` to `O(1e-9)`, converting most failures into successes at negligible cost.

**Implementation sketch (in `acqf_optimization.py`):**
```python
# After optimize_acqf returns candidates (shape [q, n_dims]):
if _has_nonlinear_equality:
    candidates = _snap_to_manifold(candidates, nonlinear_constraints_local, bounds)
```
where `_snap_to_manifold` runs a small number (≤20) of Newton / gradient-descent steps to drive `|f(x)|` below `1e-6`.

**Evaluation steps:**
1. Baseline (no snap): run harness, record ok_rate, ask_s, max_violation.
2. With snap: same harness, record same metrics + acqf_delta.
3. Compare: did ok_rate improve? Did ask_s change? Did acqf_delta stay near zero?

**Expected outcome:** ok_rate improves significantly (most failures were boundary misses at `|f|≈1e-3`). ask_s unchanged (snap is O(ms)). acqf_delta small (candidate barely moves on the manifold).

**Risk:** If snap moves the candidate significantly (e.g. from the acquisition maximum to a different manifold point), it could reduce candidate quality. Track `acqf_delta` to detect this.

---

### Option B — Reduced default n_restarts for equality domains

**Hypothesis:** `n_restarts=20` × slow scipy = 7 min. Reducing to `n_restarts=5` would cut `ask_s` by ~4× with modest ok_rate impact, since most restarts are redundant on a curved 1D manifold.

**Implementation sketch (in `acqf_optimization.py` or as a domain-aware default in `BotorchOptimizer`):**
```python
# In the IC generator setup path, if equality constraints present:
num_restarts = min(num_restarts, 5)  # cap for equality-constrained domains
```

**Evaluation steps:**
1. Baseline: n_restarts=20, record ok_rate, ask_s.
2. n_restarts=5: same metrics.
3. n_restarts=2 (current test setting): same metrics.

**Expected outcome:** ask_s drops proportionally. ok_rate may drop slightly (fewer restarts = less coverage of acquisition landscape). Key question: is ok_rate with n_restarts=5 comparable to n_restarts=20?

**Note:** This is a workaround, not a fix. The optimizer is still slow per restart; we just run fewer. Best combined with Option A.

---

### Option C — Augmented Lagrangian acquisition optimizer (research)

**Hypothesis:** Replacing the ±tol slab with an AL formulation removes the curved slab entirely. scipy can then minimize the unconstrained AL objective without linesearch failures, and the equality is enforced via the Lagrange multiplier update.

**Implementation sketch:**
- Custom `optimize_acqf` wrapper that iterates: fix multipliers → minimize AL objective with scipy (unconstrained) → update multipliers → repeat until convergence.
- Requires modifying how `optimize_acqf` is called in `acqf_optimization.py`.

**Evaluation steps:** Same as Option A, but also measure convergence (number of AL outer iterations).

**Timeline:** High effort — prototype first, evaluate viability before full implementation.

---

### Execution order

1. **Option A first** — low effort, tests the "easy win" hypothesis. If ok_rate reaches ≥80% on FM-3 with snap, that justifies including it in a future PR.
2. **Option B alongside** — one-line change, easy to measure in the same harness run.
3. **Option C later** — only if A+B are insufficient or if the goal is a research publication.

**Update (2026-06-29):** Option A (snap) was implemented and committed on branch `fix/post-opt-equality-snap`. Option C (AL) was prototyped — see Section 9 below.

### Harness commands for evaluation

```bash
# Baseline (no snap, default n_restarts=20)
python -m bofire.benchmarks.nonlinear_failure_modes.run \
  --mode strategy --scenario fm3_circle_equality_string_d2_r0.5 \
  --seeds-range 0-4 --strategy sobo --ask-ns 1,4 --n-initial 8 \
  --out bofire/benchmarks/nonlinear_failure_modes/_runs/eval_baseline.jsonl

# After Option A implementation (snap enabled)
python -m bofire.benchmarks.nonlinear_failure_modes.run \
  --mode strategy --scenario fm3_circle_equality_string_d2_r0.5 \
  --seeds-range 0-4 --strategy sobo --ask-ns 1,4 --n-initial 8 \
  --out bofire/benchmarks/nonlinear_failure_modes/_runs/eval_snap.jsonl

# FM-2 control (must not regress)
python -m bofire.benchmarks.nonlinear_failure_modes.run \
  --mode strategy --scenario fm2_boundary_ball_shell_d2_r0.5 \
  --seeds-range 0-4 --strategy sobo --ask-ns 1,4 --n-initial 8 \
  --out bofire/benchmarks/nonlinear_failure_modes/_runs/eval_fm2_control.jsonl
```

---

## 9. Augmented Lagrangian (Option C) — Prototype Results

**Status:** Prototype implemented in `bofire/benchmarks/nonlinear_failure_modes/augmented_lagrangian.py`.
**Experiment:** `compare_al_vs_botorch.py` — head-to-head on FM-3 circle equality (`x₀² + x₁² = 0.25`, `dim=2`, `n_restarts=5`, 5 seeds).

### 9.1 Experimental setup

Both methods optimise the same fitted `qLogEI` acquisition function on the same 8 seed experiments (points exactly on the circle manifold). Both use 5 restarts.

- **Method A (BoTorch current):** `optimize_acqf` with the ±`eq_tol = 1e-3` slab encoding + post-opt snap (20 Adam steps at `lr=1e-3`).
- **Method B (AL prototype):** `optimize_acqf_al` — outer loop over 8 AL iterations, inner loop of 100 LBFGS steps with strong Wolfe linesearch (box bounds only).

### 9.2 Results (5 seeds: 42, 7, 13, 99, 123)

| Metric | Method A (BoTorch + snap) | Method B (AL) |
|--------|--------------------------|---------------|
| ok_rate | 5/5 | 5/5 |
| valid_rate (tol=1e-5) | **0/5** | **5/5** |
| mean wall-clock | 8.05s | **2.87s** (2.8× faster) |
| mean \|f(x*)\| | 3.29×10⁻⁵ | **3.24×10⁻⁷** (100× more precise) |

Per-seed breakdown:

| Seed | A elapsed | A \|f\| | A valid | B elapsed | B \|f\| | B valid |
|------|-----------|---------|---------|-----------|---------|---------|
| 42 | 8.19s | 2.50e-05 | FAIL | 2.90s | 2.32e-08 | **PASS** |
| 7 | 7.98s | 2.98e-05 | FAIL | 2.59s | 4.13e-07 | **PASS** |
| 13 | 7.94s | 6.25e-05 | FAIL | 2.07s | 2.68e-07 | **PASS** |
| 99 | 8.02s | 2.88e-05 | FAIL | 3.42s | 2.25e-07 | **PASS** |
| 123 | 8.10s | 1.81e-05 | FAIL | 3.36s | 6.91e-07 | **PASS** |

Acqf values were comparable across both methods (within ~1–2%).

### 9.3 Mechanistic explanation

**Why Method A fails validation (even with snap):**

The snap runs 20 Adam steps at `lr=1e-3`. This reduces `|f|` from ~1e-3 to ~2e-05, but `validate_candidates` uses `tol=1e-5`. Adam has low curvature information and doesn't converge tightly in 20 steps.

**Why AL passes validation naturally:**

The outer loop escalates `μ` geometrically (1 → 4 → 16 → ... → 16,384). At `μ = 16384`, the `(μ/2)f(x)²` penalty dominates the objective, forcing `|f|` toward machine precision. The inner LBFGS with strong Wolfe handles the *unconstrained* (just box-bounded) subproblem without linesearch failures.

**Convergence pattern (representative restart):**

```
outer=0  mu=1e+00  |f|=8e-01  (far from manifold)
outer=2  mu=1.6e+01 |f|=1e-01  (approaching)
outer=4  mu=2.6e+02 |f|=3e-03  (on the scale of eq_tol)
outer=6  mu=4.1e+03 |f|=1e-05  (within validation tolerance)
outer=7  mu=1.6e+04 |f|=2e-07  (converged, deep precision)
```

### 9.4 Open questions for further research

1. **IC diversity:** Our AL restarts use gradient-based projection from random box samples — these cluster near the same manifold basin. BoTorch's `gen_batch_initial_conditions` uses `n_raw_samples` draws with more global spread. Can we combine BoTorch's IC strategy with AL's inner loop?

2. **Inequality + equality mixed domains:** Does AL generalise cleanly to mixed constraints (add inequality penalty `max(0, f(x))²`)? Initial hypothesis: yes, but need to verify on FM-1 (ball inequality) + FM-3 (circle equality) combined scenarios.

3. **Acquisition regret:** In this experiment, AL found comparable or slightly better acqf values than BoTorch. But this is a single acquisition step. Over a full BO loop (20+ iterations), does AL's imprecise global search accumulate regret? Need to run the full strategy harness, not just single-ask comparisons.

4. **Scaling to higher dimensions:** The 2D circle is a 1D manifold. In `dim=10` with a sphere equality, the manifold is 9D and random restarts have more coverage. Does the speedup persist?

5. **LBFGS history depth:** We use default `history_size=100`. On low-dimensional problems, a smaller history may be sufficient.

### 9.5 Path forward

Short term (confirm viability):
- Run full BO loop with AL strategy (not just single-ask comparison) using the harness.
- Test on FM-3 with `dim=3, r=0.5` to check scaling.
- Run FM-2 (ball inequality) to verify AL doesn't regress on inequality-only domains.

Medium term (integration):
- Add AL as an optional optimizer path in `acqf_optimization.py` (activate when `NonlinearEqualityConstraint` present).
- Benchmark against BoTorch default on the full strategy harness (`n_initial=8`, 20 seeds, 5 BO iterations).

Long term (research):
- Compare AL vs. Riemannian gradient descent on the constraint manifold.
- Investigate whether the multiplier `λ` encodes useful information about the optimal point on the manifold (dual variable = shadow price of the constraint).
- Write up as a case study: "Why the ±tol encoding is a liability for second-order optimizers on curved equality constraints."

---

## 10. Mixed constraints — competing gradients analysis (2026-06-30)

### 10.1 Experimental finding

When AL is extended to mixed equality + inequality domains (circle equality `x₀²+x₁²=r²` plus half-plane inequality `x₀+x₁≥0`), equality convergence collapses: `|f|` stays at ~10⁻³ instead of reaching ~10⁻⁷ as in the pure equality case. All three methods (BoTorch, AL-quadratic, AL-rockafellar) fail BoFire validation on the mixed domain in the initial run.

An equality-warmup fix (`eq_warmup_iters=3`: run equality-only outer iterations before activating inequality penalties) partially recovers this — AL-rockafellar reaches 2/5 valid, AL-quadratic 1/5, vs BoTorch 0/5. But the warmup is a scheduling heuristic, not a mathematical fix.

### 10.2 Mathematical analysis of the competing-gradient problem

At any point `x` near the circle `f(x) = x₀²+x₁²−r² = 0`:

- The **equality gradient** `∇f = [2x₀, 2x₁]` points **radially outward** — normal to the manifold.
- The **inequality gradient** for `g = −(x₀+x₁)` is `[−1, −1]` — a fixed direction.

`[−1, −1]` projected onto the radial direction `[x₀, x₁]/r` gives a non-zero component: `(−x₀−x₁)/r`. That component is **normal to the circle** and directly competes with the equality gradient during inner optimization. LBFGS sees both, and neither wins cleanly.

The warmup just delays the conflict. The real fix would make the inequality enforce itself **in the tangential space** of the manifold, leaving the normal direction solely for equality enforcement.

### 10.3 Three mathematically principled approaches

**Option 1 — Riemannian gradient projection:**
At each inner step, after computing the full gradient `∇L`, project out the normal component before applying the step:

```
∇^R L = ∇L − (∇L · n̂) n̂,   where n̂ = ∇f / |∇f|
```

The optimizer then only moves along the manifold. Equality is enforced by a retraction step (projection back onto the manifold) after each gradient step. Inequalities enforce themselves via the tangential component of their gradient — no conflict.

**Option 2 — Null-space parametrisation:**
Reparametrise `x` as a point on the manifold using local coordinates. For the circle, this is simply `x(θ) = [r·cos θ, r·sin θ]` — the optimisation becomes unconstrained in `θ`, and inequalities are handled as box constraints on `θ`. The equality is exact by construction. Generalises to higher dimensions via a local basis for the null space of `∇f^T`.

**Option 3 — ADMM / operator splitting:**
Split into two alternating subproblems — one handles only equality (project to manifold), one handles only inequality and acquisition (optimise along manifold). No competing gradients because they are never in the same objective.

### 10.4 Connection to original research direction

This is exactly what RESEARCH.md Section 7 flagged as the deep fix: *"acquisition function optimisation that explicitly moves along the constraint manifold (Riemannian gradient descent)"*. The AL mixed-constraint experiment exposed the problem concretely — the competing normal gradient is measurable, reproducible, and mathematically characterisable.

### 10.5 Next implementation steps

Prototype **Option 2 (null-space parametrisation)** first — it is the cleanest for the circle/sphere case and reduces to a 1D unconstrained problem, giving a sharp test of whether manifold-awareness eliminates the competing-gradient failure.

Then prototype **Option 1 (Riemannian gradient projection)** — it generalises to arbitrary smooth equality manifolds without needing an explicit parametrisation, making it the most promising path for integration into `acqf_optimization.py`.

Option 3 (ADMM) remains a fallback if both 1 and 2 show convergence issues on non-convex acquisition landscapes.

---

## 11. Manifold-aware optimisers — experimental results (2026-06-30)

### 11.1 Implementation

Two new methods implemented in
`bofire/benchmarks/nonlinear_failure_modes/manifold_optimizer.py`:

**`optimize_acqf_null_space_sphere`** (Option 2 — θ parametrisation):
- For equality `x₀²+x₁²+…=r²` reparametrise as `x(θ)=[r·cos θ, r·sin θ]`.
- Optimise over `θ ∈ ℝ` with Adam, applying inequality penalties in θ-space.
- Equality is exact by construction (no retraction, no multiplier). The equality
  gradient literally does not exist in the optimisation problem.

**`optimize_acqf_riemannian`** (Option 1 — gradient projection + retraction):
- General manifold approach for any smooth `f(x)=0`.
- At each step: project the acquisition+inequality gradient onto the tangent space
  of `f` (null space of `∇f ᵀ`), then update `x` in the projected direction.
- Periodically retract back onto the manifold by minimising `f(x)²` with LBFGS.
- No equality term in the objective — only inequalities.

### 11.2 Results on pure equality (circle, no inequality)

| Method | `|eq|` mean | Valid 5/5 |
|--------|------------|-----------|
| θ parametrisation | **5.5e-18** | ✓ |
| Riemannian GD | 2.2e-7 | ✓ |
| AL (Section 9) | ~1e-7 | ✓ |

All three methods work on the pure equality case. θ achieves floating-point
machine epsilon; Riemannian is ~2e-7 after LBFGS retraction.

### 11.3 Results on mixed domain (circle equality + half-plane inequality)

Setting: `x₀²+x₁²=0.25` plus `−x₀−x₁ ≤ 0` (i.e. `x₀+x₁ ≥ 0`).
Tolerance: eq < 1e-5 AND ineq < 1e-4. 5 seeds, 8 restarts, seed data n=8.

| Method | Valid | `|eq|` mean | ineq mean | acqf (valid) |
|--------|-------|------------|-----------|--------------|
| **θ parametrisation** | **5/5** | **5.5e-18** | **0** | −10.47 |
| Riemannian GD | 4/5 | 1.4e-8 | 7.4e-4 | −11.53 |
| AL-Rockafellar+warmup (Section 10) | **5/5** | 1.8e-6 | 1.3e-6 | **−5.32** |

### 11.4 Key findings

**1. θ parametrisation completely eliminates competing gradients.**
The circle equality is encoded into the parametrisation, not as a penalty term.
At every iterate, `f(x(θ)) = 0` exactly by construction. Inequality penalties
operate purely in θ-space and never corrupt the normal direction of the manifold.
Result: machine-epsilon equality (5.5e-18), zero inequality violation, 5/5 valid.

**2. Riemannian gradient projection partially solves the problem.**
Projecting the gradient onto the tangent space removes the radial component,
so the equality retraction is not disrupted. Equality precision is 1.4e-8 —
excellent. But one seed (seed=3) fails with ineq=3.7e-3.

The Riemannian failure has an exact mathematical cause: the inequality gradient
`∇g = [−1, −1]` is projected onto the tangent space of the circle, giving
`∇g_tangent = [−1, −1] − ((−1·x₀ − 1·x₁)/r²)[x₀, x₁]`.
At the point `(−r/√2, −r/√2)` (most infeasible corner), the tangent is `[1, −1]/√2`,
and `∇g_tangent · tangent = (−1+1)/√2 = 0` — the inequality penalty has **zero
tangential gradient** at the hardest corner. The optimizer stalls there.

**3. AL finds better acquisition values than θ.**
AL-Rockafellar achieves acqf = −5.32 (mean) vs θ's −10.47. The multiplier
dynamics of AL may provide implicit exploration that the pure angular parametrisation
lacks: AL searches over the full x-space while θ is confined to a 1D search.

**4. Precision vs. exploration trade-off.**
θ is maximally precise (machine epsilon) but may get stuck: the 1D θ-space has
a simple landscape but limited restarts coverage. AL searches in 2D (full space)
with higher variance but finds better optima. Riemannian inherits this tension.

### 11.5 Implications for integration

For pure equality constraints (no inequality):
→ **θ / Riemannian** are the natural choice. Both are faster and more precise than AL.

For mixed equality + inequality:
→ **θ dominates** on validity (5/5, machine epsilon) but may miss good acquisition
   values if the feasible arc is complex or the acquisition landscape is multimodal.
→ **AL** is a pragmatic fallback: less precise (1.8e-6) but finds better acqf values.

A hybrid strategy would combine both:
1. Run Riemannian / θ for tight constraint satisfaction.
2. Use AL as a tiebreaker when the θ search stalls in a low-acqf region.

### 11.6 Next steps

1. **Improve θ exploration**: increase restarts, add momentum or simulated annealing
   in θ-space, or use LBFGS over θ (exact gradient via chain rule is available).
2. **Fix Riemannian ineq stalling**: when `|∇g_tangent|` is small, take a larger
   retraction step toward the feasible side, or add a fallback inequality penalty
   only when the tangential component is below a threshold.
3. **Hybrid θ+AL**: run both in parallel, return the candidate with better acqf
   value among those satisfying all constraints.
4. **Generalise θ to ellipsoids**: `Σᵢ aᵢxᵢ² = c` admits the same reparametrisation
   with scaled coordinates — covers many real experimental constraints.
5. **Extend to multiple equalities** (`codim > 1`): use a local frame (QR of Jacobian)
   to reduce to an `(n − k)`-dimensional unconstrained problem.
