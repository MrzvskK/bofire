# T3a — theory pass: is this a paper, and of what kind?

Answers the four questions in [`KICKOFF.md`](KICKOFF.md) §"The job of this chat".
Theory/scoping only — no code (per the hard rule: nothing until Q1 is answered).

---

## Verdict (read first)

**T3a is a paper, but a theory-note-tier one — TMLR / AISTATS, not ICLR/NeurIPS main.**
The theory *half-closes*:

- **A1's `(d−1)`-dim rate is provable — but only in the well-conditioned regime** (`m = inf_M‖∇h‖`
  bounded below, smoothness `ν ≥ 1`). The chicken-and-egg closes as a contraction. It **provably
  fails** (the self-consistency map becomes expansive) exactly in the near-degenerate-Jacobian
  regime that the parent paper targets. So A1 is a theorem whose hypothesis excludes Tier 1.
- **A3 is not a statistical "first."** The bound `dist(x,M) ≲ ε_n/m` and the value-vs-location
  duality are both already in the level-set literature (Willett–Nowak 2007; Cuevas et al. 2006;
  minimax-Hausdorff line). Defensible only as "first in the BO/retraction setting, with a
  matching lower bound."
- **A2's gradient assumption is a convenience** for the black-box regime this paper is about.
  It is realistic for adjoint/autodiff *simulators* — but those are exactly the setting where
  `h` is *known*, so A2's natural home overlaps the regime where T3a's premise is weakest.

**Recommended framing:** "Feasibility under uncertainty" = **A1 (rate, well-conditioned)** +
**A3-as-graceful-degradation (ill-conditioned)** + a **lower bound** tying the residual error to
`m`. The residual concentrates in the degenerate-Jacobian regime — which is the unifying pitch
already in the archive (`../TRACK_T3_unknown_manifolds.md` §2). Synthetic experiments only.

**Recommended sequencing:** behind T1 (primary) **and** behind T3b. T3a and T3b compete for the
"second theory paper" slot; T3b has the cleaner chain (parametric `n^{−1/2}` residual, no
washout) and a concrete applied home (arXiv 2606.08611). **Best single next move:** write up the
A1 self-consistency lemma properly (~2 weeks). If it is crisp, T3a is a standalone note; if it is
messy, fold A3 + the lower bound into T3b and drop A1.

---

## Q1 — Is A1's `(d−1)`-dim active-manifold-regression rate provable?

**Short answer: yes under `m ≳ 1, ν ≥ 1`; no when `m → 0`. The active-allocation step is real,
not hand-waving, and has precedent — but the write-up is an assembly whose one new lemma is the
coupled tube-width recursion.**

### The mechanism (why `(d−1)` is even plausible)

`ε_n ≍ n^{−ν/(2ν+d)}` is the minimax sup-norm rate for learning `h` over a `d`-dim region with a
*uniform* design. We do not need `h` over a `d`-dim region — only on/near `M = {h=0}`, a
`(d−1)`-dim set. Anisotropic local-polynomial regression with a design concentrated in a tube of
normal half-width `w` around `M`:

- tangential bandwidth `b_t` (in `d−1` directions), normal bandwidth `b_n`;
- **bias** `≲ b_t^ν + b_n^ν`;
- **variance** `≲ σ² / (N_loc)` with `N_loc ≍ [n / (w · A)] · b_t^{d−1} · b_n`, `A = vol_{d−1}(M)`.

Set `b_n = w` (use the whole tube). Then `N_loc ≍ n b_t^{d−1} / A`, variance `≍ σ²A/(n b_t^{d−1})`,
and balancing against `b_t^{2ν}`:

```
b_t ≍ n^{−1/(2ν+d−1)}        ⇒   tangential error  ≍ n^{−ν/(2ν+d−1)}
```

plus a **normal-bias floor `w^ν`**. If `w` could be driven to 0, the rate is the intrinsic
`n^{−ν/(2ν+(d−1))}`. This is the standard "regression adapts to intrinsic dimension" phenomenon
(Kpotufe 2011, k-NN local intrinsic dim; Bickel–Li 2007) transported from *data on a manifold*
to *design concentrated near a codim-1 manifold*.

### The chicken-and-egg, made precise

`w` cannot go to 0: the sampling tube must contain the true `M`, and all we know is `M̂ = {μ_h=0}`
with `d_H(M̂, M) ≍ ε_n / m` (plug-in level-set bound, Cuevas–González-Manteiga–Rodríguez-Casal
2006; non-degeneracy `m = inf_M ‖∇h‖ > 0` + positive reach). So `w ≳ c · ε_n/m`.

Round-to-round self-consistency (each round: shrink the tube to the current manifold uncertainty,
resample `n` points in it, refit):

```
ε  ≍  C · n^{−ν/(2ν+d−1)}   +   (ε / m)^ν
        └── intrinsic rate ──┘   └─ normal-bias floor from w ≍ ε/m ─┘
```

- **`ν ≥ 1`, `m ≥ 1`:** `(ε/m)^ν ≤ ε/m ≤ ε`, higher-order for small `ε`. Fixed point at the
  intrinsic rate. **A1 holds:** `ε_n = Θ(n^{−ν/(2ν+(d−1))})`, a genuine improvement over the
  ambient `n^{−ν/(2ν+d)}` (e.g. `ν=2, d=5`: `n^{−1/3}` vs `n^{−2/9}`).
- **`ν = 1`, `m > 1`:** contraction, fixed point `ε ≍ n^{−1/(d+1)}/(1 − 1/m)` — rate holds,
  constant inflates as `m ↓ 1`.
- **`m < 1` (Tier-1 near-degenerate Jacobian):** the map `ε ↦ (ε/m)^ν` is **expansive**;
  the recursion does not contract. Concentrating samples does not buy the intrinsic rate — you
  cannot localize the tube faster than you can localize `M`, and near a degeneracy you cannot
  localize `M`. **A1 fails precisely in the parent paper's hardest tier.**

### Precedent — the active-allocation step is not new, which cuts both ways

- **Willett–Nowak, "Minimax optimal level-set estimation" (2007)** — adaptive partitioning that
  refines only cells meeting the boundary; achieves rates governed by the *boundary's* dimension,
  not ambient `d`. Their error metric *already* combines location error and function-deviation —
  the A3 duality.
- **Shekhar–Javidi, "Multiscale GP Level Set Estimation" (AISTATS 2019)** — GP version;
  hierarchical partitions, effort concentrated near `{f = τ}`; rate depends on boundary
  covering numbers. Needs a margin/noise (Tsybakov) condition = the analogue of `m > 0`; rate
  degrades as the margin vanishes — consistent with the recursion above.
- **Mason–Camilleri et al. (2021), "Nearly optimal algorithms for level set estimation"** —
  experiment-design LSE, near-minimax.

**So:** the `(d−1)`-effective-dimension via adaptive sampling is established for level-set
*classification*. A1's contribution is (i) sup-norm / Hausdorff recovery of `h` near `M` (not
region classification), (ii) the self-consistent tube-width recursion as the vehicle, (iii) the
BO regret consequence. The recursion lemma is the one piece that is not off-the-shelf. A
top-venue reviewer will still call the paper an assembly. **AISTATS/TMLR-realistic, not more.**

### What a proof needs (if pursued)

1. GP sup-norm posterior contraction on a tube around `M` (van der Vaart–van Zanten 2011),
   restricted to the concentrated design — need a design-dependent contraction rate, not the
   uniform-design one. **This is the technical crux and where "assembly" could become "new".**
2. Plug-in level-set / Hausdorff bound with the `1/m` constant (Cuevas 2006; minimax-Hausdorff
   arXiv 1905.02897 for the matching lower bound).
3. The self-consistency fixed-point argument above, stated as: ∃ tube-shrink schedule `w_k` s.t.
   after `K` rounds the estimator achieves `n_K^{−ν/(2ν+(d−1))}` when `m ≥ m_0 > 0`.
4. Constrained GP-UCB *along* `M̂`: regret `≲ √(T γ_T^{(d−1)}) + Σ_t ε_{n_t}` with `γ` at
   dimension `d−1`.
5. **Lower bound:** no estimator achieves `dist(x,M) = o(ε_n/m)` uniformly over the
   non-degeneracy class — makes the `m`-dependence intrinsic, not an artefact.

---

## Q2 — Is A3 ("control feasibility *distance*, not *value*") a genuine "first"?

**No, not as a statistical statement. Yes, narrowly, as a BO-setting claim with a lower bound.**

Already in the literature:

| Piece | Where |
|---|---|
| `d_H({ĥ=0}, {h=0}) ≲ ‖ĥ−h‖_∞ / inf‖∇h‖` | Cuevas–González-Manteiga–Rodríguez-Casal 2006 (plug-in level sets); Chen–Genovese–Wasserman (ridges/level sets); minimax-optimal Hausdorff version arXiv 1905.02897 |
| The value-vs-location duality as a combined error metric | Willett–Nowak 2007 (their metric penalizes *both* location error and `|f − τ|`) |
| Gradient/margin (`1/‖∇f‖`, Tsybakov margin) as the governing constant | Standard across LSE theory |

**What is plausibly new (and defensible):**

1. **Retraction preserves the bound.** Show the parent paper's conditioning-aware (pseudo-inverse
   Newton) retraction applied to `μ_h` produces an iterate with `dist(x, M) = O(ε_n/m)` — i.e.
   landing exactly on `{μ_h = 0}` loses nothing versus the statistical limit, and *penalty*
   methods (EPBO, CUQB) control `|h|` but leave `dist(x,M)` at `O(|h|/m)` with a *worse*
   constant because they stop at `|μ_h| ≍ |∇f|/(ρ‖∇μ_h‖) > 0`.
2. **BO trajectory consequence:** cumulative feasibility-*distance* regret bound.
3. **Lower bound** (Q1 step 5) making `1/m` unavoidable → "the washout concentrates in the
   degenerate-Jacobian regime" becomes a theorem, not an observation.

**Recommendation:** drop the bare word "first." Claim: *"the first feasibility-distance guarantee
for an estimated equality constraint in Bayesian optimization, with a matching lower bound,"* and
cite Cuevas / Willett–Nowak explicitly as the statistical origin. Anything stronger will draw a
correct reviewer rebuttal.

---

## Q3 — Does A2's "simulator provides `∇h`" match a real benchmark, or is it a convenience?

**Realistic for simulators; a convenience for this paper's regime.**

- **Genuinely available:** adjoint PDE solvers, autodiff simulators, aircraft-trim residuals,
  photonic/topology inverse design — all yield `∇h` at ~1× the cost of `h` (archive §5A).
  Gradient-enhanced GP / co-kriging is a well-established constant-factor variance reduction
  (`d+1` scalars per eval). Newton retraction wants `∇h` anyway.
- **The catch (archive §5A takeaway):** in those same simulator settings `h` is almost always
  *known in closed form* — conservation laws, discretized PDE state equations, equilibrium
  forms. The fully-unknown-functional-`h` regime that A1 needs rarely comes with a cheap
  adjoint. For a genuinely black-box expensive `h` you get **one scalar per eval, no gradient**.
- **Impact on the theory:** A2 shrinks the *constant* in `ε_n`, not the *rate*. It does not
  change the `(d−1)` story. So A2 is legitimately a **secondary / "if available" contribution**,
  not load-bearing. Do not let a theorem depend on it.

**Concrete matches if a gradient-enhanced variant is written:** parent paper's **P5** (LMTD
constraint) and **P6/P7** (equilibrium equalities) — hide a coefficient behind a noisy oracle,
expose `∇h` via autodiff of the known form. But that is the **grey-box (T3b)** setting, not
nonparametric. In pure nonparametric T3a, A2 needs a bespoke synthetic "simulator with adjoint
but unknown functional form" — defensible but visibly artificial.

---

## Q4 — Paper skeleton + synthetic experiment plan (conditional on the A1 lemma being clean)

### Skeleton — "Feasibility Under Uncertainty: Retracting onto an Estimated Constraint Manifold"

1. **Intro.** Known-`h` retraction (parent paper) → `|h| ~ 1e-13`. Unknown `h`: plug-in
   retraction capped at `ε_n`. Contributions: (C1) active manifold-restricted estimation →
   intrinsic `(d−1)` rate when well-conditioned; (C2) feasibility-*distance* guarantee for the
   retracted iterate + lower bound; (C3) the residual concentrates in the degenerate-Jacobian
   regime, where the method degrades gracefully and penalty methods fail outright.
2. **Setup.** `h` unknown/expensive, GP posterior; `M`, `M̂`; non-degeneracy `m`, positive
   reach; the two error sources (penalty bias — removed by retraction; `ε_n` — the target).
3. **Warm-up: plug-in retraction.** `|h(x)| = O(ε_n)`, `dist(x,M) = O(ε_n/m)`. Contrast with
   finite-`ρ` penalty stopping at `|∇f|/(ρ‖∇μ_h‖)`. (Mostly transplant + Cuevas cite.)
4. **A1 — active manifold-restricted estimation.** Anisotropic local regression in a tube;
   self-consistent tube-width recursion; Theorem 1: under `m ≥ m_0`, `ν ≥ 1`, a tube-shrink
   schedule attains `ε_n = Θ(n^{−ν/(2ν+(d−1))})`. Remark: recursion expansive for `m < ...`.
5. **A3 — feasibility distance + lower bound.** Theorem 2: retracted iterate has
   `dist(x,M) = O(ε_n/m)`, matched by Theorem 3 (lower bound: `Ω(ε_n/m)` over the
   non-degeneracy class).
6. **BO regret.** Constrained GP-UCB along `M̂`: `≲ √(T γ_T^{(d−1)}) + Σ_t ε_{n_t}`.
7. **A2 (short).** Gradient-enhanced GP: constant-factor `ε_n` reduction; `∇h`-consistency for
   Newton retraction. Explicitly "when available."
8. **Experiments (synthetic).**
9. **Limitations.** Degenerate-Jacobian regime (no rate, only graceful degradation);
   thin current-practice motivation for pure nonparametric `h` (be honest — cite §5).

### Experiment plan (synthetic only)

**Benchmark family — "hidden equality":** take each of the parent paper's P1–P7 equality
constraints `h*(x)`, expose it only through a noisy oracle `y = h*(x) + η`, `η ~ N(0, σ²)`.
Objective `f` from the same problem. Tier label (1/2/3 by `m = inf_M‖∇h*‖`) carried over —
the Tier axis is the story.

**Methods:**

| Method | Constraint handling |
|---|---|
| **AMR-R** (ours) | active tube sampling + Newton retraction onto `{μ_h=0}` |
| naive M-R | Newton retraction onto `{μ_h=0}`, uniform constraint design (no active tube) |
| penalty-on-surrogate | `min f + (ρ/2) μ_h²`, moderate `ρ` |
| **EPBO** (Lu–Paulson 2024) | exact-`ℓ1` penalty merit on GP mean |
| **CUQB** (Paulson et al. 2023) | two-sided quantile bound |
| (opt.) BE-CBO band analogue | `|μ_h| ≤ κσ_h` acceptance band |

**Metrics vs constraint-eval budget `n`:**

1. **true `|h*(x_n)|`** at the returned point — log scale; expect AMR-R to track `n^{−ν/(2ν+d−1)}`,
   naive M-R `n^{−ν/(2ν+d)}`, penalties to plateau.
2. **true `dist(x_n, M)`** (numerically, via projection onto `h*=0`) — the A3 quantity; break
   out **by Tier**. Headline plot: distance-error vs `1/m` across P1–P7 at fixed budget.
3. **feasibility rate** at `τ ∈ {1e-2, 1e-3, 1e-5}`.
4. **simple regret** on `f`.
5. **wasted-infeasible-eval count**.
6. **rate-fit**: log-log slope of metric 1 vs `n`, with a bootstrap CI, against the predicted
   exponents for `d ∈ {3,5,8}` — the plot that verifies Theorem 1.
7. **ablation**: tube-shrink schedule on/off; `b_n` fixed vs adaptive.

**Scales:** `d ∈ {3,5,8}`, `ν` via Matérn-3/2 vs 5/2 truth, 20–40 seeds, budget to `n ~ 200–400`
constraint evals. Synthetic `h*` with a tunable `m` (e.g. `h*(x) = m·g(x)` with `g` fixed) to
sweep the well-conditioned → degenerate axis continuously and show the recursion's predicted
breakdown.

**Success criterion for the paper:** metric-6 slopes match `−ν/(2ν+d−1)` for AMR-R and
`−ν/(2ν+d)` for naive M-R (Tier 3), *and* AMR-R degrades gracefully (no blow-up, beats EPBO/CUQB
on metric 2) at Tier 1 while the rate claim is explicitly withdrawn there.

---

## Bottom line for the user

1. **Q1:** A1 provable in the well-conditioned regime as a contraction; provably breaks at
   `m → 0`. New content = the tube-width recursion + design-dependent GP contraction. Assembly
   risk is real → AISTATS/TMLR ceiling.
2. **Q2:** A3 is not a statistical first (Willett–Nowak 2007, Cuevas 2006). Salvageable as
   "first in BO, with a matching lower bound." Drop the bare "first."
3. **Q3:** A2 realistic for simulators, but simulators have known `h`. Keep it secondary,
   non-load-bearing.
4. **Q4:** Skeleton + synthetic plan above. Motivation stays thin (per §5) → position as theory.
5. **Sequencing:** T3a is a paper, but rank it **after T1 and T3b**. Cheapest decisive step:
   spend ~2 weeks writing the A1 self-consistency lemma. Clean ⇒ standalone TMLR note.
   Messy ⇒ move A3 + lower bound into T3b, drop A1.

## Sources

- [Willett & Nowak, Minimax optimal level-set estimation (2007)](https://www.ncbi.nlm.nih.gov/pubmed/18092596)
- [Shekhar & Javidi, Multiscale GP Level Set Estimation (AISTATS 2019)](https://proceedings.mlr.press/v89/shekhar19a.html)
- [Mason, Camilleri et al., Nearly Optimal Algorithms for Level Set Estimation (2021)](https://arxiv.org/pdf/2111.01768)
- [Cuevas, Plug-in estimation of general level sets (2006)](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-842X.2006.00421.x)
- [Minimax Hausdorff estimation of density level sets (arXiv 1905.02897)](https://arxiv.org/abs/1905.02897)
- [Kpotufe, k-NN Regression Adapts to Local Intrinsic Dimension](https://www.columbia.edu/~skk2175/Papers/kNNRegressionLocRatesFullVersion.pdf)

## Log

- 2026-09-03 — theory pass on Q1–Q4 (this doc). Verdict: TMLR/AISTATS-tier theory note,
  sequence after T1 and T3b; next step = write the A1 self-consistency lemma.
