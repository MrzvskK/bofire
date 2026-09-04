# T3a — novelty verification (two-stage sweep)

Stage 1 = Bayesian-optimization community. Stage 2 = general optimization / statistics.
Sweep date 2026-09-04. Complements the archive prior-art table
([`../TRACK_T3_unknown_manifolds.md`](../TRACK_T3_unknown_manifolds.md) §4) — this doc is
narrower (the three T3a claims) and refreshed to 2025–26.

## The three claims under test

- **C1** — active, manifold-restricted estimation of an unknown equality function `h` →
  intrinsic `(d−1)`-dim rate `n^{−ν/(2ν+(d−1))}` (vs ambient `n^{−ν/(2ν+d)}`), feeding
  retraction of acquisition candidates onto `{μ_h = 0}`.
- **C2** — a feasibility-*distance* guarantee `dist(x, M) = O(ε_n/m)` for the retracted iterate,
  with a matching lower bound.
- **C3** — the residual error concentrates in the degenerate-Jacobian regime (`m = inf_M‖∇h‖
  → 0`); the method degrades gracefully there while penalty methods fail outright.

---

## Stage 1 — Bayesian optimization community

### Nearest work

| Work | What it does | Overlap with T3a | Blocks? |
|---|---|---|---|
| **EPBO** — Exact-Penalty BO (Lu & Paulson, Technometrics 2024) | exact-`ℓ1` penalty merit on GP means; black-box **equalities**; can start infeasible | same problem class (unknown equality in BO); penalty-on-surrogate, **no retraction, no manifold, no rate** | No — primary baseline/foil |
| **CUQB** (Paulson et al. 2023) | deterministic quantile-bound acqf; equalities via two-sided bounds | same class; no retraction, no manifold parametrisation, no dimension-specific rate | No — baseline |
| **COBALt** — Constrained BO w/ Adaptive Active Learning (arXiv 2310.08751) | GP surrogates + confidence sets; adaptively trades constraint active-learning vs objective BO; **simple-regret** bound with standard `γ_T` | *closest on "active learning of constraints"*; **inequality only**, feasible-*region* classification, no manifold retraction, **theory has no intrinsic/boundary dimension** | No — cite as the BO active-constraint-learning cousin |
| **COMBOO** / Constraint Active Search (Malkomes 2021; arXiv 2310.08751 family) | inequality "regions of satisfaction", coverage | inequality regions, not equality-manifold landing | No |
| **"BO: Which Constraints Matter?"** (arXiv 2512.17569, Dec 2025) | decoupled constraint eval; identifies *binding* constraints; empirical only | budget allocation across constraints; **inequality only, no retraction, no theory** | No — cite (most recent on constraint-eval budgeting) |
| **Entropy-based active constraint learning** (npj Comp. Mater. 2023); **BO-ACL** (IISE Trans. 2025) | Bayesian classification of feasible boundary, entropy acqf | inequality feasibility boundary, classification; no equality manifold, no retraction | No |
| **LCBO** — Local Constrained BO (arXiv 2603.07965, ICML 2026) | local BO, `‖c‖²` penalty on surrogate, asymptotic-KKT; released code inequality-only | penalty family; **unreproducible** (see D2_PRESTUDY) | No — foil |
| **In-BO** — Intrinsic BO on Complex Constrained Domain (arXiv 2301.12581); Extrinsic BO on Manifolds (2212.13886); Graph-GP BO on manifolds (2210.10962) | BO with a GP built **on a known manifold / known complex domain** | "manifold" + "BO" keyword overlap; geometry is **known and fixed**, not learned from constraint data; no equality constraint being estimated | No — orthogonal (kernel layer, known geometry) |
| **Safe BO** (SafeOpt family; "Safe BO under Unknown Constraints", IEEE 2020) | unknown inequality **and equality** constraints, reformulated as **probability-of-violation** bounds; safe-set expansion | handles unknown equalities — but as a *probabilistic* acceptance band, **no retraction onto `{μ_h=0}`, no manifold estimation rate**, discretised domain | No — cite as the "equality via a `\|μ_h\|≤κσ_h` band" precedent |
| **BE-CBO** — Boundary Exploration CBO (Tian et al., ICML 2024) | **binary** feasibility, NN-ensemble, explores feasible/infeasible boundary with band `l(x)=0.5−σ_E(x)` | its band is the analogue of an acceptance band, but for a *classification* boundary of a full-dim set, not a regression zero-level-set; no retraction | No — cite as the analogue (co-author J. Dürholt = BoFire/BASF) |
| **GPIS / COGIS** (arXiv 2410.00157, robotics) | end-effector retracted onto a GP implicit surface `{μ(x)=0}` | **the only work that retracts onto a learned zero-level-set** — but real-time control, **no acquisition function, no evaluation budget, no rate** | No — mild precedent for the retraction primitive, not a competitor |

### Stage 1 verdict

- **The triple {unknown functional `h` + hard equality + retraction of optimization iterates
  onto `{μ_h=0}` + evaluation budget} is unoccupied in BO.** Confirmed a second time.
- **C1's *use* (active equality-constraint learning driving retraction) is new to BO.** COBALt is
  the closest and is inequality-region classification with no dimension-aware theory.
- **C1's *mechanism* is not new** — see Stage 2 (active-learning minimax theory). This caps the
  contribution's perceived novelty.
- **C2 in BO:** no constrained-BO paper gives a feasibility-*distance* (vs constraint-*value* or
  probability-of-feasibility) guarantee for an estimated equality. Genuinely open in BO.
- **C3:** no BO paper formalises the degenerate-Jacobian regime for an estimated equality. Open.

---

## Stage 2 — general optimization / statistics

### Nearest work

| Work / field | What it does | Overlap with T3a | Blocks? |
|---|---|---|---|
| **RBDO with adaptive Kriging** — EGRA (Bichon 2008), AK-MCS (Echard 2011), AK-RBDO, single-/double-loop adaptive-Kriging RBDO (2023–25) | build a Kriging surrogate of a limit-state `g`; **actively add samples near `{g=0}`** via a learning function (EFF / U / H); solve a design optimisation s.t. a **chance constraint** `P[g≤0] ≤ p_f` | **the general-opt precedent for active sampling near a learned zero-level-set.** But: (i) chance constraint on a **full-dimensional failure region**, not a hard equality defining a measure-zero manifold; (ii) **no retraction** — the RBDO optimum keeps a safety margin, never lands on `{g=0}`; (iii) learning function refines the surrogate, it is **not an objective optimised along the manifold**; (iv) **empirical — no convergence-rate / dimension theory**, heuristic stopping | No — **the #1 Stage-2 citation**; differentiate on hard-equality + retraction + rates |
| **Active-learning minimax theory** — Castro–Willett–Nowak "Faster Rates in Regression via Active Learning" (NIPS 2005); Castro–Nowak "Minimax Bounds for Active Learning" (IEEE IT 2008); Minsker 2012; k-NN active learning under local smoothness (Kpotufe/Njike; Locatelli et al. 2017–19) | active sampling concentrated near a `(d−1)`-dim **boundary fragment** `x_d = φ(x_1..x_{d−1})` → excess-risk / regression rates whose exponent carries `(d−1)`, strictly faster than passive; requires a **margin condition** (analogue of `m > 0`) | **this *is* C1's mechanism.** Boundary-fragment active regression already turns ambient `d` into `(d−1)` under a margin condition, and rates **degrade when the margin vanishes** — exactly the C3 story | **Partially** — C1's rate phenomenon is **not novel**. T3a must claim only: sup-norm/Hausdorff recovery of a regression zero-level-set *for retraction*, the self-consistent tube-width coupling (`w ≍ d_H(M̂,M) ≍ ε_n/m`), and the BO-regret consequence |
| **Plug-in level-set / manifold estimation** — Cuevas–González-Manteiga–Rodríguez-Casal 2006; Willett–Nowak "Minimax optimal level-set estimation" 2007; minimax-Hausdorff (arXiv 1905.02897); Chen–Genovese–Wasserman (ridges) | rates for `d_H({ĥ=0}, {h=0})` vs `‖ĥ−h‖_∞ / inf‖∇h‖`; Willett–Nowak's metric **already combines** location error and `|f−τ|` | **this *is* C2's statistical content.** `dist ≲ ε_n/m` and the value/location duality both pre-exist | **Partially** — C2 is **not a statistical first**. Salvage: "first feasibility-distance guarantee *for a retracted optimisation iterate*, with a matching lower bound in the BO setting" |
| **Inexact-TR PDE-constrained optimisation** — Heinkenschloss–Ridzal; Kouri–Ridzal–van Bloemen Waanders; Zahr (arXiv 2405.14827); ROM/neural-operator trust regions | `h = R(u,x)=0` (**known**, discretised PDE), solved **inexactly**; trust region drives inexactness → 0 with globally convergent theory | same shape ("optimise subject to an inexactly-satisfied equality"); but `h` **known in form**, inexactness is **numerical and controllable on demand**, not statistical and sample-limited; no GP, no acquisition | No — cite as "the analogous framework when inexactness is numerical, not statistical" |
| **Constrained DFO — feasible / inexact-restoration methods** (arXiv 2402.11920 2024; 2401.08277 2024; SQP-without-derivatives) | derivative-free in the objective, **constraint derivatives known**; feasibility restoration steps | derivatives of the constraint are assumed known → not our setting | No |
| **Stochastic approximation on manifolds** — Riemannian Robbins–Monro (Mertikopoulos et al., COLT); Riemannian stochastic approximation (arXiv 2206.06795) | Robbins–Monro with vector addition replaced by retraction / exp map, on a **known** manifold | machinery precedent for **A4** (retract→evaluate→update as stochastic root-finding); manifold is known there, here it is being **estimated and refined** | No — cite for A4 machinery |
| **Manifold sampling** — Larson–Menickelly–Wild (arXiv 2011.01283 and predecessors) | DFO for nonsmooth `h∘F`; tracks **activity manifolds** of the nonsmooth outer function inside a trust region | **terminology collision only** — "manifold" = pieces where `h` is smooth, nothing to do with a constraint surface | No — cite once to disambiguate the term |
| **Guaranteed-feasibility surrogate optimisation** (arXiv 2107.10190) | strictly-feasible candidates when constraint functions are **cheap** | requires cheap constraints; no rate for an estimated equality | No |

### Stage 2 verdict

- **C1's rate mechanism is prior art** (Castro–Nowak boundary-fragment active learning). T3a's
  defensible C1 novelty shrinks to: (a) sup-norm recovery of a regression zero-level-set as the
  target quantity, (b) the **self-consistent tube-width fixed-point** analysis (design region
  tied to current `M̂` uncertainty) — this is the one lemma with no off-the-shelf form, (c) the
  GP + BO-regret wrapper.
- **C2's statistical bound is prior art** (Cuevas 2006; Willett–Nowak 2007). Defensible novelty:
  the retraction-preserves-the-bound theorem + BO trajectory regret + the lower bound making
  `1/m` intrinsic.
- **C3 is the least-occupied claim.** RBDO/adaptive-Kriging degrades near a flat limit state but
  nobody formalises "the residual concentrates in the degenerate-Jacobian regime, and here is a
  lower bound." The active-learning margin condition is the closest, and it is stated as an
  *assumption*, not analysed as a *regime*. **Make C3 central.**

---

## Consolidated novelty statement (what to claim, what to drop)

**Keep / lead with:**

1. **The setting**, framed for optimisation: retracting acquisition candidates onto an
   *estimated* hard-equality manifold under an evaluation budget — unoccupied in BO and in
   general optimisation (RBDO never retracts, PSE inexact-TR has `h` known).
2. **C3** — degenerate-Jacobian regime: graceful-degradation bound + lower bound tying the
   residual to `m`. Ties directly to the parent paper's Tier-1 contribution.
3. **The self-consistent tube-width lemma** as the technical core of C1.
4. **C2 reframed** — "first feasibility-*distance* guarantee for a retracted optimisation
   iterate, with a matching lower bound," citing Cuevas / Willett–Nowak as the statistical
   origin.

**Drop / soften:**

- Any bare "first to …" on C1's `(d−1)` rate or C2's distance bound. Cite Castro–Nowak and
  Cuevas explicitly and position T3a as *transporting* those to constrained BO + coupling them
  through retraction.
- "novel active learning of the constraint" — COBALt + the RBDO field own active constraint
  sampling. T3a's active learning is *for retraction onto an equality*, which is the narrow new
  part.

**Mandatory citations (or a reviewer will supply them):** Castro–Willett–Nowak 2005;
Castro–Nowak 2008; Cuevas et al. 2006; Willett–Nowak 2007; EGRA (Bichon 2008); AK-MCS (Echard
2011); EPBO (Lu–Paulson 2024); COBALt (2310.08751); Safe-BO-under-unknown-constraints;
Larson–Menickelly–Wild (term disambiguation); inexact-TR PDE-constrained (Kouri–Ridzal / Zahr).

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Reviewer: "C1 is Castro–Nowak boundary-fragment active learning re-skinned" | **High** | Lead with C3 + the tube-width lemma; frame C1 as a corollary/transport, not the headline |
| Reviewer (Stage 2 / PSE): "this is adaptive-Kriging RBDO for the limit state" | **High** | The hard-equality + retraction + rate distinction, stated up front with an RBDO comparison paragraph |
| Reviewer: "C2 distance bound is Cuevas 2006" | Medium | Reframe to the retracted-iterate + lower-bound claim; cite Cuevas |
| Motivation thin for pure nonparametric `h` (archive §5) | Medium | Position as theory (TMLR/AISTATS); synthetic experiments; do not oversell an applied home |
| A1 recursion does not close cleanly for `ν < 1` or non-product smoothness | Medium | State the theorem for `ν ≥ 1`, `m ≥ m_0`; the `m→0` case becomes C3 (a feature) |

## Bottom line

Novelty **survives, at TMLR/AISTATS altitude**, and only if repositioned: **C3 leads**, C1 and
C2 are transports of known statistical results (Castro–Nowak; Cuevas / Willett–Nowak) into the
constrained-BO + retraction setting, glued by the self-consistent tube-width lemma. The
headline *setting* (retraction onto an estimated equality manifold for optimisation) is
genuinely unoccupied in both communities — RBDO/adaptive-Kriging is the closest general-opt
neighbour and does not retract, does not do hard equalities, and has no rate theory.

## Sources

- [Castro, Willett, Nowak — Faster Rates in Regression via Active Learning (NIPS 2005)](https://rmcastro.win.tue.nl/publications/nips05_active.pdf)
- [Castro & Nowak — Minimax Bounds for Active Learning (IEEE IT 2008)](https://rmcastro.win.tue.nl/publications/castro_IT_minimax.pdf)
- [Cuevas — Plug-in estimation of general level sets (2006)](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-842X.2006.00421.x)
- [Willett & Nowak — Minimax optimal level-set estimation (2007)](https://www.ncbi.nlm.nih.gov/pubmed/18092596)
- [Minimax Hausdorff estimation of density level sets (arXiv 1905.02897)](https://arxiv.org/abs/1905.02897)
- [COBALt — Constrained BO with Adaptive Active Learning of Unknown Constraints (arXiv 2310.08751)](https://arxiv.org/html/2310.08751)
- [Bayesian Optimisation: Which Constraints Matter? (arXiv 2512.17569)](https://arxiv.org/html/2512.17569v1)
- [AK-MCS — Echard et al. 2011](https://www.sciencedirect.com/science/article/abs/pii/S0167473011000038)
- [RBDO using adaptive Kriging — single-loop / double-loop (2023)](https://www.sciencedirect.com/science/article/abs/pii/S0951832023003009)
- [Manifold Sampling for Nonsmooth Nonconvex Compositions (arXiv 2011.01283)](https://arxiv.org/abs/2011.01283)
- [Riemannian stochastic approximation algorithms (arXiv 2206.06795)](https://arxiv.org/pdf/2206.06795)
- [Inexact Trust-Region Methods for PDE-Constrained Optimization (Kouri–Ridzal)](https://link.springer.com/chapter/10.1007/978-1-4939-8636-1_3)
- [A Feasible Method for Constrained Derivative-Free Optimization (arXiv 2402.11920)](https://arxiv.org/pdf/2402.11920)
- [Safe Bayesian Optimization under Unknown Constraints (IEEE 2020)](https://ieeexplore.ieee.org/document/9304209/)

## Log

- 2026-09-04 — two-stage novelty sweep. Verdict: novelty survives at TMLR/AISTATS if C3 leads;
  C1/C2 are transports of Castro–Nowak / Cuevas–Willett–Nowak, not firsts. Headline setting
  unoccupied in both communities; RBDO adaptive-Kriging is the closest Stage-2 neighbour and
  does not retract / does no hard equality / has no rate theory.
