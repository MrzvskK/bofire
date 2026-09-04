# Track T3 — Feasibility under uncertainty: retracting onto an estimated constraint manifold

**Status:** research direction, secondary to [T1](TRACK_T1_mixed_manifolds.md). The user
liked the direction; this doc is the consolidated framing (problem + program). Next move: the
user decides how to proceed.

**Parent context:** [`../D2_PRESTUDY.md`](../D2_PRESTUDY.md) (Thread 3), submitted paper
`../report/MANIFOLD_OPTIMISERS_PAPER.tex`, [`../PROJECT_SUMMARY.md`](../PROJECT_SUMMARY.md).

---

## 1. The problem (what the original T3 framing identified)

The submitted paper retracts acquisition candidates onto a constraint manifold `M = {h(x)=0}`
to **machine precision**, using Newton retraction — but only when `h` is **known in closed
form**. Ask what happens when `h` is *not* known exactly and must be learned from data.

Model `h` with a GP, posterior `(μ_h, σ_h)` after `n` (noisy, expensive) constraint
evaluations. True manifold `M = {h=0}`, estimate `M̂ = {μ_h = 0}`.

**The returned candidate's true infeasibility has two error sources:**

| Source | Size | Fixed by retraction? |
|---|---|---|
| **Penalty bias** — a finite-`ρ` quadratic penalty `min f + (ρ/2)μ_h²` stops at `\|μ_h\| ≈ \|∇f\|/(ρ‖∇μ_h‖)` | grows as `‖∇μ_h‖ → 0` (degenerate Jacobian); `ρ→∞` triggers the `μ_h²` false-convergence trap | **Yes** — landing exactly on `{μ_h=0}` removes it. This is the parent paper's contribution, transplanted to the estimated `μ_h`. |
| **GP estimation error** `ε_n = ‖μ_h − h‖_∞` on a tube around `M` | nonparametric: `ε_n ≍ (n/\log n)^{−ν/(2ν+d)}` — slow (`ν=2, d=5` ⇒ `n^{−2/9}`) | **No** — retraction onto a biased `μ_h` lands on a systematically wrong manifold. |

So naive plug-in retraction is **capped at `ε_n`**. The parent paper's `|h| ~ 1e-14` headline
does not transfer for free. **`ε_n` — and what to do about it — is the research problem.**

Note the two error sources are not symmetric: retraction already beats a *practical*
(finite-`ρ`) penalty by the whole penalty-bias term, and that term is largest exactly in the
degenerate-Jacobian regime the parent paper targets. The open question is whether we can also
push past `ε_n`.

---

## 2. The research program (how to beat `ε_n`)

Ordered strongest-first. Not scooped — the prior-art sweep (§4) confirms **nobody retracts
onto a learned equality manifold for optimization at all**.

### A1 — Active learning of `h` along the retraction path → intrinsic `(d−1)`-dim rate

`ε_n ≍ n^{−ν/(2ν+d)}` is the rate for learning `h` over a `d`-dimensional region. We only need
`h` accurate **on and near `M`**, a `(d−1)`-dimensional set. The retract → evaluate → update
loop *already* concentrates constraint evaluations on `M̂`; formalise this as
**manifold-restricted active regression** and the rate should improve to `n^{−ν/(2ν+(d−1))}`.

- **This is [T1](TRACK_T1_mixed_manifolds.md)'s intrinsic-dimension argument on the
  *estimation* side** — one principle (intrinsic beats ambient dimension), two instances
  (acquisition search in T1; constraint estimation here).
- **Analysis risk:** chicken-and-egg — need `M̂` to know where to sample, need samples for
  `M̂`. Precedent exists (active level-set estimation: Willett–Nowak; Shekhar–Javidi 2019) but
  it is delicate. **This is the pivotal open question — needs its own focused pass.**

### A2 — Gradient-enhanced constraint surrogates

If the constraint comes from a simulator that also yields `∇h` cheaply (adjoint solvers,
autodiff sims), gradient-enhanced GP / co-kriging shrinks `ε_n` sharply (`d+1` scalars per
evaluation). Newton retraction *wants* `∇h` anyway. Narrows scope to grey-box / simulator
settings — which is where the motivation is (§5).

### A3 — Control feasibility *distance*, not constraint *value*

`dist(x, M) ≤ ε_n / ‖∇h‖`. When `‖∇h‖` is small — the Tier-1 degenerate-Jacobian regime — the
*distance* error blows up even with `|h|` controlled. **The `ε_n` washout is worst exactly
where the parent paper's contribution matters most.** A method that (a) estimates `∇h` well
near `M` (A2) and (b) retracts with the parent paper's conditioning-aware pseudo-inverse step
could be **the first method to control feasibility *distance* for an estimated equality
constraint** — a crisp "first" claim.

- **Novelty check needed:** the *statistical* piece (Hausdorff distance `d_H(M̂, M)` vs
  `ε_n/‖∇h‖`) exists in plug-in level-set / manifold estimation theory (Cuevas–Walther;
  Genovese et al.). Novelty = *using* it for retraction + the conditioning connection + the BO
  regret consequence. Verify against that literature before claiming "first."

### A4 — retract → evaluate → update as GP-Newton stochastic root-finding

Don't learn `h` globally. Treat the loop as a Newton-like iteration on an unknown function:
each expensive eval sits on `M̂`, maximally informative about local manifold position; near
`x*` it pins `h` down fast (local polynomial / `n^{−1/2}` or better). Connects to stochastic
root-finding (Robbins–Monro; GP root-finding) — combining it with an acquisition objective and
a manifold is new.

### Unifying pitch

> **Feasibility under uncertainty.** Plug-in retraction onto an unknown constraint manifold is
> capped at the GP error `ε_n`. We show: (A1) allocating constraint evaluations along the
> retraction path turns this into an intrinsic `(d−1)`-dimensional estimation problem; (A3)
> with gradient information, conditioning-aware retraction controls feasibility *distance*, not
> just constraint *value*; and the residual error concentrates in the degenerate-Jacobian
> regime, where the method degrades gracefully and penalty methods fail outright.

**Theoretical spine:** A1 + A3. **Effort if pursued as the primary bet:** ~3 months.

---

## 3. Two regimes

The program above applies differently depending on how `h` is unknown.

### 3a. Nonparametric — `h` a fully-unknown black-box function, GP-modelled

- **Where A1 lives** — the `(d−1)`-vs-`d` rate story only exists when `h` is a genuine
  function, not a finite parameter family.
- **Motivation is thin** (see §5): cross-field search found that hard physical equalities are
  almost always *known* (conservation laws, equilibrium forms, PDE state equations) —
  practitioners already retract exactly. The genuinely-unknown-functional-`h` + hard-equality
  + BO-budget + no-gradients combination is rare in current practice. Whether that is a true
  absence or a chicken-and-egg artefact of no method existing is unresolved — but it means
  this regime, as a standalone applied paper, is hard to place (workshop / TMLR).
- Could still support a **theory-only paper** ("feasibility under uncertainty", A1 + A3,
  synthetic experiments) if the A1 rate closes.

### 3b. Grey-box — `h(x; θ_h)` known in form, unknown parameters `θ_h`

- **Retraction stays exact in `x`** — land exactly on `{h(·; θ̂)=0}` using the parent paper's
  Newton retraction verbatim, with `θ̂` plugged in.
- **Residual is parametric:** `|h(x;θ̂) − h(x;θ*)| ≈ ‖∂h/∂θ‖·‖θ̂−θ*‖ = O_p(n^{−1/2})` — much
  faster than the nonparametric `ε_n`. No washout.
- **A1 does not apply** (finite parameter vector, not a function). **A3 does** — `dist(x,M)`
  still degrades as `‖∇h‖ → 0`, so the conditioning story carries.
- **Concrete home + weak foil exist** (§5C): `arXiv:2606.08611` does exactly this with a crude
  residual-penalty on a ~12-experiment budget.
- **Cleaner theory chain:** MLE/posterior contraction of `θ̂` (`n^{−1/2}`) → delta-method
  retraction error → regret. Cleaner than 3a's assembled nonparametric rates — but less
  novel-looking for the same reason.
- **This is the near-term, lower-risk version** — a viable second paper after T1, ~4–6 weeks.
  Venue AISTATS / strong TMLR, borderline ICLR if the theory + a decisive win over
  EPBO/CUQB/physics-penalty land.

---

## 4. Prior-art sweep (evidence: nobody does this)

No one retracts acquisition candidates onto a *learned* equality manifold for BO. The
unknown-equality field is entirely **penalty / exact-penalty / quantile acquisition** on the
surrogate.

| Work | What it does | Relation to T3 |
|---|---|---|
| **EPBO** — Exact-Penalty BO (Lu & Paulson, Technometrics 2024, arXiv 2105.13245) | Exact-`ℓ1` penalty merit on GP means; closed-form acqf; black-box **equalities**; can start infeasible | **Closest existing method.** Penalty-on-surrogate, not retraction. **Primary foil/baseline.** |
| **CUQB** (Paulson et al., arXiv 2305.03824) | Deterministic quantile-bound acqf; environmental-model calibration + reactor RTO benchmarks; equalities via two-sided quantile bounds | No retraction, no manifold parametrisation. Strong baseline; its calibration benchmark is reusable. |
| **Exact-AL local BO** (arXiv 2506.00648) | Exact augmented-Lagrangian acqf for nonlinear equalities; gradient-enhanced; merit `< 1e-5` | Penalty family, no retraction. Baseline. |
| **LCBO** (arXiv 2603.07965, ICML 2026) | Local BO, `‖c‖²` penalty on surrogate, asymptotic-KKT | Released code is inequality-only, no baselines, unreproducible. Foil, not prior art. |
| **BE-CBO** — Boundary Exploration (Tian et al., ICML 2024) | **Binary** feasibility, NN-ensemble, explores the feasible/infeasible *boundary* with a band `l(x)=0.5−σ_E(x)` | **Closest methodological cousin** — its boundary band is the analogue of an `\|μ_h\|≤κσ_h` acceptance band, but for a *classification* boundary of a full-dimensional set, not a regression zero-level-set. No retraction. Cite as the analogue. (Co-author J. Dürholt is a BoFire/BASF contributor.) |
| **Level-set estimation** (Gotovos 2013, Bryan 2005, Shekhar–Javidi 2019) | Classify `{f ≥ τ}` region via GP confidence bounds | Treat the level set as a *region to classify*, never a surface to land on. Does not block. |
| **Constraint Active Search / COMBOO** (Malkomes 2021; arXiv 2310.08751) | Inequality "regions of satisfaction", coverage | Not equality-manifold landing. Does not block. |
| **BO on the Equilibrium Manifold** (arXiv 2606.29299) | Objective over an equality-defined "equilibrium manifold" | **Instructive negative:** solves the equality *exactly by Newton* (it's known). Reinforces: known ⇒ retract; unknown ⇒ nobody does. |
| **GPIS / COGIS** (arXiv 2410.00157) | Robot end-effector retracted onto a GP implicit surface `{μ(x)=0}` | **Real precedent for retracting onto a learned zero-level-set** — but real-time control, no acquisition function, no evaluation budget. Mild precedent, not a competitor. |
| Plug-in level-set error theory (Cuevas–Walther); manifold estimation (Genovese et al.) | Rates for `{ĥ=0}` vs `{h=0}` | **Theory ingredients** for A1 / A3 — the novelty check for A3 lives here. |
| Implicit-Manifold GP Regression (Fichera–Billard, NeurIPS 2023); Matérn GPs on Riemannian manifolds (Borovitskiy 2020) | GP *surrogate* on a *known* manifold | Orthogonal — kernel layer, known geometry. |

**Net:** not scooped. The neighbourhood (EPBO/CUQB, BE-CBO's band, GPIS retraction, LSE
theory) is dense enough that the paper must answer crisply: *what does retraction buy over the
exact penalty?* — which is what A1 + A3 are for.

---

## 5. Motivation (cross-field)

Two passes. The first searched the BO literature for the target problem class (circular — no
method ⇒ no problem framed as BO). The second searched **cross-field**: where do other
communities solve expensive equality-constrained optimization today, and would learned-manifold
BO beat them?

### 5A. Where hard equalities genuinely live — and why they're mostly *known*

| Domain | Hard equality? | Known or unknown? | Current method | T3 fit |
|---|---|---|---|---|
| Chemical flowsheet / recycle / equilibrium | yes (`K_eq=∏a_i^{ν_i}`, Gibbs min) | **form known**; activity coeffs / kinetics expensive or mis-parameterised | IPOPT on implicit form (deterministic); penalty in acqf (experimental budget) | **grey-box (3b) — the wedge** |
| Data reconciliation (mass/energy balance) | yes | **known** conservation laws | SQP orthogonal projection onto the constraint space | not a home; but a clean existence proof that "project onto a known equality manifold" is routine engineering |
| PDE-constrained / adjoint (`R(u,x)=0`) | yes | **known** (discretised PDE) | adjoint, reduced-space SQP; inexact case (ROM / neural-operator) has mature error-aware trust-region / AL methods (Zahr arXiv 2405.14827) | reviewer would point here — the field is *ahead* on inexact-equality opt, just not with GPs (too high-dim for BO) |
| Aircraft MDO trim (`net forces = 0`) | yes | **known** residual | trim solved *exactly by Newton inside each aero eval* | already retracted onto; no unknown-`h` version |
| Metabolic flux balance (`S·v=0`) | yes | **known**, and **linear** | LP over the flux polytope (ms) | linear equality — already handled exactly; no retraction-difficulty story |
| Underdetermined inverse problems (geophysics/imaging) | soft target | known / linearised forward map | Tikhonov regularisation; **hierarchical regularization of solution ambiguity** (S2590055222000014) *literally* optimises a secondary criterion over the solution manifold | cleanest structural match to the "secondary objective over the solution set" reframe — but regularisation-solved, not sample-limited, not BO |
| Photonic / metamaterial inverse design | soft FOM `‖T(x)−T_target‖²` | forward model has adjoints | gradient topology optimisation; neural surrogates; generative "design portfolios" over a *performance sublevel set* (arXiv 2510.05160) | reframe-adjacent; the manifold is a sublevel set, not an equality |
| Ensemble Kalman inversion | soft data-fit | expensive forward model | EKI — derivative-free, sample-efficient, no BO | possible *foil* for the inverse-design framing; pure inversion, no objective over the solution set |

**Takeaway:** where the equality is genuinely physical it is also genuinely known, and exact
retraction (Newton / adjoint / projection) is already standard practice — which *corroborates
the parent paper* (PSE has its own "implicit beats surrogate" result: Ma et al. arXiv
2310.09307). The nonparametric-`h` regime (3a) has no strong applied home in *current*
practice. Whether that's a true gap or a chicken-and-egg artefact is genuinely open; either
way, 3a is better positioned as theory.

### 5B. The "inverse-design-with-secondary-objective" reframe

Framing: the equality is "expensive forward model `S(x) = target`"; optimise a secondary
objective (cost, robustness, manufacturability) over the solution set. Real and recognised
under scattered names (lexicographic / goal programming; regularised inverse problems;
design centering; inverse design with a figure of merit) — no unified treatment. Helps
**rhetorically** (removes the "why an exact equality" question — answer: "it's a spec") but
not **mathematically**: for nonparametric `S` the washout is unchanged. **Use it as the
motivating story for grey-box (3b), not as a rescue of 3a.**

### 5C. Grey-box (3b) — concrete, citable cases

1. **Multi-product chemical reactor, known energy balance, unknown kinetics** — *BO of a
   Multi-Product Chemical Reactor Using Composite Models and Partial Physics Knowledge*
   (arXiv 2606.08611, 2026). Steady-state energy balance of known form, unknown Arrhenius
   parameters; **"the residual is incorporated into the acquisition function"**; budget
   **~12 evaluations**. **Exactly 3b, handled with a penalty, sample efficiency the point.**
   Direct foil + motivation.
2. **Reactive distillation with unknown activity coefficients** — `K_eq=∏a_i^{ν_i}` known,
   `a_i(x,T)` from PC-SAFT expensive → GP-surrogated (*BITS for GAPS*, arXiv 2511.16815;
   Winz et al. 2021, adaptive sampling *because* PC-SAFT calls are expensive).
3. **Gibbs-reactor implicit-vs-surrogate** (arXiv 2310.09307) — PSE's own result that exact
   equality treatment beats the surrogate at equal cost. Independent corroboration.
4. **Heat-transfer correlations with unknown coefficients** (Nusselt / friction-factor / LMTD
   with unknown `U`) — the parent paper's P5 is an LMTD constraint. Standard grey-box class in
   HEN synthesis.
5. **Aircraft trim against an uncertain (GP) aero database** — trim residual known in form,
   coefficients from a GP database ⇒ trimming = retract onto `{h(x;θ̂)=0}`. Not currently
   framed this way.

Sample efficiency *is* a documented pain point (10-iteration budgets; an adaptive-sampling
subfield for equilibrium surrogates). Current practice — "surrogate the expensive part,
penalise the residual" — is what the parent paper's evidence says is wrong for an equality.

**Honest weaknesses of 3b:** (i) PSE already prefers exact/implicit where affordable, so a
PSE reviewer may see it as "obvious, just do it in BO"; (ii) most PSE optimisation is
deterministic-NLP and doesn't need BO; (iii) the theory chain is clean but assembled from
off-the-shelf parts; (iv) T1 dominates on tractability + BoFire fit.

---

## 6. Minimal experiment (grey-box / 3b — the near-term version)

- **Flagship benchmark:** the parent paper's **P6 (catalytic CSTR)** or **P7 (esterification)**
  equilibrium equality, with the rate constant `k` and/or an activity coefficient hidden
  behind a noisy oracle `θ*`; optionally add a manufacturability/cost secondary term so the
  "solution-set" framing is explicit. **Second benchmark:** CUQB's environmental-model
  calibration constraint.
- **Method:** SingleTaskGP for `f`; parametric Bayesian posterior (or GP-on-residual) for
  `θ_h` updated each round; Newton retraction onto `{h(x;θ̂)=0}` via `manifold_optimizer.py`.
- **Baselines:** EPBO, CUQB, the **physics-residual-penalty of arXiv 2606.08611**,
  penalty-on-surrogate `μ_h²`, naive M-R using `μ_h` in place of `h`.
- **Metrics:** true `|h(x;θ*)|` vs budget; feasibility rate at `τ ∈ {1e-3, 1e-5}`; simple
  regret; wasted-infeasible-eval count; `θ̂` contraction for retraction-consistent vs
  penalty-drift queries; (A3) feasibility *distance* vs `‖∇h‖` across the domain.
- **Needs:** a constraint-model ↔ retraction coupling BoFire does not currently have (harness
  only, no library change). **Effort ~4–6 weeks.**

---

## 7. Open questions before committing effort

1. **Is A1's `(d−1)`-dim active-manifold-regression rate provable?** The pivotal question for
   the nonparametric regime and the theoretical spine. Chicken-and-egg; precedent in active
   level-set estimation. → needs a focused theory pass.
2. **Is A3 a genuine "first"?** Novelty check against plug-in level-set / manifold estimation
   theory (Cuevas–Walther, Genovese) — the statistical distance bound exists; is *using it for
   conditioning-aware retraction + BO regret* new?
3. **Which regime to target** — 3b (near-term, lower-risk, second paper after T1) vs a bigger
   "feasibility under uncertainty" theory paper (3a + A1, higher-risk, higher-ceiling)?
4. Does A2's gradient-enhanced-surrogate assumption match a real benchmark, or is it a
   convenience?

---

## Log

- 2026-09-03 — track opened; LCBO paper + code reviewed (see `../D2_PRESTUDY.md`).
- 2026-09-03 — scoping round 1: BO-literature criterion (circular). Prior-art sweep good; the
  "no-go" verdict on it was premature.
- 2026-09-03 — washout reframed as the research problem; angles A1–A4 drafted.
- 2026-09-03 — scoping round 2: corrected cross-field motivation criterion. Findings folded
  into §5. Nonparametric regime (3a) has thin *current-practice* motivation; grey-box (3b) has
  concrete cases + a published weak foil (arXiv 2606.08611).
- 2026-09-03 — **doc reframed** (this version): "dead / no-go" scaffolding removed; structured
  as problem (§1) + research program (§2, A1–A4) + regimes (§3) + evidence (§4–5). T3 is a
  live direction, secondary to T1. Awaiting user decision on §7.
