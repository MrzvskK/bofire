# T3a — working notes (feasibility under uncertainty; nonparametric `h`)

Focused extract of [`../TRACK_T3_unknown_manifolds.md`](../TRACK_T3_unknown_manifolds.md) for
the nonparametric regime. That archive doc has the full prior-art table (§4) and cross-field
motivation (§5) — not duplicated here.

---

## 1. The problem

`h: ℝ^d → ℝ` unknown, expensive. GP posterior `(μ_h, σ_h)` from `n` noisy evaluations. True
`M = {h=0}`, estimate `M̂ = {μ_h=0}`. Returned candidate's true infeasibility:

| Source | Size | Removed by retraction? |
|---|---|---|
| Penalty bias — finite-`ρ` `min f + (ρ/2)μ_h²` stops at `\|μ_h\| ≈ \|∇f\|/(ρ‖∇μ_h‖)` | blows up as `‖∇μ_h‖→0`; `ρ→∞` triggers `μ_h²` false-convergence | **yes** — landing exactly on `{μ_h=0}` removes it (parent paper, transplanted) |
| GP error `ε_n = ‖μ_h − h‖_∞` | nonparametric `(n/\log n)^{−ν/(2ν+d)}` — slow (`ν=2, d=5` ⇒ `n^{−2/9}`) | **no** — retraction onto a biased `μ_h` lands on a wrong manifold |

Naive plug-in retraction is capped at `ε_n`. **`ε_n` is the research problem.**

## 2. The program (A1–A4)

### A1 — active learning of `h` along the retraction path → intrinsic `(d−1)`-dim rate

`ε_n` is the rate for learning `h` over a `d`-dim region. We only need `h` accurate on/near
`M`, a `(d−1)`-set. The retract→evaluate→update loop already concentrates constraint evals on
`M̂`; formalise as **manifold-restricted active regression** → rate `n^{−ν/(2ν+(d−1))}`.

- **T1's intrinsic-dimension principle on the estimation side.** One principle, two instances.
- **Analysis risk:** chicken-and-egg. Precedent: active level-set estimation (Willett–Nowak;
  Shekhar–Javidi 2019). **This is THE open question for the chat.**

### A2 — gradient-enhanced constraint surrogates

Simulator yields `∇h` (adjoint / autodiff) ⇒ co-kriging shrinks `ε_n` (`d+1` scalars/eval);
Newton retraction wants `∇h` anyway. Narrows scope to grey-box/simulator settings.

### A3 — control feasibility *distance*, not constraint *value*

`dist(x,M) ≤ ε_n/‖∇h‖`. Small `‖∇h‖` (Tier-1) blows the distance error even with `|h|`
controlled ⇒ **the washout is worst exactly where the parent paper matters most.** A2 +
conditioning-aware retraction (parent paper's pseudo-inverse step) ⇒ candidate "first method
to control feasibility *distance* for an estimated equality." **Novelty check needed** vs
plug-in level-set / manifold estimation theory (Cuevas–Walther; Genovese et al.) — the
statistical distance bound exists there; novelty = the retraction + BO-regret use.

### A4 — retract→evaluate→update as GP-Newton stochastic root-finding

Treat the loop as a Newton-like iteration on an unknown function; local evals near `x*` pin
`h` down fast. Connects to stochastic root-finding (Robbins–Monro; GP root-finding).

## 3. Theory sketch (assembled — the risk is "combination of known rates")

1. GP sup-norm contraction on a tube around `M*`: `‖μ_h − h‖_∞ = ε_n` (van der Vaart–van
   Zanten).
2. Non-degeneracy `m = inf_{M*}‖∇h‖ > 0` + positive reach ⇒ plug-in level-set bound
   `d_H(M̂_n, M*) = O(ε_n/m)` (Cuevas–Walther).
3. Retracted candidate has `|h| = O(ε_n)` ⇒ feasibility rate `→ 1` after `n ≳ τ^{−(2ν+d)/ν}`.
4. Constrained GP-UCB *along* `M̂_n`: regret `≤ O(√(T·γ_T^{M})) + Σ_t O(ε_{n_t})`
   (`γ_T^M` at dimension `d−1`).

A1 is the step that would make this *not* a pure assembly — the `(d−1)` rate has to come from
the active allocation, not be assumed. A3 is the step that ties it to the parent paper.

## 4. Motivation (why theory-leaning)

Two cross-field passes (archive §5): where hard equalities are genuinely physical, they are
also genuinely *known* (conservation laws, equilibrium forms, PDE state equations) and
practitioners already retract exactly (Newton / adjoint / projection) — which **corroborates
the parent paper** (PSE has its own "implicit beats surrogate" result, Ma et al. arXiv
2310.09307). The fully-unknown-functional-`h` + hard-equality + BO-budget + no-gradients
combination is rare in *current* practice. Whether that's a true gap or a chicken-and-egg
artefact of no method existing is genuinely open — but it means T3a is best positioned as a
**theory contribution** (rates + mechanism, synthetic experiments), not an applied paper.

Closest methodological cousins (none block): **EPBO** (exact-penalty BO, black-box
equalities), **CUQB** (quantile-bound), **BE-CBO** (feasibility-boundary band — the analogue
of an `|μ_h| ≤ κσ_h` acceptance band, but for classification), **GPIS/COGIS** (robotics —
*does* retract onto a learned zero-level-set, but real-time control, no acquisition function).

## 5. Venue

- Empirical only: NeurIPS/ICML **workshop** or **TMLR**.
- + a clean A1 rate + A3 "first" + a decisive win over EPBO/CUQB on feasibility-at-fixed-budget:
  **AISTATS** / strong **TMLR** / **borderline ICLR**.

## 6. Open questions (the chat's deliverable)

1. Is A1's `(d−1)`-dim active-manifold-regression rate provable? (pivotal)
2. Is A3 a genuine "first" vs Cuevas–Walther / Genovese?
3. Does A2's gradient assumption match a real benchmark?
4. If yes to 1–2: paper skeleton + synthetic experiment plan.

## Log

- 2026-09-03 — track split out of the combined T3 doc into its own chat/notes.
- 2026-09-04 — two-stage novelty sweep → [`NOVELTY.md`](NOVELTY.md). Stage 1 (BO): the triple
  {unknown functional `h` + hard equality + retraction under a budget} is **unoccupied**; COBALt
  (2310.08751) is the closest active-constraint-learning cousin and is inequality-region only,
  no rate theory. Stage 2 (general opt): **C1's `(d−1)` rate mechanism is prior art**
  (Castro–Nowak boundary-fragment active learning); **C2's `dist ≲ ε_n/m` is prior art**
  (Cuevas 2006, Willett–Nowak 2007). **RBDO adaptive-Kriging** (EGRA/AK-MCS) is the #1 Stage-2
  neighbour — actively samples near a learned `{g=0}` — but never retracts, does a *chance*
  constraint not a hard equality, and has no rate theory. **Verdict:** novelty survives at
  TMLR/AISTATS **only if C3 leads** (degenerate-Jacobian regime + lower bound); C1/C2 are
  transports glued by the self-consistent tube-width lemma. Drop all bare "first" on C1/C2.
- 2026-09-03 — theory pass done → [`ANSWERS.md`](ANSWERS.md). Q1: A1 rate provable via a
  self-consistent tube-width recursion **when `m = inf_M‖∇h‖ ≳ 1`, `ν ≥ 1`**; the recursion is
  **expansive (fails) for `m → 0`** — the parent paper's Tier-1 regime. Q2: A3 is **not** a
  statistical first (Willett–Nowak 2007, Cuevas 2006 already have `d_H ≲ ε_n/m` + the
  value/location duality); salvage as "first in BO + matching lower bound", drop bare "first".
  Q3: A2 realistic for adjoint/autodiff simulators but those have *known* `h` → keep A2
  secondary, non-load-bearing. Verdict: **TMLR/AISTATS-tier theory note**, framing = A1 (rate,
  well-conditioned) + A3-as-graceful-degradation (ill-conditioned) + lower bound. Sequence
  **after T1 and T3b**. Next step: write the A1 self-consistency lemma (~2 wk); clean ⇒
  standalone note, messy ⇒ fold A3+lower-bound into T3b and drop A1.
