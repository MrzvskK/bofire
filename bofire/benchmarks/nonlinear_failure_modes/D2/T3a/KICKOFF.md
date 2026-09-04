# T3a — kickoff (feasibility under uncertainty; nonparametric `h`)

## First steps

1. Read [`../SHARED_CONTEXT.md`](../SHARED_CONTEXT.md).
2. Read [`NOTES.md`](NOTES.md) (this folder) — the focused working doc.
3. For the full prior-art sweep and cross-field motivation, read
   [`../TRACK_T3_unknown_manifolds.md`](../TRACK_T3_unknown_manifolds.md) §4–5 (archive).
4. Memory: your file is `d2-t3a.md` — edit only that one.

## One-paragraph statement

The constraint `h(x)=0` is a **fully unknown black-box function**, modelled by a GP. Plug-in
retraction onto `{μ_h = 0}` is capped at the GP error `ε_n`. The research program (**A1–A4**)
is about beating that floor:

- **A1 (pivotal)** — active learning of `h` *along the retraction path* → intrinsic
  `(d−1)`-dim estimation rate `n^{−ν/(2ν+(d−1))}` instead of ambient `n^{−ν/(2ν+d)}`. This is
  T1's intrinsic-dimension principle on the estimation side.
- **A3** — control feasibility *distance* (`dist(x,M) ≤ ε_n/‖∇h‖`), not constraint *value*;
  the washout is worst in the degenerate-Jacobian regime the parent paper targets → "first
  method to control feasibility distance for an estimated equality constraint."
- **A2** — gradient-enhanced constraint surrogates (adjoint/autodiff) shrink `ε_n`.
- **A4** — retract→evaluate→update as GP-Newton stochastic root-finding.

**Framing:** *"Feasibility under uncertainty."* **Likely a theory-leaning paper**
(TMLR / AISTATS / borderline ICLR) — the nonparametric regime has thin *current-practice*
motivation (see NOTES §Motivation), so the contribution has to be the rates + the mechanism,
with synthetic experiments.

## The job of this chat

**Decide whether T3a is a paper, and of what kind.** Concretely:

1. **Is A1's `(d−1)`-dim rate provable?** Pivotal. Chicken-and-egg (need `M̂` to sample, need
   samples for `M̂`). Precedent: active level-set estimation (Willett–Nowak; Shekhar–Javidi
   2019). This needs a real theory pass — sketch the proof or find where it breaks.
2. **Is A3 a genuine "first"?** The statistical piece (`d_H(M̂, M)` vs `ε_n/‖∇h‖`) exists in
   plug-in level-set / manifold estimation theory (Cuevas–Walther, Genovese et al.). Novelty =
   using it for conditioning-aware retraction + the BO regret consequence. Verify.
3. **A2's assumption** — does "simulator provides `∇h`" match a real benchmark, or is it a
   convenience?
4. If the theory closes: draft the paper skeleton + a synthetic experiment plan (unknown-`h`
   benchmark = hide a P1–P7 equality behind a noisy oracle; baselines EPBO, CUQB,
   penalty-on-surrogate, naive M-R using `μ_h`).

## Hard rules

Research harness only — no BoFire/BoTorch constraint-handling changes before @jduerholt.
Theory/scoping first; no code until the A1 question is answered.
