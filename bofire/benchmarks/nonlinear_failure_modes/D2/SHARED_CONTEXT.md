# D2 — shared context (read once per track chat)

All D2 tracks extend one submitted paper and share one problem geometry. This doc is the
common background so each track's `KICKOFF.md` can stay short.

---

## The parent paper (submitted, awaiting reviews)

**"Manifold-Aware Acquisition Optimizers for Constrained Bayesian Optimization"** — AAAI
submission, PDF was at `~/Downloads/AAAI_submission__Manifold_optimization_ (4).pdf`, LaTeX at
`../report/MANIFOLD_OPTIMISERS_PAPER.tex`.

- **Setting:** the inner acquisition optimization `max_x α(x) s.t. h_i(x)=0, g_j(x)≤0`, with
  `h_i` **known closed-form**.
- **Problem it solves:** equality constraints define a measure-zero manifold; ambient
  penalty / Augmented-Lagrangian optimisers fail to get onto and stay on it. Three geometric
  failure modes (FM1 competing gradients, FM2 tangential blind spot, FM3 infeasible-arc IC
  bias) cascade; a 4th (`h²` false-convergence when `∇h ≈ 0`) appears on practical benchmarks.
- **Method:** manifold-aware inner optimisers **M-θ / M-R / M-LN / M-H** + a **Newton
  retraction** that solves `h(x)=0` directly and stops on `|h|` (not `‖∇(h²)‖`).
- **Headline result:** M-R and M-LN reach **100% validity at `|h| ~ 1e-13`** on all 7
  practical benchmarks (P1–P7, 40 seeds); strongest baseline MCBO 88% at `|h| ~ 1e-4` on
  Tier-1 (near-degenerate Jacobian) problems.
- **Benchmark tiers by constraint-Jacobian conditioning:** Tier 1 near-degenerate (P1, P3),
  Tier 2 boundary-singular (P2, P4, P5), Tier 3 well-conditioned (P6, P7).

## The through-line: *intrinsic dimension beats ambient dimension*

Every D2 track is an instance of one principle — when the feasible set has known geometric
structure, working *on* the manifold (intrinsic dimension `d − m`) beats working in the
ambient `±τ` tube (dimension `d`).

- **T1** — on the *acquisition* side, with a category-dependent family of manifolds.
- **T3** — on the *constraint-estimation* side, when the manifold itself is learned.

## The washout (the T3 problem statement)

When `h` is **not** known exactly and must be learned, plug-in retraction onto `{μ_h = 0}` has
true infeasibility capped at the GP error `ε_n = ‖μ_h − h‖`. A finite-`ρ` penalty is *worse* —
it adds a bias `≈ |∇f|/(ρ‖∇μ_h‖)` that blows up as `‖∇μ_h‖ → 0` (the Tier-1 regime). So
retraction already wins by that bias term; `ε_n` is the open target. Angles **A1–A4** (see
`../TRACK_T3_unknown_manifolds.md` §2) are the program for beating `ε_n`.

## Key files

| Path | What |
|---|---|
| `../report/MANIFOLD_OPTIMISERS_PAPER.tex` | the submitted paper |
| `../PROJECT_SUMMARY.md`, `../STUDY_SUMMARY.md` | full research history |
| `../D2_PRESTUDY.md` | the 4-thread pre-study + rankings + LCBO review |
| `../manifold_optimizer.py` | M-R (`optimize_acqf_riemannian`), M-LN (`optimize_acqf_local_nullspace`), Newton retraction |
| `../benchmarks_practical.py` (P1–P2), `../benchmarks_process.py` (P3–P8) | the benchmark suite |
| `../benchmarks_mixed_manifold.py`, `../run_pt1.py` | **T1's** P-T1 benchmark + driver (built 2026-09-03) |
| `REBUTTAL_PREP.md` (one level up) | the D1 / AAAI-rebuttal plan |

## Hard rules (all tracks)

1. **No changes to nonlinear-constraint handling in BoFire core or BoTorch** until the user
   has spoken with **@jduerholt**. All D2 work is in the research harness
   (`bofire/benchmarks/nonlinear_failure_modes/`) — new files only, no edits to
   `bofire/data_models/`, `bofire/strategies/`, `bofire/surrogates/`, or the BoTorch package.
2. **BoFire gotcha:** `Inputs.get_keys()` returns keys **alphabetically sorted** — the bounds
   tensor and every optimiser candidate vector are in sorted-key order. Pick variable names
   whose sorted order equals the logical order, and assert it.
3. Treat arXiv papers with a grain of salt — check code / reproducibility where it matters
   (as was done for LCBO: paper says `c(x)=0`, released code is inequality-only).

## The four tracks

| Track | Chat | Memory file | Status |
|---|---|---|---|
| **T1** — mixed discrete–continuous constraint manifolds | the original session | `d2-t1.md` | PRIMARY, GO, scaffold built |
| **T3a** — feasibility under uncertainty (nonparametric `h`, program A1–A4) | new chat | `d2-t3a.md` | scoping: is A1's rate provable? |
| **T3b** — grey-box equality manifolds (`h(x;θ_h)`) | new chat | `d2-t3b.md` | 2nd paper after T1; foil arXiv 2606.08611 |
| **D1** — AAAI rebuttal prep (Track A) | new chat | `d1-rebuttal-prep.md` | setup only — do NOT start until reviews land |

**Memory discipline:** each chat edits **only its own** memory file. `d2-active-research.md` is
a read-only umbrella index.
