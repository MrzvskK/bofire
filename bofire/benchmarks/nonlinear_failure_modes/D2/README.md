# D2 — new research directions (sub-project)

Parallel workstreams extending the manifold-optimizers paper (AAAI submission, awaiting
reviews). Shared background: [`SHARED_CONTEXT.md`](SHARED_CONTEXT.md). Pre-study + rankings +
LCBO review: [`D2_PRESTUDY.md`](../D2_PRESTUDY.md).

## Tracks — one chat each

| Track | Chat | Kickoff | Working doc | Memory file | Status |
|---|---|---|---|---|---|
| **T1** — mixed discrete–continuous constraint manifolds | original session | [`T1/KICKOFF.md`](T1/KICKOFF.md) | [`TRACK_T1_mixed_manifolds.md`](TRACK_T1_mixed_manifolds.md) | `d2-t1.md` | **PRIMARY — GO**, scaffold built |
| **T3a** — feasibility under uncertainty (nonparametric `h`, program A1–A4) | new chat | [`T3a/KICKOFF.md`](T3a/KICKOFF.md) | [`T3a/NOTES.md`](T3a/NOTES.md) + [`TRACK_T3_unknown_manifolds.md`](TRACK_T3_unknown_manifolds.md) | `d2-t3a.md` | scoping — is A1's rate provable? |
| **T3b** — grey-box equality manifolds (`h(x;θ_h)`) | new chat | [`T3b/KICKOFF.md`](T3b/KICKOFF.md) | [`T3b/NOTES.md`](T3b/NOTES.md) + `TRACK_T3_unknown_manifolds.md` | `d2-t3b.md` | 2nd paper after T1; foil arXiv 2606.08611 |
| **D1** — AAAI rebuttal prep ("Track A") | new chat | [`D1_rebuttal/KICKOFF.md`](D1_rebuttal/KICKOFF.md) | [`../REBUTTAL_PREP.md`](../REBUTTAL_PREP.md) | `d1-rebuttal-prep.md` | **setup only — do NOT start until reviews land** |

Shelved: T2 (degenerate-Jacobian retraction — cross-field reinvention risk), T4 (stochastic +
batch — low ceiling). Both in `../D2_PRESTUDY.md`.

## How to start a track chat

1. Open a **new Claude Code chat in this repo** (`/Users/kmorozov/Documents/GitHub/bofire`).
   It auto-loads `CLAUDE.md` and the memory index `MEMORY.md` — the D2 context is already
   partly seeded.
2. First message:
   > Work on track **T3b**. Read `bofire/benchmarks/nonlinear_failure_modes/D2/T3b/KICKOFF.md`
   > and continue from "The job of this chat".
3. That's it — the kickoff points at `SHARED_CONTEXT.md` + the track's notes. No copy-paste of
   long context, no export.

## Memory discipline (important — 4 chats, 1 memory folder)

- Each chat edits **only its own** memory file (`d2-t1.md` / `d2-t3a.md` / `d2-t3b.md` /
  `d1-rebuttal-prep.md`).
- `d2-active-research.md` is a **read-only umbrella index** — don't write to it from a track
  chat.
- Same git branch for all tracks (`research/post-opt-snap`) — tracks touch different files.
  Switch to a worktree/branch per track only if real conflicts appear.

## The through-line

Every track is one principle — *intrinsic dimension beats ambient dimension when the feasible
set has known structure*: T1 on the acquisition side (category-dependent manifold family), T3
on the constraint-estimation side (learned manifold). See `SHARED_CONTEXT.md`.
