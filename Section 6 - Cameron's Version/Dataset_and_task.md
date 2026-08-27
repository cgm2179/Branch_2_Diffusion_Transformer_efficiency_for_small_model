# §6 — Feedforward MLP: the measurement ruler · Dataset & Task

The "no-structure" control — the ruler against which the structured architectures (§7 CNN, §8 attention)
are read. With no separable channels the MLP exposes only the global scalar (**K=1**), so it *cannot* escape
the barrier law — the cleanest place to read the scaling law. Built at **P≈1.01M** (same as the §8 ViT) so
the three-factor **cos sweep** is directly comparable across architectures.

## Dataset — student/teacher regression (no real data)
A fixed random **teacher** MLP (`IN_DIM → 128 → OUT_DIM`, e.g. 64→128→16, frozen, deterministic per `SEED`)
defines the target `y = teacher(x)`. Fresh `x ~ N(0,I)` each step. A fixed held-out set `(Xte, Yte)` is
drawn once from the teacher. No real dataset — the point is to read `cos²`/M\* and fit quality exactly.

## Task
The **student** (a ~1.01M deep-**tanh** MLP) regresses the teacher: loss = **MSE**, quality =
**R² = 1 − MSE/Var(Y)** (0–1, fraction of variance explained). tanh is chosen for the paper's "depth dial"
(saturation degrades per-mode alignment 1.00→0.821). On this quadratic objective the antithetic estimator is
**exactly unbiased at every σ** (paper's Prop.).

## Regimes (Tests 1–3)
| regime | Test | status | notebooks |
|---|---|---|---|
| 1 | from-scratch | **built** | `01_backprop`, `02_two_factor_hebbian`, `03/04/05_three_factor_{clean,normal,noisy}` (cos≈0.5/0.09/0.01), `06_compare_results` |
| 2 | fine-tune | planned | LoRA/QLoRA rank sweep: M\* vs P, three-factor vs backprop-LoRA vs two-factor |
| 3 | inference-time | planned | per-instance adaptation |

**regime 1** trains the *same* student (same teacher + init, `SEED`) five ways on **equal wall-clock**,
mirroring §8: Adam/MSE control, target-blind Hebbian (R²≈0), and the three-factor probe rule at three cos
budgets (the cos sweep). Because MLP forwards are ~17× cheaper than the §8 ViT's, even the **clean** M\*
budget gets many steps — so unlike §8's ViT it should actually train from scratch. Metrics: test MSE, R²,
cos/M\*, cost (fwd-equiv), tape-free memory.

## Reuse
Same equal-wall-clock harness as §8 (Drive checkpoint/auto-resume, live plot, `results/*.json`, the vmap
antithetic estimator — with MSE in place of cross-entropy). The existing root notebooks
`Mstar_Probe_Budget_and_Cost_Comparison` / `Training_Accuracy_and_Total_Cost` are the small-scale M\*-vs-P
measurements this ruler extends.

## Pre-registered bars
*TODO before the real run:* three-factor clean reaches R² ≥ (target) within budget; the cos-sweep ordering
(does clean or noisier win?) — a genuine test vs §8, where noisier-more-steps won on classification.
