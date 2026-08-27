# §6 — Feedforward MLP: the measurement ruler · Dataset & Task

The "no-structure" control — the ruler against which the structured architectures (§7 CNN, §8 attention)
are read. With no separable channels the MLP exposes only the global scalar (**K=1**), so it *cannot* escape
the barrier law; but its parameter count is small, so the budget stays affordable — making it the cleanest
place to **pin the scaling laws**.

## Dataset
Synthetic regression — **no real data**. A fixed random target function; inputs drawn i.i.d. Deterministic
per seed. (This is a measurement task, not a benchmark: the point is to read `cos²`/M\* exactly, not to hit
an accuracy on real data.)

## Task
Regress the target with a small MLP and **measure the estimator against the true gradient**:
- Base model: a **1-16-1 tanh MLP (P = 49)** — anchors the parameter axis.
- Wider / rank-swept variants sweep **P** to trace the alignment law.

**What we measure**
- `cos²(ĝ_M, ∇L) = M/(M+P+1)` — the alignment law, swept across P.
- **M\*** vs P — should be linear (`M* ≈ (P+1)/3` at cos=½).
- The **K=1 endpoint** of the barrier law (the MLP is the only-one-channel case).
- The **depth dial**: as tanh layers saturate, per-mode alignment falls **1.00 → 0.821** — the concrete
  place gradient separability breaks down and the budget climbs.

## Regimes (Tests 1–3) — each mirrors §8's method set on the equal-wall-clock harness
| regime | Test | what it does | notebooks |
|---|---|---|---|
| 1 | from-scratch | train the MLP with no autograd; three-factor vs backprop vs two-factor; track the activation-tape memory gap | `01_backprop`, `02_two_factor_hebbian`, `03_three_factor`, `04_compare_results` |
| 2 | fine-tune | LoRA/QLoRA rank sweep: M\* vs P, three-factor vs backprop-LoRA vs two-factor | same 4-notebook set |
| 3 | inference-time | per-instance adaptation | same set |

Because the MLP is tiny, **three-factor at the full M\* budget is affordable here** — that is exactly the
point of the ruler (the barrier bites but stays payable, so the laws can be read cleanly).

## Reuse
Folds in the existing root notebooks `Mstar_Probe_Budget_and_Cost_Comparison.ipynb` (the M\* sweep) and
`Training_Accuracy_and_Total_Cost.ipynb`, plus `Memory_and_Parallelism_vs_Backprop_Fixed_A.ipynb` for the
tape-memory column. Harness reused from §8 (`Section 8 - Cameron's Version`).

## Pre-registered bars
*TODO before running:* target alignment (cos=½), the M\*-vs-P slope to confirm (predicted ~50 at P≈150),
and the depth level at which per-mode alignment drops below ~0.9.
