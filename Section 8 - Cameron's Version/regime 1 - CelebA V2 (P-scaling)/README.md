# Section 8 · regime 1 (CelebA) · V2 — capacity (P) scaling

**Question.** On CelebA 64×64, 40-attribute classification: does adding ViT capacity **P** raise test
accuracy, and does the forward-only **three-factor** estimator resist the overfitting that **backprop**
shows at small P?

Backprop at ~1M overfits this task quickly (see the base `regime 1 - CelebA` run). The three-factor
gradient is an unbiased ES estimate of the *Gaussian-smoothed* loss; its variance grows with P, so this
study asks whether extra capacity helps a forward-only learner or just makes the estimate too noisy.

## The four runs (one shared wall-clock)

| Notebook | Method key | ~P | Optimizer | Notes |
|---|---|---|---|---|
| `01_three_factor_supernoisy_1M` | `three_factor_supernoisy_1M` | ~1.01M | forward-only ES | M=16 probes (super-noisy) |
| `02_three_factor_noisy_10M` | `three_factor_noisy_10M` | ~10.5M | forward-only ES | M=64 probes |
| `03_three_factor_noisy_25M` | `three_factor_noisy_25M` | ~25.2M | forward-only ES | M=64 probes |
| `04_backprop_500M` | `backprop_500M` | ~503M | Adam + backprop | bf16 autocast + gradient checkpointing |
| `05_compare_results` | — | — | — | accuracy-vs-P + overfitting figures |

All four use the same `TIME_BUDGET_HOURS` ("same time clock") and write to
`Drive/MyDrive/Section8_r1_celeba_v2/results/<method>.json` (separate from the base run).

## How the probe budget scales (important)
Probe count **M is a fixed small budget**, held roughly constant across P — it is **not** scaled to a
target cos. So per-step cost stays `O(M·P)` (linear) and avoids the `O(P²)` wall that a cos-matched
budget would hit (that wall is exactly why the base three-factor notebooks `assert P < 3_000_000`; the
V2 notebooks lift that guard on purpose). Consequence: higher P ⇒ **lower realized cos** (noisier
gradient) **and fewer steps** in equal wall-clock. Each build cell prints the exact P, the realized cos,
and the forward-passes/step. `M` and `CHUNK_PROBES` are one-line knobs at the top of each notebook.

## Expected shape of the result (not a bug)
- **backprop@500M** should **overfit**: train BCE dives below test BCE (a positive, widening gap).
- **three-factor** test BCE moves **slowly**, more so at higher P — at 25M the realized cos is ~0.002,
  so a low / slow-learning result there is the informative "ES variance wall", not a failure.
- With backprop at 500M only (no matched-P backprop controls), the optimizer-vs-capacity contrast is
  suggestive, not a fully controlled ablation. You can overlay the completed `backprop`@1M / `backprop_100M`
  runs in `05` (see `EXTRA_RESULTS_DIRS`) for a fuller backprop P-curve, with no new compute.

## How to run (Colab, A100)
1. Runtime → A100. Each training notebook: run top-to-bottom. First run downloads CelebA (Kaggle) into
   the shared data cell; upload `kaggle.json` when prompted.
2. Sanity check first: set `SMOKE = True` in a notebook for a ~20s CPU end-to-end pass, then set it back.
3. 500M memory: built for a 40GB A100 via bf16 autocast + block gradient checkpointing (`USE_CHECKPOINT=True`).
   If you still OOM, drop `BATCH` 64→32. On an 80GB A100 you can turn checkpointing off for speed.
4. After the four runs finish, run `05_compare_results` for the accuracy-vs-P and overfitting figures and
   the exported CSV/zip under `Section8_r1_celeba_v2/exports`.
