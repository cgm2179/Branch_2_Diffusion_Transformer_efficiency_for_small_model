# §6 Feedforward MLP — regime 1 (from scratch, student/teacher regression)

Five rules train the **same** ~1.01M deep-tanh MLP (same teacher + seed), matched on **equal wall-clock**, regressing a fixed random teacher (loss = MSE, quality = R^2). Same P as the §8 ViT, so the three-factor **cos sweep** (clean 0.5 / normal 0.09 / noisy 0.01) is directly comparable across architectures.

1. `01_backprop` — Adam + MSE
2. `02_two_factor_hebbian` — target-blind control (R^2≈0)
3–5. `03/04/05_three_factor_{clean,normal,noisy}` — forward-only probe at cos≈0.5/0.09/0.01
6. `06_compare_results` — MSE/R^2 curves, cost/memory, cos-sweep head-to-head, predicted-vs-target, table

Each checkpoints to Google Drive (`Section6_regime1/`) every ~2 min (auto-resume), draws a live plot, and writes `results/<method>.json`. The MLP is the *ruler*: no separable channels (pure K=1), so it's the cleanest read of the scaling law; MLP forwards are ~17x cheaper than the ViT's, so even the clean M* run gets many steps.
