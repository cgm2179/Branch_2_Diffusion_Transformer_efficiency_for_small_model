# §8 Attention — regime 1 (from scratch, MNIST ViT)

Three training rules on the **same** ViT (P≈1.01M), matched on **equal wall-clock** (`TIME_BUDGET_HOURS`). Run the three trainers on separate A100 Colab runtimes, then compare.

1. `01_backprop.ipynb` — Adam + cross-entropy (strong control)
2. `02_two_factor_hebbian.ipynb` — target-blind Hebbian (control to beat, ≈chance)
3. `03_three_factor.ipynb` — forward-only antithetic zeroth-order (this work)
4. `04_compare_results.ipynb` — loss curves, predictions, cost/memory, summary table

Each trainer checkpoints to Google Drive (`Section8_regime1/`) every ~2 min (disconnect-safe), draws a live hourly loss-vs-time graph, and writes `results/<method>.json`.
