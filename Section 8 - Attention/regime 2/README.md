# §8 Attention — regime 2 (fine-tune: frozen HF ViT-Base + LoRA, CIFAR-10)

Four ways to adapt the **same** frozen `google/vit-base-patch16-224` via an **identical LoRA config** (rank 4, query/value, last 2 layers) + a fresh 10-way head, matched on **equal wall-clock**. Run the four trainers on separate A100 Colab runtimes, then compare.

1. `01_lora_backprop.ipynb` — LoRA, Adam through the FP16 frozen base (strong control)
2. `02_qlora_backprop.ipynb` — QLoRA, Adam through a 4-bit (nf4) frozen base (GPU/bitsandbytes)
3. `03_two_factor_hebbian.ipynb` — target-blind Hebbian on the adapters (≈chance)
4. `04_three_factor.ipynb` — forward-only antithetic probe on the adapters (this work)
5. `05_compare_results.ipynb` — loss curves, cost/memory, summary table

Each checkpoints the **adapters only** to Google Drive (`Section8_regime2/`) every ~2 min (disconnect-safe auto-resume), draws an hourly live loss-vs-time graph, and writes `results/<method>.json`.

Note: three-factor probes only the small adapters, but each probe is a full ViT-Base forward, so it is probe-expensive (a few hundred steps in 5 h). That is the honest cost of backprop-free big-model adaptation — the paper's 'possibility, not degree'. Set `LORA_TARGETS=['dense']` to move adapters onto the FFN sublayers for the separability sweep.
