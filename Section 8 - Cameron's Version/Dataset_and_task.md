# §8 — Attention & Transformers: the Intermediate Case · Dataset & Task

Attention sits **between** the MLP ruler (§6) and the CNN native fit (§7): a transformer block stacks a
**separable** sublayer (feed-forward / projections, gradient factorizes → cheap) on a **coupled** one (the
attention map — softmax couples every query to every key, so S grows with context; multi-head claws some
back, K ≈ #heads). Prediction: local training is affordable on the separable sublayers, expensive on the
mixing (M\* ≳ S/K). **Regimes 1–2 are built and shipped; 3–4 are planned.**

## Dataset
- **MNIST** 28×28 — regime 1 (from-scratch ViT).
- **CIFAR-10** (upsampled to 224) — regime 2 (fine-tune a frozen ImageNet ViT-Base).

## Task
Image **classification** with a Vision Transformer.
- regime 1: a **from-scratch ViT** (P ≈ 1.01M; dim 176, depth 4, 8 heads, patch 7).
- regime 2: a **frozen `google/vit-base-patch16-224` (86M)** + LoRA adapters on the last-2-block attention
  projections + a fresh 10-way head.

## Regimes (Tests 1–4)
| regime | Test | status | notebooks |
|---|---|---|---|
| 1 | from-scratch | **built** | `01_backprop`, `02_two_factor_hebbian`, `03_three_factor`, `04_compare_results`, plus a **cos sweep**: `05_three_factor_v2` (cos≈0.09) and `06_three_factor_v3` (cos≈0.01) |
| 2 | fine-tune | **built** | `01_lora_backprop`, `02_qlora_backprop`, `03_two_factor_hebbian`, `04_three_factor`, `05_compare_results` |
| 3 | inference-time | planned | per-instance adaptation of the separable sublayers |
| 4 | escape-lever sweep | planned | vary head-group count K and attention window S; check M\* ≳ S/K |

## Key finding so far (regime 1)
From-scratch full-model local training is barrier-limited (K=1, M\*≈P/3 ⇒ few steps). The **cos sweep** is
the story: v1 (cos≈0.5, ~few clean steps) vs v2 (cos≈0.09) vs v3 (cos≈0.01) — many noisier steps in equal
wall-clock. Reducing the probe budget trades **per-step gradient precision** (not data) for **step count**;
the compare notebook's §7 head-to-head reads down the cos column to find where the trade stops paying off.

## Harness
Equal wall-clock (`TIME_BUDGET_HOURS`), Drive checkpointing (adapters/params every ~2 min, auto-resume),
hourly live loss-vs-time graph, `results/<method>.json`. This harness is the template §6/§7/§10 reuse.

## Shipped
Regime 1 → Branch_2 PR #1 (merged); cos-sweep v2/v3 + compare → PR #3/#4; regime 2 → PR #2 (merged) with a
LoRA-targeting fix in follow-up PRs.
