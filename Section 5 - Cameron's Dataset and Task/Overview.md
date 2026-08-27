# §5 — Methods & Experimental Protocol · Experiment Overview

Overview of the whole experimental program behind the paper (numbering per
`Research_Paper_Section_1-4_Draft.pdf`). Part I (§1–§4) is theory; §5–§10 test it. Each architecture is
put through the **same three-regime battery** and read against the **same baselines** and **pre-registered
bars**. This file is the map; each empirical section has its own `Dataset_and_task.md`.

## What we are testing
The forward-only **three-factor** rule — `ĝ = ξ·(L(θ+σξ) − L(θ−σξ)) / 2σ` — as a backprop replacement,
across a deliberately ordered sweep of architectures:

| § | Architecture | Role | Key structural fact |
|---|---|---|---|
| 6 | Feedforward MLP | the **ruler** | no separable channels → K=1, can't escape the barrier law |
| 7 | CNN Diffusion (MNIST) | the **native fit** | spectral-diagonal denoiser → coupling scope S=1 |
| 8 | Attention / ViT | the **intermediate case** | separable FFN/projections + coupled attention map |
| 10 | Applications | the **practical surface** | where backprop is unavailable/uneconomical |

## The three-regime battery (per architecture)
Each empirical section runs the same regimes (some add a sweep):
1. **from-scratch** — train with no autograd; three-factor vs backprop, head-to-head.
2. **fine-tune** — LoRA/QLoRA-subspace adaptation; rank sweep of M\* vs P.
3. **inference-time** — per-instance adaptation.
4. **sweep** *(structured archs only)* — §7 temporal-bin sweep (M\*∝1/B), §8 escape-lever sweep (M\*≳S/K).

## Harness (shared)
Seeded runs with JSON provenance for every measurement — no result comes "out of a notebook" without a
recorded seed and config. Runs are matched on **equal wall-clock**, checkpoint to Google Drive every ~2 min
(disconnect-safe, auto-resume), draw a live loss-vs-time graph, and write `results/<method>.json`.

## Metrics
- **`cos²(ĝ_M, ∇L)`** — gradient alignment (the scaling-law axis).
- **M\*** — probe budget implied by the target alignment.
- **Downstream endpoints** — accuracy, loss, or sample quality.
- **Cost** — forward-pass-equivalents; **memory** — the tape-free (training-memory = inference-memory) gap.

## Baselines
Two honest baselines at **equal fresh-sample budget**: **backpropagation (with LoRA, Adam)** — the strong
control the local rule must approach; and the **two-factor Hebbian** rule — the target-blind control it must
beat (expected ≈ chance).

## Pre-registered pass bars
Every test's success criterion is declared **before** the run to keep verdicts honest. Bars are recorded in
each section's `Dataset_and_task.md` before its notebooks are run.

## Sections
- [§6 — Feedforward MLP](../Section%206%20-%20Cameron's%20Version/Dataset_and_task.md)
- [§7 — CNN Diffusion](../Section%207%20-%20Cameron's%20Version/Dataset_and_task.md)
- [§8 — Attention / ViT](../Section%208%20-%20Cameron's%20Version/Dataset_and_task.md) *(regimes 1–2 built)*
- [§10 — Applications](../Section%2010%20-%20Cameron's%20Versions/Dataset_and_task.md)
