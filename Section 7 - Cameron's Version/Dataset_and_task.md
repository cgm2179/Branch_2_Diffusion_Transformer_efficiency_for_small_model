# §7 — CNN Diffusion: a Native Fit · Dataset & Task

The **positive result**: diffusion on a convolutional denoiser hands the local rule exactly the
separability the barrier law rewards. The training signal is a per-pixel, per-noise-level residual, and in
the Parseval (spectral) basis the denoiser is **diagonal** — each mode is its own channel, so the coupling
scope is **S = 1**. Diffusion also supplies a *second* feedback axis: the **timestep**.

## Dataset
**MNIST**, 28×28 grayscale (torchvision). Small enough to train a diffusion model from scratch in a
notebook; the spectral structure is clean enough to expose the S=1 native fit.

## Task
**DDPM-style denoising diffusion** on a small **CNN denoiser**:
- Forward noising schedule (β/ᾱ), denoising-score-matching (**DSM**) loss = predict the per-pixel,
  per-noise-level residual.
- Generation = reverse sampling from noise.
- **Endpoints:** DSM loss, and sample quality (a generated-digit grid; optionally FID-lite / a proxy).

**What we measure** — the same cos²/M\* machinery as §6, plus the diffusion-specific channels:
- The per-mode Wiener-gain optimum is closed-form (quadratic objective) → the local rule converges linearly.
- **Timestep bins** as independent feedback channels: binning timesteps yields per-bin adapters that
  multiply feedback bandwidth K and, by the barrier law, cut the budget.

## Regimes (Tests 1–4)
| regime | Test | what it does | notes |
|---|---|---|---|
| 1 | from-scratch | MNIST diffusion, backprop vs local head-to-head | **biggest new build** — a fresh small CNN DDPM (the earlier WDSGB diffusion notebooks were scratched) |
| 2 | fine-tune | LoRA-subspace post-training; rank sweep M\* vs P vs backprop-LoRA | |
| 3 | inference-time | per-instance repair (single subject/style) | feeds §10.1 test-time personalization |
| 4 | temporal sweep | M\* vs number of timestep bins B | expect **M\* ∝ 1/B** until it plateaus; plateau height measures the off-Gaussian coupling gap Δ(t) |

Notebooks per regime mirror §8 (`01_backprop / 02_two_factor_hebbian / 03_three_factor / 04_compare_results`)
on the equal-wall-clock harness — swapping the loss (DSM instead of CE) and the model (CNN denoiser).

## Reuse
§8 harness (equal wall-clock, Drive checkpoint, live plot, results JSON, antithetic estimator). The MNIST
`.jpg` data already in `archive/` is available offline; torchvision MNIST is the default.

## Pre-registered bars
*TODO before running:* from-scratch — local reaches a target DSM loss / recognizable samples within budget;
temporal sweep — M\* falls at least ~1/B over B∈{1,2,4,8} before plateauing.
