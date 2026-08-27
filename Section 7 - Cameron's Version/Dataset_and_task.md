# §7 — CNN: a Native Fit · Dataset & Task

CNN on MNIST — the structured-but-local architecture. **regime 1 (built)** is image **reconstruction** of
binarized MNIST (an Autoencoder and a Denoiser); the paper's full **DDPM diffusion** is deferred to a later
regime. Built at **P≈1.01M** (same as §6/§8) so the three-factor **cos sweep** is comparable across
architectures.

## Dataset
**MNIST**, binarized: `x = (pixel > 0.5)` ∈ {0,1}, 28×28 (torchvision). Pixel-accuracy therefore has an
**all-background floor ≈ 0.87** (most pixels are 0) — methods must beat that.

## Task — recreate the image, score per-pixel accuracy
A **CNN autoencoder** (conv encoder → dense latent bottleneck → Upsample+Conv decoder → 28×28 logits, ~1.01M)
reconstructs the image. Loss = per-pixel **BCE-with-logits**; metric = **pixel-accuracy** = fraction of the
784 pixels with `sigmoid(logit)>0.5 == target` (0–1). Two regime-1 variants:
- **Autoencoder** (`CORRUPT_SIGMA=0`): input = clean binary image, target = same; the bottleneck makes it
  non-trivial.
- **Denoiser** (`CORRUPT_SIGMA=0.5`): input = image + Gaussian noise, target = clean image.

## Regimes
| regime | what | status |
|---|---|---|
| 1 — Autoencoder | reconstruct binarized MNIST | **built** — `regime 1 - Autoencoder/` |
| 1 — Denoiser | recover clean image from noisy input | **built** — `regime 1 - Denoiser/` |
| 2 | fine-tune (LoRA rank sweep M\* vs P) | planned |
| 3 | inference-time (per-instance repair) | planned — feeds §10.1 |
| 4 | **DDPM diffusion** + temporal-bin sweep (M\*∝1/B) | planned — the paper's "native fit" §7 |

Each built regime runs **5 methods on the same ~1.01M CNN, equal wall-clock**: `01_backprop` (Adam+BCE),
`02_two_factor_hebbian` (target-blind Hebbian on the dense layers; conv kernels left at init → ≈ background),
`03/04/05_three_factor_{clean,normal,noisy}` (cos≈0.5/0.09/0.01), `06_compare_results` (BCE + pixel-acc
curves, cost/memory, cos-sweep head-to-head, a **sample-reconstruction grid** input·target·reconstruction).

## Reuse & harness
Same equal-wall-clock harness as §6/§8 (Drive checkpoint/auto-resume, hourly live plot, per-minute heartbeat,
`results/*.json`, the vmap antithetic estimator — with BCE in place of cross-entropy/MSE). Decoder uses
**Upsample+Conv (not ConvTranspose)** for vmap compatibility; no BatchNorm (vmap-clean). CNN forward
(~10M FLOPs/img) sits between the §6 MLP and §8 ViT, so the clean-budget three-factor gets a moderate number
of steps.

## Pre-registered bars
*TODO before the real run:* backprop pixel-acc ≥ (target) above the 0.87 background floor; three-factor
clean reaches a recognizable reconstruction; the cos-sweep ordering vs §6/§8.
