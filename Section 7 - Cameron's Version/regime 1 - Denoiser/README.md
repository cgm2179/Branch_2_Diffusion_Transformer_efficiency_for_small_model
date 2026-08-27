# §7 CNN — regime 1 - Denoiser (from scratch, binarized MNIST)

Five rules train the same ~1.01M CNN autoencoder (denoising: input = image + Gaussian noise), matched on equal wall-clock; loss = per-pixel BCE, metric = pixel-accuracy (all-background floor ~0.87). Three-factor cos sweep clean/normal/noisy = cos 0.5/0.09/0.01. Checkpoints to Drive (`Section7_r1_denoiser/`), live plot, `results/<method>.json`. Compare in `06_compare_results.ipynb`.
