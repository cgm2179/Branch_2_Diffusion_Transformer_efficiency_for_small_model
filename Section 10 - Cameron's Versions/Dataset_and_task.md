# §10 — Applications to Generative AI · Dataset & Task

Three properties of the rule — **tape-free, gradient-free, probe budget known in advance** — pick out a
family of deployments where backprop is unavailable or uneconomical. Seven directions; we **build the 3
flagship** and **scaffold the other 4**.

## Flagship (build)

### 10.1 — Test-time personalization of diffusion
- **Dataset/model:** a pretrained diffusion model (reuse §7's MNIST DDPM; later a HF `diffusers` checkpoint)
  + a **single** subject/style/prompt instance.
- **Task:** adapt the model to that one instance **at inference, forward-only, per instance** — no backward
  pass, no offline loop. The flagship use of the §7 native fit + the per-instance-repair regime.
- **Baseline:** backprop per-instance adaptation where feasible; **endpoint** = the personalized sample.

### 10.4 — Adapting quantized / black-box models  *(the killer app)*
- **Dataset/model:** an **INT4/INT8** generative model (bitsandbytes 4-bit; reuse the §8 regime-2 qLoRA
  stack), or a query-only / API model.
- **Task:** forward-only adaptation where **∇L = 0 exactly** (quantized plateaus) or is unavailable (API).
  There is **no backprop baseline** — that impossibility *is* the result ("possibility, not degree").
- **Reuse:** `Nondifferentiable_Channel_Demo.ipynb` is the seed (training through a `round()`/quantized
  channel where autograd returns the zero vector but the probe estimate points the right way).

### 10.3 — Aligning to non-differentiable rewards
- **Dataset/model:** a generator + a **non-differentiable** metric / verifier / human-style reward.
- **Task:** forward-only preference/verifier/metric alignment (an RLHF/RLAIF-style loop) for a reward that
  carries no gradient — exactly the discrete-signal regime. **Endpoint** = reward improvement by evaluation
  alone.

## Scaffold (documented, not built yet)
- **10.2 On-device / edge fine-tuning** — training-memory = inference-memory lets a model be adapted where
  the activation tape never fit (phones, browsers, MCUs); the demo is a **memory benchmark** turning the
  43.8× tape gap into deployability.
- **10.5 Long-context adaptation without the tape** — reverse-mode's Θ(LT) tape is infeasible; the local
  rule's cost is independent of context length.
- **10.6 Streaming / continual adaptation** — track distribution shift on a live generation stream at
  bounded per-step cost.
- **10.7 Pricing adaptation as a service level** — M\* is predictable in advance, so a deployment can budget
  the forward-pass cost to hit a target quality **before** running (an analytical/cost-model demo, not a
  training notebook).

## Structure
Each flagship app is its own folder under this section, mirroring §8 where a baseline exists
(`03_three_factor` forward-only vs a backprop control + `04_compare_results`); for 10.4 there is no backprop
baseline by construction. Same equal-wall-clock harness, Drive checkpointing, results JSON.
