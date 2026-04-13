# projectAttack

`projectAttack` is a private working repository for studying adversarial patch and projection attacks against vision-language-action policies in robotics.

This codebase builds on top of OpenVLA and related robotics tooling, but it is maintained here as an independent project workspace focused on attack research, rollout-based optimization, and evaluation utilities.

## What Is In This Repo

- `VLAAttacker/`: attack implementations, rollout-based optimizers, wrappers, and transforms
- `evaluation_tool/`: offline analysis and evaluation helpers
- `scripts/`: runnable shell entrypoints for training, rollout, probing, and evaluation
- `experiments/`: environment and policy evaluation helpers
- `tests/`: lightweight regression and sanity checks

## What Is Not Versioned Here

This repository is intended to stay code-only. Large local dependencies and runtime artifacts should remain outside Git history, including:

- `openvla-main/`
- `LIBERO/`
- `run/`
- local datasets and checkpoints
- large archives such as demo zips

## Local Setup

```bash
conda create -n projectAttack python=3.10 -y
conda activate projectAttack

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

git clone <your-private-projectattack-url> projectAttack
cd projectAttack
pip install -e .

pip install packaging ninja
ninja --version
pip install "flash-attn==2.5.5" --no-build-isolation
```

## Local Dependencies

This repo assumes you prepare large external dependencies locally instead of tracking them in Git:

- OpenVLA / related dataset assets under a local `openvla-main/`
- LIBERO installed locally for environment evaluation
- local dataset directories such as `dataset/` or `datasets/`

If you use the provided scripts, prefer running them from the repository root so they can resolve paths relative to the current checkout.

## Common Commands

Generate legacy attacks:

```bash
bash scripts/run_TMA.sh
bash scripts/run_UADA.sh
bash scripts/run_UPA.sh
```

Run rollout-based attack variants:

```bash
bash scripts/run_UADA_rollout.sh
bash scripts/run_UADA_rollout_diffusion.sh
bash scripts/run_UADA_rollout_online_env.sh
```

Run probe experiments:

```bash
bash scripts/run_UADA_rollout_online_env_probe.sh
bash scripts/run_UADA_rollout_online_env_probe_round2.sh
bash scripts/run_UADA_rollout_online_env_probe_siglip.sh
```

Run offline SigLIP comparison:

```bash
bash scripts/run_siglip_embedding_eval.sh
```

## Notes

- `projectAttack` keeps the existing Python package/import layout for compatibility.
- Public upstream references to OpenVLA and LIBERO remain where they are needed for attribution or dependency setup.
- Before publishing this repository, replace placeholder GitHub URLs with your actual private repository URL.
