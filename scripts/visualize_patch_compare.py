#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def find_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for key in ("patch", "tensor", "projection_texture"):
            val = obj.get(key)
            if torch.is_tensor(val):
                return val
        for val in obj.values():
            if torch.is_tensor(val):
                return val
    raise ValueError(f"No tensor found in object type: {type(obj)}")


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.detach().cpu().float()

    if t.ndim == 4:
        t = t[0]

    if t.ndim == 3 and t.shape[0] in (1, 3):
        arr = t.permute(1, 2, 0).numpy()
    elif t.ndim == 3 and t.shape[-1] in (1, 3):
        arr = t.numpy()
    elif t.ndim == 2:
        arr = t.numpy()[..., None]
    else:
        raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)

    vmin = float(arr.min())
    vmax = float(arr.max())

    if vmin >= 0.0 and vmax <= 1.0:
        out = arr
    elif vmin >= -1.0 and vmax <= 1.0:
        out = (arr + 1.0) / 2.0
    else:
        span = max(vmax - vmin, 1e-8)
        out = (arr - vmin) / span

    return np.clip(out, 0.0, 1.0)


def collect_patch_paths(run_dir: Path):
    mapping = {}
    for sub in run_dir.iterdir():
        if not sub.is_dir():
            continue
        patch = sub / "patch.pt"
        if not patch.exists():
            continue
        if sub.name.isdigit():
            mapping[int(sub.name)] = patch
        elif sub.name == "last":
            mapping["last"] = patch
    return mapping


def choose_baseline(mapping, baseline_step=None):
    numeric_steps = sorted([k for k in mapping.keys() if isinstance(k, int)])
    if baseline_step is not None and baseline_step in mapping:
        return baseline_step
    if 0 in mapping:
        return 0
    if 49 in mapping:
        return 49
    if not numeric_steps:
        raise ValueError("No numeric-step patch found for baseline selection.")
    return numeric_steps[0]


def main():
    parser = argparse.ArgumentParser(description="Visualize patch comparison.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--baseline-step", type=int, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = run_dir / "patch_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = collect_patch_paths(run_dir)
    if "last" not in mapping:
        raise ValueError("No last/patch.pt found in run directory.")

    baseline_key = choose_baseline(mapping, args.baseline_step)
    current_key = "last"

    baseline_patch = find_tensor(torch.load(mapping[baseline_key], map_location="cpu"))
    current_patch = find_tensor(torch.load(mapping[current_key], map_location="cpu"))

    baseline_img = tensor_to_image(baseline_patch)
    current_img = tensor_to_image(current_patch)

    if baseline_img.shape != current_img.shape:
        raise ValueError(
            f"Patch shape mismatch: baseline={baseline_img.shape}, current={current_img.shape}"
        )

    diff = np.abs(current_img - baseline_img)
    diff_mean_map = diff.mean(axis=2)

    # Figure 1: baseline vs current + diff heatmap
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=180)
    axes[0].imshow(baseline_img)
    axes[0].set_title(f"Baseline patch (step={baseline_key})")
    axes[0].axis("off")

    axes[1].imshow(current_img)
    axes[1].set_title("Current patch (last)")
    axes[1].axis("off")

    im = axes[2].imshow(diff_mean_map, cmap="magma")
    axes[2].set_title("Abs diff heatmap (mean over RGB)")
    axes[2].axis("off")
    cbar = fig1.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("|delta|")

    l1 = float(diff.mean())
    linf = float(diff.max())
    fig1.suptitle(
        f"Patch comparison | baseline={baseline_key}, current=last | mean|delta|={l1:.6f}, max|delta|={linf:.6f}",
        fontsize=11,
    )
    fig1.tight_layout()

    fig1_path = out_dir / "patch_compare_baseline_vs_current.png"
    fig1.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: evolution grid for all saved checkpoints
    numeric_steps = sorted([k for k in mapping.keys() if isinstance(k, int)])
    ordered_keys = numeric_steps + (["last"] if "last" in mapping else [])

    n = len(ordered_keys)
    fig2, axes2 = plt.subplots(1, n, figsize=(4 * n, 4), dpi=180)
    if n == 1:
        axes2 = [axes2]

    for ax, key in zip(axes2, ordered_keys):
        img = tensor_to_image(find_tensor(torch.load(mapping[key], map_location="cpu")))
        ax.imshow(img)
        ax.set_title(f"step={key}")
        ax.axis("off")

    fig2.suptitle(
        "Patch evolution across saved checkpoints",
        fontsize=11,
    )
    fig2.tight_layout()

    fig2_path = out_dir / "patch_evolution_saved_steps.png"
    fig2.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig2)

    meta = {
        "run_dir": str(run_dir),
        "baseline_selected": baseline_key,
        "baseline_reason": "explicit" if args.baseline_step is not None and args.baseline_step == baseline_key else (
            "found_step_0" if baseline_key == 0 else "found_step_49" if baseline_key == 49 else "earliest_available_step"
        ),
        "current_selected": "last",
        "saved_steps": ordered_keys,
        "output_images": [str(fig1_path), str(fig2_path)],
        "diff_stats": {
            "mean_abs_delta": l1,
            "max_abs_delta": linf,
        },
    }

    meta_path = out_dir / "patch_viz_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(str(fig1_path))
    print(str(fig2_path))
    print(str(meta_path))


if __name__ == "__main__":
    main()
