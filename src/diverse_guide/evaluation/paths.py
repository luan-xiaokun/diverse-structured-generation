"""Shared path utilities for generation and evaluation scripts."""

from pathlib import Path


def get_data_dir_path(args) -> Path:
    segments = [args.model.split("/")[-1].lower()]
    baseline_backend = getattr(args, "baseline_backend", "internal")
    ablation_component = getattr(args, "ablation_component", None)
    if not args.baseline and ablation_component is not None:
        segments.append(f"no_{ablation_component}")
    if args.top_k is not None:
        segments.append(f"top_k_{args.top_k}")
    if args.top_p is not None:
        segments.append(f"top_p_{args.top_p}")
    if args.temperature is not None:
        segments.append(f"temperature_{args.temperature}")
    data_dir = Path("data/baseline") if args.baseline else Path("data/diverse")
    if getattr(args, "baseline", False) and baseline_backend != "internal":
        data_dir = data_dir / baseline_backend
    return data_dir / "-".join(segments)
