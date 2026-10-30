"""Verify that Hugging Face causal LMs load locally and generate on CUDA."""

from __future__ import annotations

import argparse
import gc
import json
from time import perf_counter
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODELS = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-8B",
    "microsoft/Phi-4-mini-instruct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local CUDA smoke test for downloaded Hugging Face models."
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model id to verify. Can be repeated. Defaults to the download_models.sh set.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("auto", "float16", "bfloat16", "float32"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Return the word OK.")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def verify_one(args: argparse.Namespace, model_id: str) -> dict[str, Any]:
    started = perf_counter()
    result: dict[str, Any] = {
        "model": model_id,
        "device": args.device,
        "dtype": args.dtype,
        "ok": False,
    }
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        load_started = perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=args.local_files_only,
            torch_dtype=dtype_from_name(args.dtype),
            trust_remote_code=args.trust_remote_code,
        )
        if args.device == "cuda":
            model = model.to("cuda")
        model.eval()
        load_seconds = perf_counter() - load_started

        inputs = tokenizer(args.prompt, return_tensors="pt")
        if args.device == "cuda":
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        generate_started = perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        generate_seconds = perf_counter() - generate_started
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        result.update(
            {
                "ok": True,
                "load_seconds": round(load_seconds, 3),
                "generate_seconds": round(generate_seconds, 3),
                "output": decoded,
            }
        )
    except Exception as exc:  # noqa: BLE001 - CLI should report every model.
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["elapsed_seconds"] = round(perf_counter() - started, 3)
        try:
            del model  # type: ignore[name-defined]
        except Exception:
            pass
        cleanup()
    return result


def main() -> None:
    args = parse_args()
    models = tuple(args.models) if args.models else DEFAULT_MODELS
    print(
        json.dumps(
            {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
                "models": models,
            },
            ensure_ascii=True,
        )
    )
    results = [verify_one(args, model_id) for model_id in models]
    for result in results:
        print(json.dumps(result, ensure_ascii=True))
    if not all(result["ok"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
