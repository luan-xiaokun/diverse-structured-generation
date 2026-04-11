"""Build helper for the optional native WD-shift kernel extension.

This script compiles the C/OpenMP backend under ``native/wd_kernel`` and
stores artifacts under ``build/native/wd_kernel``.

Usage:
    python scripts/build_wd_kernel.py
    python scripts/build_wd_kernel.py --clean
    python scripts/build_wd_kernel.py --check
    python scripts/build_wd_kernel.py --copy-to-package
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build optional native WD-shift kernel extension."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run `make clean` before building.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check whether the built library exists; do not build.",
    )
    parser.add_argument(
        "--copy-to-package",
        action="store_true",
        help=(
            "Also copy the built library to src/diverse_guide/_native "
            "(useful for packaging workflows)."
        ),
    )
    return parser.parse_args()


def _lib_name_for_platform(system: str) -> str:
    return "wd_kernel.dll" if system == "Windows" else "wd_kernel.so"


def _print_platform_hint(system: str) -> None:
    if system == "Darwin":
        print("Hint (macOS): install OpenMP first with `brew install libomp`.")
    elif system == "Linux":
        print(
            "Hint (Linux): ensure gcc with OpenMP is available "
            "(for Ubuntu/Debian: `sudo apt-get install build-essential`)."
        )
    elif system == "Windows":
        print(
            "Hint (Windows): run this in MSYS2 MinGW64 shell with "
            "`mingw-w64-x86_64-gcc` installed, or use WSL2."
        )


def _run_make(native_dir: Path, target: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", target],
        cwd=native_dir,
        check=False,
        text=True,
        capture_output=True,
    )


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    native_dir = root / "native" / "wd_kernel"
    build_dir = root / "build" / "native" / "wd_kernel"
    package_native_dir = root / "src" / "diverse_guide" / "_native"

    system = platform.system()
    lib_name = _lib_name_for_platform(system)
    build_lib_path = build_dir / lib_name
    package_lib_path = package_native_dir / lib_name
    source_lib_path = native_dir / lib_name

    if not native_dir.exists():
        print(f"Error: missing directory: {native_dir}")
        return 2

    if args.check:
        if build_lib_path.exists():
            print(f"Found native kernel library: {build_lib_path}")
            return 0
        if package_lib_path.exists():
            print(f"Found native kernel library: {package_lib_path}")
            return 0
        print(
            "Native kernel library not found in either:\n"
            f"  - {build_lib_path}\n"
            f"  - {package_lib_path}"
        )
        _print_platform_hint(system)
        return 1

    if args.clean:
        clean_res = _run_make(native_dir, "clean")
        if clean_res.returncode != 0:
            print(clean_res.stdout, end="")
            print(clean_res.stderr, end="", file=sys.stderr)
            print("Failed to clean native kernel build artifacts.", file=sys.stderr)
            return clean_res.returncode
        if build_lib_path.exists():
            build_lib_path.unlink()
        if package_lib_path.exists():
            package_lib_path.unlink()

    print(f"Building native WD kernel in: {native_dir}")
    build_res = _run_make(native_dir, "all")
    if build_res.returncode != 0:
        print(build_res.stdout, end="")
        print(build_res.stderr, end="", file=sys.stderr)
        print("Build failed.", file=sys.stderr)
        _print_platform_hint(system)
        return build_res.returncode

    if build_lib_path.exists():
        print(f"Build succeeded: {build_lib_path}")
    elif source_lib_path.exists():
        # Backward compatibility: if Makefile still emits into native dir,
        # move it into the canonical build output location.
        build_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_lib_path, build_lib_path)
        source_lib_path.unlink(missing_ok=True)
        print(f"Build succeeded: {build_lib_path}")
    else:
        print(
            (
                "Build command finished, but expected artifact was "
                "not found in either:\n"
                f"  - {build_lib_path}\n"
                f"  - {source_lib_path}"
            ),
            file=sys.stderr,
        )
        return 1

    if args.copy_to_package:
        package_native_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(build_lib_path, package_lib_path)
        print(f"Copied to package-native path: {package_lib_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
