# Diverse Structured Generation

This repository implements a diversity enhancing method for structured generation of large language models.
Currently, it supports structured generation with regular expression constraints, but theoretically it also applies to context-free grammars.
Some code is adapted from [Outlines](https://github.com/dottxt-ai/outlines), and [uthash](https://github.com/troydhanson/uthash) is used for implementing the weighted degree kernel with shifts.

## Setup

We recommend using `uv` to set up this project.
`cargo` is required to build the Rust extension.

```bash
# install Python dependencies
uv sync
# activate the virtual environment
source .venv/bin/activate
# build and install the minimal_dfa Rust/Python extension
cd minimal_dfa
make build-extension-release
make install
# build the wd kernel with shift C extension
cd ../wd_kernel
make
```

In addition, large language models need to be downloaded and placed in `models/` directory.
Generated texts will be saved in `data/` directory.

## Run

We recommend using `poethepoet` to run the script.

```bash
# uv run poe gen --help to see detailed usage
# uv run poe eval --help to see detailed usage
uv run poe gen css-color
uv run poe eval css-color
```
