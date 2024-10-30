# ONNX Model Merger

A tool to merge ONNX models, specifically designed to combine a backbone with its head while maintaining proper node connections.

This is purely a proof of concept, not a all production-ready.

## Features

- Merges two ONNX models (backbone and head)
- Automatically handles node name prefixing to avoid conflicts
- Includes model simplification using ONNX-Simplifier
- Configurable input/output node mapping

## Installation

The project uses [devenv](https://devenv.sh) with [uv](https://docs.astral.sh/uv/) package manager.

0. Install Nix and Devenv (and maybe direnv if you want).

1. Optional if you have `direnv` - Enter in the devenv shell:
   ```bash
   devenv shell
   ```

2. Sync the dependencies if needed:
   ```bash
   uv sync
   ```

## Usage

The merger tool takes two ONNX models as input and combines them:

```bash
python merger.py <backbone_model.onnx> <head_model.onnx> [options]
```

If you are not the the environment, you may want to use:

```bash
uv run python merger.py
```

### Options

- `-o`, `--out_node_name`: Output node name from the backbone (default: "Identity_1:0")
- `-i`, `--in_node_name`: Input node name for the head (default: "args_tf_0")

### Example

```bash
uv run python merger.py example/backbone.onnx example/fc.onnx
```

The merged model will be saved as `<backbone_name>_<head_name>.onnx` in the same directory as the backbone model.
