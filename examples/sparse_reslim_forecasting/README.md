# Sparse-Reslim forecasting example

This directory is a small, self-contained example of **deterministic weather
forecasting** with Sparse-Reslim. It is intended to show the core method and the
launch workflow without copying the complete ECCV research code into ORBIT-2.
EDM and diffusion-based generation are intentionally not included here.

For the complete paper implementation, experiment configurations, and advanced
training code, see the
[full Sparse-Reslim ECCV repository](https://github.com/janet-sw/Sparse-Reslim).

## What the example does

`model.py` implements the forecasting path in four steps:

1. Embed each weather variable into spatial patch tokens and aggregate the
   variables at each location.
2. Process all tokens through the early dense Transformer blocks.
3. Send only `keep_ratio` of the tokens through the middle sparse blocks. The
   sparse residual updates are scattered back to their original locations;
   skipped tokens remain on an identity path.
4. Process the restored dense grid through the late blocks, decode the forecast,
   and add the parallel convolutional residual forecast.

The default route is **1 dense → 4 sparse → 1 dense**, keeping 25% of the patch
tokens in the four middle blocks. Routing is parameter-free and independently
samples a spatial subset for each item in the batch.

## Installation

Follow the PyTorch installation for your AMD or NVIDIA system in the root
[ORBIT-2 README](../../README.md), then install ORBIT-2 from the repository root:

```bash
pip install -e .
```

## Quick smoke test

The smoke test uses synthetic tensors, does not need ERA5 data, and checks both
the forward and backward passes:

```bash
python examples/sparse_reslim_forecasting/train.py --smoke-test
```

A successful run prints a forecast shape, the sparse token count, and a finite
loss value.

## ERA5 data layout

The training example reads the same split-oriented NPZ layout used by ORBIT-2:

```text
ERA5_DIR/
├── normalize_mean.npz
├── normalize_std.npz
├── train/*.npz
├── val/*.npz
└── test/*.npz
```

Each yearly or monthly NPZ file must contain the requested variable keys. Each
array should have shape `[time, latitude, longitude]` or
`[time, 1, latitude, longitude]`. The normalization files must contain a mean
and standard deviation for every requested variable.

## Launch forecasting

From the repository root, launch the default single-variable, six-timestep
forecast on one device:

```bash
bash examples/sparse_reslim_forecasting/launch.sh /path/to/ERA5_DIR \
  --max-epochs 30 \
  --batch-size 16 \
  --pred-range 6
```

The same command can be run directly with Python:

```bash
python examples/sparse_reslim_forecasting/train.py /path/to/ERA5_DIR \
  --max-epochs 30
```

By default, both the input and target are `2m_temperature`. To forecast multiple
variables, list them explicitly; every output variable must also be present in
the inputs because the model uses a residual forecasting path:

```bash
bash examples/sparse_reslim_forecasting/launch.sh /path/to/ERA5_DIR \
  --input-vars 2m_temperature 10m_u_component_of_wind \
  --output-vars 2m_temperature 10m_u_component_of_wind
```

Useful options include:

- `--history`: number of input timesteps (default: `1`)
- `--window`: spacing between history timesteps (default: `1`)
- `--pred-range`: forecast lead in stored timesteps (default: `6`)
- `--keep-ratio`: fraction of tokens processed by sparse blocks (default: `0.25`)
- `--num-dense-early` and `--num-sparse-middle`: block schedule
- `--accelerator cpu|gpu|auto` and `--devices`: Lightning device selection
- `--limit-train-batches`: short end-to-end debugging runs
- `--output-dir`: logs and best checkpoint location

This example deliberately targets one CPU or GPU for a clear first run. The
[full Sparse-Reslim repository](https://github.com/janet-sw/Sparse-Reslim)
contains the paper-scale configurations and distributed training workflow.
