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
tokens in the four middle blocks. Routing is parameter-free. While training it
independently samples a spatial subset for each item in the batch; in `eval()`
mode it reuses one fixed subset of the same size, so validation, testing, and
inference return the same forecast for the same input.

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
`[time, 1, latitude, longitude]`. The normalization files must contain a single
scalar mean and standard deviation for every requested variable, so only
single-level variables are supported.

Files are read one at a time and samples never cross a file boundary, so every
NPZ needs at least `(history - 1) * window + pred_range + 1` timesteps. Shorter
files in a split directory, such as a `climatology.npz`, contribute no samples
and are simply skipped.

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

## Short end-to-end check

To verify data loading, the forward and backward passes, checkpointing, and
logging without waiting for a full epoch, cap every stage:

```bash
python examples/sparse_reslim_forecasting/train.py /path/to/ERA5_DIR \
  --max-epochs 1 --patience 0 \
  --limit-train-batches 20 --limit-val-batches 5 --limit-test-batches 5
```

Because the splits are streamed as `IterableDataset`s, `--limit-val-batches`
and `--limit-test-batches` are what keep the validation and test stages short;
`--limit-train-batches` alone still walks the whole validation split.

`--limit-val-batches 0` switches validation off entirely. Early stopping and
best-checkpoint selection both monitor `val/mse`, so they are disabled together
with it and the last epoch is checkpointed instead.

## Useful options

Data and forecast setup:

- `--input-vars` and `--output-vars`: variable keys to read (default:
  `2m_temperature`); every output variable must also be an input
- `--history`: number of input timesteps (default: `1`)
- `--window`: spacing between history timesteps (default: `1`)
- `--pred-range`: forecast lead in stored timesteps (default: `6`)
- `--num-workers`: dataloader workers; the NPZ files are sharded across them
  (default: `2`)

Model size and routing:

- `--keep-ratio`: fraction of tokens processed by sparse blocks (default: `0.25`)
- `--num-dense-early` and `--num-sparse-middle`: block schedule; their sum must
  not exceed `--depth`, and the remainder becomes the late dense blocks
- `--patch-size`: patch side length (default: `2`); the grid height and width
  must both be divisible by it, and it is the main control over token count and
  therefore over memory and step time
- `--depth`, `--embed-dim`, `--num-heads`: Transformer size (default: `6`,
  `128`, `4`)

Optimisation and runtime:

- `--batch-size` (default: `16`), `--max-epochs` (default: `30`)
- `--lr` (default: `5e-4`), `--weight-decay` (default: `1e-5`)
- `--patience`: early-stopping patience on `val/mse`; `0` disables it
  (default: `5`)
- `--seed`: seeds weight initialisation, training-time routing, and the
  dataloader workers (default: `0`); the evaluation routing subset is fixed
  independently of it
- `--accelerator cpu|gpu|auto` and `--devices`: Lightning device selection;
  the streaming dataset only shards its files over dataloader workers, so
  `--devices` must stay `1` and the script refuses anything larger
- `--limit-train-batches`, `--limit-val-batches`, `--limit-test-batches`: short
  end-to-end debugging runs
- `--output-dir`: logs and best checkpoint location

This example deliberately targets one CPU or GPU for a clear first run. The
[full Sparse-Reslim repository](https://github.com/janet-sw/Sparse-Reslim)
contains the paper-scale configurations and distributed training workflow.
