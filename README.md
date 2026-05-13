# ICML 2026 Sequential Design Demo

This repository contains a compact implementation demo for the sequential design method used in our ICML 2026 paper. The code builds Monte Carlo Q-value datasets, trains neural Q-function approximators, runs backward sequential decision training, and evaluates the learned sequential designs against randomized assignment baselines.

## Repository Structure

| File | Purpose |
| --- | --- |
| `utils_self.py` | Shared utilities for data generation, Legendre basis construction, robust and non-robust objective evaluation, model creation, metrics, and dataset serialization. |
| `Sequential_DP_robust.py` | Builds the last-stage Q-value dataset for the robust objective. |
| `Sequential_DP_nonrobust.py` | Builds the last-stage Q-value dataset for the non-robust variance/imbalance objective. |
| `Training_Q_last_net_robust_nonscal.py` | Trains the last-stage robust Q network. |
| `Training_Q_last_net_nonrobust_nonscal.py` | Trains the last-stage non-robust Q network. |
| `Training_Q_net_iteration_robust_nonscal.py` | Runs backward iterative training for robust Q networks. |
| `Training_Q_net_iteration_nonrobust_nonscal.py` | Runs backward iterative training for non-robust Q networks. |
| `Evaluation_sequential_design.py` | Evaluates sequential robust, sequential non-robust, and randomized designs. |
| `run_all_robust.sh` | End-to-end robust pipeline. |
| `run_all_nonrobust.sh` | End-to-end non-robust pipeline. |

## Requirements

The implementation is written for Python 3. Recommended packages:

```bash
pip install numpy pandas scipy patsy torch matplotlib seaborn
```

CUDA is optional but recommended for the larger default Monte Carlo settings. The scripts automatically fall back to CPU when CUDA is unavailable.

## Quick Start

Run the robust pipeline:

```bash
bash run_all_robust.sh
```

Run the non-robust pipeline:

```bash
bash run_all_nonrobust.sh
```

Both shell scripts expose the main configuration through environment variables. For example:

```bash
GPU_ID=0 N_EXP=20 B=8000 B_SHARE=8000 B_ITER=8000 bash run_all_robust.sh
```

## Pipeline

1. Build the final-stage Q-value dataset:

```bash
python Sequential_DP_robust.py --gpu 0 --n_exp 20 --B 8000 --B_share 8000 --type_obj obj
python Sequential_DP_nonrobust.py --gpu 1 --n_exp 20 --B 8000 --B_share 8000 --type_obj var
```

2. Train the last-stage Q network:

```bash
python Training_Q_last_net_robust_nonscal.py --gpu 0 --n_exp 20 --type_obj obj --out_dir plots_output_obj
python Training_Q_last_net_nonrobust_nonscal.py --gpu 1 --n_exp 20 --type_obj var --out_dir plots_output_var
```

3. Run backward iterative training:

```bash
python Training_Q_net_iteration_robust_nonscal.py --gpu 0 --n_exp 20 --obj_type obj --out_dir plots_output_obj
python Training_Q_net_iteration_nonrobust_nonscal.py --gpu 1 --n_exp 20 --obj_type var --out_dir plots_output_var
```

4. Evaluate the learned sequential designs:

```bash
python Evaluation_sequential_design.py
```

## Outputs

Typical generated artifacts include:

- `Qval_dataset_{n_exp}_{type_obj}.txt`: tab-delimited Q-value training datasets.
- `plots_output_*/*.pth`: trained model checkpoints.
- `plots_output_*/*.png`: training curves and prediction diagnostics.
- `logs/*.log`: runtime and training logs.
- `results_output/*.csv`: evaluation summaries and per-run estimates.
- `results_output/*.png`: evaluation box plots.

These files can be large. For GitHub release, consider excluding generated datasets, checkpoints, logs, and plots unless they are intentionally provided as reproducibility artifacts.

## Notes

- Default Monte Carlo and training sizes are set for paper-scale experiments and may require substantial runtime and GPU memory.
- Chunking arguments such as `--bk_batch_size`, `B_SHARE`, and `B_ITER` can be reduced for smaller machines.
- Random seeds are exposed through script arguments or shell environment variables to support reproducible runs.
