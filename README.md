# pFLEDL

Research implementation of pFLEDL, with evidential calibration followed by cross-entropy local training in a federated learning simulation. The implementation is built on PFLlib. The command-line algorithm name is `FedEDL`.

This is a draft code release for the manuscript revision. Paper title, citation, repository URL, and a validated release tag will be added once finalized. This checkout contains ongoing experiments; it is not yet a frozen reproduction release.

## Code layout

- `system/main.py`: experiment entry point, model selection, seed handling, and algorithm dispatch.
- `system/flcore/clients/clientedl.py`: pFLEDL calibration and local training; `EDL_Loss`.
- `system/flcore/servers/serveredl.py`: client selection, model initialization, training, and aggregation.
- `system/flcore/servers/serverbase.py`: shared evaluation, stopping, and timing utilities.
- `system/flcore/trainmodel/`: model definitions.
- `system/utils/`: data loading and result utilities.
- `dataset/generate_FEMNIST_pat.py`: pathological FEMNIST partition generation.
- `system/Computation_WM.ipynb`: computation-cost analysis of archived FEMNIST experiments.

## Installation

Use Python 3.12 as a starting point, matching the local experiment environment. Install compatible PyTorch and torchvision builds for your GPU/CUDA environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy
python -m pip install -r system/requirements.txt
```

For notebook analysis, also install `jupyterlab`. Package versions in the existing requirements file are not pinned; a clean-install validation and versioned environment specification remain necessary before freezing the release. The current EDL loss uses explicit CUDA tensor creation, so use a CUDA device for this implementation.

## Dataset preparation

Training expects generated client files at `dataset/<dataset-name>/train/<client-id>.npz` and `test/<client-id>.npz`. Run experiments from `system/` so relative dataset paths resolve correctly. Raw data, generated client datasets, checkpoints, and experiment outputs are excluded from Git by default.

### FEMNIST

Follow the LEAF download/preprocessing instructions in `dataset/generate_FEMNIST.py`, placing the processed metadata and raw images under `dataset/FEMNIST/`. Then:

```bash
cd dataset
python generate_FEMNIST_pat.py 20 4 unbalance
cd ..
```

This creates `FEMNIST-partition_pat-clients_20-cpc_4-balance_False-full-train_075`.
The generator pools writers and creates a pathological label-skew partition: 20 clients, 62 classes, four clients per class, unbalanced sample counts, and a 75/25 train/test split. This experimental partition does **not** preserve natural writer-based clients. Here `cpc_4` means clients per class, not four classes per client.

### CIFAR-100 and other manuscript datasets

The current checkout contains generated partitions but lacks the CIFAR-100 generation script. The command below requires the named partition to have already been prepared. Add the exact partition-generation script and instructions before claiming a complete reproduction package. Do not substitute a different split while retaining the historical partition name.

## Run pFLEDL

For the manuscript CIFAR-100 pathological partition, with three runs and early stopping disabled:

```bash
cd system
python -u main.py \
  --algorithm=FedEDL \
  --dataset=Cifar100-partition_pat-clients_20-niid_True-balance_False-batch_10-classpc_10-alpha_01-train_075 \
  --num_classes=100 --num_clients=20 --join_ratio=1 \
  --model=CNN --global_rounds=500 --times=3 --seed=941 \
  --auto_break=false --goal=cifar100_pfledl_w30_c5 \
  2>&1 | tee cifar100_pfledl_w30_c5.log
```

Run indices 0, 1, and 2 use seeds 941, 942, and 943. The server loop includes round 0, so `--global_rounds=500` executes 501 iterations. Early stopping defaults to enabled; explicitly disable it for the fixed-budget CIFAR-100 experiments. Use a distinct `--goal` for each ablation variant to distinguish result filenames. Reusing the same algorithm, dataset, goal, and run index can overwrite results.

For FEMNIST, replace the dataset name with the generated FEMNIST directory and set `--num_classes=62`. Set `--join_ratio=0.5` for 50% participation. Use the stopping configuration matching the experiment being reproduced.

## Calibration settings and ablations

Calibration parameters are currently set in `clientEDL.__init__`, rather than exposed as command-line options:

| Code setting | Meaning | Current value |
|---|---|---:|
| `self.le` | Calibration epochs | 5 |
| `self.warm_up` | Calibration initialization threshold | 30 |
| `self.lam` | EDL regularization weight | 0.01 |
| `self.beta` | Calibration loss multiplier | 0.02 |

With the current global-round indexing, calibration starts when `round_index + 1 >= self.warm_up`. A threshold of 30 corresponds to round index 29. Change `self.le` to 0, 1, 3, or 5 for the calibration-epoch ablation, and use the same seeds across variants. Record a source snapshot and all settings for every run: console arguments alone do not capture these hardcoded values.

`self.rand_percent` currently reads 50, but the active calibration loader reads the full local training data without applying that field. Do not interpret it as a verified 50% calibration subsample. Historical ablation variants must be documented against their actual implementation snapshots.

## Results and computational-cost analysis

Training writes metrics and timing data under `system/results/`. Timing CSVs contain round index, duration in seconds, and cumulative recorded runtime. Runtime covers sequential simulation work inside the round timer, including calibration and evaluation, but does not measure physical distributed network latency or complete process runtime outside those boundaries.

`system/Computation_WM.ipynb` analyzes archived results using explicit warm-up, participation, and calibration mappings. Its current paths reference the original server and must be updated for another machine. The archived timing inputs are not bundled by default. To make the analysis portable, distribute a documented timing-data bundle and replace server-specific paths before the final release.

## Attribution and license

This code contains PFLlib-derived source files with existing GNU GPL version 2-or-later copyright/license notices, including attribution to Jianqing Zhang. Preserve those notices. A complete release must include the applicable upstream license text and third-party acknowledgments; do not replace the inherited license with an unrelated permissive license.

## Citation and availability

The final paper citation and public repository URL are pending. Only describe the code as publicly available in the manuscript/rebuttal after the repository is public and its contents and reproduction instructions have been verified.
