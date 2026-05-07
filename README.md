# SynthesizedECG

This repository provides scripts for ECG preprocessing, SSL/DGM model training, and synthetic ECG generation.

## Project overview

- Data preparation
  - `src/prep/dataset`: converts raw ECG datasets to train/val/test pickles
  - `src/prep/SimECG-N`: Python-based 1D ECG simulator and data export
  - `src/prep/SimECG-M`: simulator binary wrapper (`ecgsyn`) and conversion to pickles
- SSL pretraining and transfer learning
  - `src/ssl_pt`: SSL pretraining (MAE) entry points and YAML builders
  - `src/ssl_clf`: classifier experiments and SSL-weight transfer workflows
- Generative baselines
  - `src/dgms`: GAN/VAE/DDPM training and generation
  - `src/diff_sssd`: SSSD-ECG workflow for diffusion pretraining and sample generation
  - `src/diff_sssd/sssd_standalone`: standalone SSSD-ECG package

The repository uses a central `config.yaml` at the repository root to keep paths and shared experiment settings.

## Environment setup

1. Prepare Docker environment
   - Ensure Docker and NVIDIA runtime are available on your host.
   - From the repository root, run:
   ```bash
   bash invoke_container.sh build
   ```
   This builds the image and starts a container with data/output directories mounted from the repo.

2. Enter container shell
   - The command above opens an interactive shell automatically at the end.
   - If you need to re-enter later:
   ```bash
   bash invoke_container.sh restart
   ```

3. Verify repository and paths inside container
   ```bash
   ls /home/$(whoami)/SynthesizedECG
   ```

4. Configure `config.yaml`
   - Update all paths that differ from your environment:
     - `experiment.path` (save roots and yaml locations)
     - `settings.*.src` for PTB-XL/CPSC/G12EC source roots
     - `settings.common.save_root` if needed
   - Use the helper script above (`scripts/download_raw_data.sh`) to place raw datasets.

Note:
- This repository provides all module-level Python dependencies inside the container image used by `invoke_container.sh build`.

## Raw data placement for reproducibility

For reproducibility, the repository expects raw PTB-XL, CPSC, and G12EC data to be available under `raw_data` at the repository root.  
The paths under `experiment.path` are configured as relative paths to this layout, so keeping the raw datasets in this location prevents config mismatch.

The following `config.yaml` entries are environment-dependent by default and should be updated for your environment:
- `settings.g12ec.src`
- `settings.cpsc.src`
- `settings.ptbxl.src`

Recommended local layout:
- PTB-XL: `raw_data/PTBXL/1.0.1/ptbxl`
- G12EC: `raw_data/G12EC/WFDB`
- CPSC2018: `raw_data/CPSC2018`

Minimal required file layout:
- PTB-XL
  - `raw_data/PTBXL/1.0.1/ptbxl/ptbxl_database.csv`
  - `raw_data/PTBXL/1.0.1/ptbxl/scp_statements.csv`
  - `raw_data/PTBXL/1.0.1/ptbxl/records500/<ID_GROUP>/<id>_hr`
- G12EC
  - `raw_data/G12EC/WFDB_v230901/*.hea`
  - `raw_data/G12EC/WFDB_v230901/*.dat`
- CPSC2018
  - `raw_data/CPSC2018/TrainingSet3/REFERENCE.csv`
  - `raw_data/CPSC2018/*/<recording_id>.mat`

`config.yaml` currently uses `experiment.path` entries such as `../../raw_data`, so placing your dataset under this `raw_data` tree or linking to it is expected.

You can use the new helper script [scripts/download_raw_data.sh](SynthesizedECG/scripts/download_raw_data.sh) to prepare this layout.

## Data download helper script

The following script is provided:
- `scripts/download_raw_data.sh`

The script validates the expected raw-data files after each step and reports missing items.

## Recommended workflow

1. Download public data
2. Prepare public data
3. Preprocess SimECG-N and SimECG-M
   1. bash scripts/prepare_simecg-n.sh
   2. For SimECG-N specifically, run:
      - `cd src/prep/SimECG-N`
      - `python gen_ecg.py`
4. (If needed) Train and generate data in `src/dgms`, `src/diff_sssd`
5. Pretrain SSL backbone in `src/ssl_pt`
6. Run downstream evaluation tasks in `src/ssl_clf`

## NeurIPS 2026 Evaluations & Datasets assets

The repository introduces one new dataset asset for submission:
- SimECG-N synthetic ECG corpus: `data/manifest_simecg_n.csv`
- Croissant metadata: `metadata/simecg_n_croissant.json`
- Data card: `DATA_CARD_SimECG-N.md`
- Code license file: `LICENSE`
- SimECG-N corpus data license file: `LICENSE_DATA`

Reproduction commands for SimECG-N:
- `bash scripts/prepare_simecg-n.sh`
- Or run from module directory:
  - `cd src/prep/SimECG-N`
  - `python gen_ecg.py`

By default, `src/prep/SimECG-N/gen_ecg.py` uses:
- seed `7`
- `syn_mode = "pt"`
- `cfg` setting `settings.syn_ecg.syncfg = "syn_ecg-04"`
- `cfg` setting `settings.common.n_syn.pt = 1_000`
- `cfg` setting `settings.common.val_size = 0.1`
- `cfg` setting `settings.common.save_root = ../../../raw_data`

Update `config.yaml` values only if you need a different save root or generation setting.

## Important notes

- Always run each module from its documented directory to keep relative config paths valid.
- Before each experiment, confirm output directories exist and are writable.
- Many scripts write large intermediate files; monitor disk usage in advance.

For each module, refer to the README inside its directory for concrete arguments and examples.
