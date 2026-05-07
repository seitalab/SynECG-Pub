# Experimental ECG dataset preparation (`src.prep.dataset`)

This directory creates split Pickle/NumPy data from raw ECG records for experiments using:
- `ptbxl.py`
- `cpsc.py`
- `g12ec.py`
- `ptbxl_for_sssd.py`

Set `settings.common.split` and raw data paths in `config.yaml` before running.

## 1. Prerequisites

- Python 3.x
- Required libraries:
  - `numpy`
  - `pandas`
  - `pyyaml`
  - `scipy`
  - `scikit-learn`
  - `wfdb`
  - `tqdm`
- Run from `src/prep/dataset` because scripts use `cfg_file = "../../../config.yaml"` via relative path.

## 2. Main `config.yaml` settings

`settings.common.save_root`
- root output directory for generated data
- downstream pipelines read from `experiment.path.data_root`, so keep paths aligned in practice

`split`
- `test.size` and `test.seed` control train/val/test split
- `train_val.seeds` defines seeds for `train_seedXXXX.pkl` and `val_seedXXXX.pkl`

`settings.cpsc.src` and `settings.cpsc.reference`
- CPSC raw data root and `TrainingSet3/REFERENCE.csv`

`settings.g12ec.src` and `settings.g12ec.dx_to_code`
- WFDB directory for `*.hea` / `*.dat`
- diagnosis code dictionary

`settings.ptbxl.src`
- PTBXL records directory `records500/`
- required same-level files `ptbxl_database.csv` and `scp_statements.csv`

## 3. Raw data assumptions

- **CPSC**
  - `REFERENCE.csv` includes `Recording`, `First_label`, `Second_label`, `Third_label`
  - signal loading extracts `lead_idx` from WFDB `.mat` field `["ECG"][0][0][2]`
- **G12EC**
  - WFDB `*.hea` / `*.dat` are readable
  - target diagnoses are selected if SNOMED code is included in `data.comments[2]`
- **PTBXL**
  - `ptbxl_database.csv` must contain `ecg_id`, `scp_codes`
  - `records500/<idgrp>/<id>_hr` must be readable (only 5000-length records are used)
- **PTBXL for SSSD**
  - Uses the same `ptbxl_database.csv` assumption and additionally saves 100Hz records from `records100`

## 4. Generation sequence

```bash
cd src/prep/dataset
python g12ec.py
python cpsc.py
python ptbxl.py
python ptbxl_for_sssd.py
```

Edit `__main__` targets to change diagnosis categories per run.

- `ptbxl.py`
  - iterates `NORM, AFIB, PVC, PAC, AFLT, WPW`
  - some classes use `thres=0`, others use `thres=100`
- `cpsc.py`
  - iterates `NORM, AF, IAVB, PAC, PVC, STD, RBBB`
- `g12ec.py`
  - currently `VPB` only (edit list if needed)
- `ptbxl_for_sssd.py`
  - builds `PTBXL-sssd_data` with multilabel coverage

## 5. Outputs

- **common outputs (`prep_base.PrepBase`)**
  - `train`: `train_seed0001.pkl`, `train_seed0002.pkl`, ...
  - `val`: `val_seed0001.pkl`, `val_seed0002.pkl`, ...
  - `test`: `test.pkl`
  - save root:
    - `save_root/PTBXL-<class>` / `save_root/G12EC-<class>` / `save_root/CPSC-<class>`
- **PTBXL-SSSD**
  - `train_seedXXXX.npy`, `val_seedXXXX.npy`, `test.npy`
  - matching `_labels.npy` files
  - save root: `save_root/PTBXL-sssd_data`

## 6. How this connects to experiments

- SSL/classification experiments usually load:
  - `train_seed{seed:04d}.pkl`
  - `val_seed{seed:04d}.pkl`
  - `test.pkl`
- Typical runs use `seed=1`.
- For SSSD `self_split`, code may reference `seed0007`, so ensure `7` is included in `cfg["split"]["train_val"]["seeds"]` if needed.

## 7. Troubleshooting

- `FileNotFoundError: ../../../config.yaml`
  - wrong working directory; always run from `src/prep/dataset`.
- `ptbxl_for_sssd.py` takes long
  - it scans both `records500` and `records100`, which is heavier for large datasets.
- CPSC `target_dx="ALL"`
  - current implementation expects filtering upstream in `make_dataset()`, so additional filtering may be needed outside `__main__`.
