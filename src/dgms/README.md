# DGM training and sample synthesis guide (`src.dgms`)

## 1. Prerequisites

- `src/dgms` scripts load `../../config.yaml` via relative path.
  Run commands from `src/dgms`.
- Experiments import modules under `codes`, so avoid running from repository root as `python src/dgms/experiment.py`.
- This directory includes mixed pipelines; behavior depends on each `dgm.yaml` and `gen.yaml`.

```bash
cd src/dgms
```

## 2. Training procedure

### 2.1 Experiment definition

`--exp` points to YAML files in:
- `src/dgms/resources/dgm_yamls/d00s/dgm0001.yaml`
- ID format: `dgm_xYs/dgmXYYY.yaml`  
  for example, `exp=9999` -> `dgm99s/dgm9999.yaml`

### 2.2 Execution

```bash
cd src/dgms
python experiment.py --exp 1 --device cuda:0
python experiment.py --exp 1 --device cpu --debug
```

When `--debug` is used, `ExperimentManager` shortens run settings (including `data_lim`, `val_lim`, `total_samples`, and scheduling settings).

### 2.3 Training flow

1. `ExperimentManager` loads YAML from `experiment.path.dgm_yaml_loc`, merges `generatives` and `experiment` sections in `config.yaml`, and builds parameters for `run_train`.
2. `run_train` selects `Trainer` or `GANTrainer` (`GANTrainer` for `dcgan` / `wgan`, otherwise `Trainer`).
3. DataLoader is built from `.pkl` under `experiment.path.data_root` with dataset `PTBXL-ALL`, `train`, `val`.
4. `prepare_model` initializes model with `dgm`; optimizer setup and training starts.
5. Evaluation, checkpointing, and periodic sample visualization are executed.

### 2.4 Expected outputs

Saved under `experiment.path.save_root_gen` (for example `dgm00s/dgm0001/<timestamp>`):
- `params.pkl`, `params.txt`
- `exp_config.yaml`
- `train_scores.json`, `eval_scores.json`
- `net.pth`
- `best_score.txt`
- `interims/` checkpoints when `save_model_every` matches
- `inputs/`, `dumps/`, `best_samples/` for visualizations

## 3. Data generation procedure

### 3.1 Generation definition

`--exp` maps to `gen.yaml` in:
- `src/dgms/resources/gen_yamls/g00s/gen0001.yaml`
- for example `--exp 9999` -> `gen99s/gen9999.yaml`

`generate.py` resolves pretrained weights via `weight_file_key` (for example `vae/v01`, `dcgan/v01`) and reads `config.yaml -> generatives.model_path`.

### 3.2 Execution

```bash
cd src/dgms
python generate.py --exp 1 --device cuda:0
python generate.py --exp 1 --device cpu --debug
```

### 3.3 Generation flow

1. `SampleGenerationManager` loads `gen.yaml` and builds `n_total_samples` and save root.
2. Run `run_generate` for `train` and `val`.
3. `Generator` loads weights and calls `model.generate(z)`.
4. `GeneratedSampleManager` writes split PKL files as `samples/idx000001.pkl`.

### 3.4 Expected output

Under `experiment.path.save_root_gen` (for example `gen00s/gen0001/<timestamp-host>/train|val/samples/...`):
- `idx000000.pkl`, `idx000001.pkl`, ...
- `time_stamps.txt`
- `args.txt`, `args.pkl`
- `sample_images/` (sample visualizations)

## 4. Important preflight checks

- The following config keys must be set before running `src/dgms`, or runs often fail:
  - `experiment.path.data_root` (data loading)
  - `experiment.path.save_root_gen` (save path)
  - `experiment.path.dgm_yaml_loc`, `experiment.path.gen_yaml_loc` (YAML paths)
  - `generatives` common section (minimum training hyperparameters)
- DGM-specific keys in `dgm.yaml` are also required for stable runs
  (especially VAE/DDPM loss and reconstruction-related keys).
