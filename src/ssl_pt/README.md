# SSL pretraining guide (`src/ssl_pt`)

This README summarizes how to run pretraining (SSL pretraining) under `src/ssl_pt`.

## 1. Running pretraining

- Entry script: `src/ssl_pt/pretrain.py`
- Use `--pt` to specify pretraining ID
- Use `--device` to set device (for example, `cuda:0` or `cuda:1`)
- Add `--debug` to run a short debug config (`epochs=2`, `eval_every=1`)

```bash
cd src/ssl_pt
python pretrain.py --pt 1 --device cuda:0
python pretrain.py --pt 1 --device cpu --debug
```

> Important: `pretrain.py` loads `../../config.yaml`, so run it from `src/ssl_pt`.

## 2. YAML config flow

Pretraining reads YAML files associated with a given `pt_id`.
These files are stored under the path defined by `experiment.path.pretrain_yaml_loc` in `config.yaml`.

- `./resources/pretrain_yamls/ptXXs/pt000Y.yaml` (for example: `pt_id=1` -> `./resources/pretrain_yamls/pt00s/pt0001.yaml`)

`src/ssl_pt/experiment.py` parses YAML via `ExperimentManagerBase._load_train_params` as follows:

- `param_type: fixed` -> use a fixed value
- `param_type: grid` -> intended for grid search (usually unused in pretraining)
- `param_type: hps` -> intended for HPS sweeps (usually unused in pretraining)

Minimal example:

```yaml
# General
architecture:
  param_type: fixed
  param_val: transformer
exp_setting_key:
  param_type: fixed
  param_val: pt_ptbxl01
ssl:
  param_type: fixed
  param_val: mae
pretrained_weight_key:
  param_type: fixed
  param_val: null
```

Common keys you may add:
- `dataset`, `data_lim`, `val_lim`
- `batch_size`, `learning_rate`, `total_samples`, `eval_every`, `save_model_every`
- `target_freq`, `max_duration`, `n_workers`, `num_lead`, `batch_size`, `optimizer`, `mask_ratio`, `max_shift_ratio`

Numeric values may be strings like `1.*1e6`; `ExperimentManagerBase._str_to_number()` converts `"A*B"` expression strings into numbers.

## 3. Templates and YAML generation

`src/ssl_pt/resources` includes YAML templates and generator scripts.
Run generator scripts from `src/ssl_pt/resources` because `gen_*` scripts depend on the current working directory.

```bash
cd src/ssl_pt/resources
python gen_yamls_r01.py
python gen_yamls_r04.py
```

By default, generated files are written to `./pretrain_yamls` in the same directory.

Main generators:
- `gen_yamls_r01.py`: expands `template001.yaml` for `pt_ptbxl01` and `pt_syn01` across `mae,dino,simclr,ibot,byol`
- `gen_yamls_r02.py` / `r03.py`: adds `pt_gan01`, `pt_vae01`, `pt_sim01` only for `mae`
- `gen_yamls_r04.py` to `r09.py`: scans `template002.yaml` variants with fine-grained dataset/data limit settings
- `gen_yamls_r10.py`: adds extra cases with `architecture=transformer` based on `template007.yaml`

Template overview:
- `template001.yaml`: fixed `architecture=transformer`; dataset defined by shared settings (`pt_ptbxl01`/`pt_syn01`)
- `template002.yaml`: explicit `dataset` and `data_lim`
- `template003.yaml` to `template007.yaml`: MAE-related RNN/MEGA/LUNA/CNN parameters

## 4. Main pretraining parameters

Common values are injected from `config.yaml -> ssl.pretrain`.

- `ssl.pretrain.common.base`
  - `target_freq`, `max_duration`, `batch_size`, `optimizer`, `aug_mask_ratio`, `max_shift_ratio`, `n_workers`, `backbone_out_dim`, `emb_dim`, and related settings
- Scenario keys: `ssl.pretrain.pt_ptbxl01`, `ssl.pretrain.pt_syn01`, `ssl.pretrain.pt_syn02`, ...
  - `pt_id` determines which `exp_setting_key` is selected
- Shared SSL model blocks:
  - e.g., `ssl.pretrain.mae.all_arch`

## 5. How pretraining runs

1. Create YAML (`pt0001`, `pt0002`, ...).
2. Run `python pretrain.py --pt <id>`.
3. `run_train(..., run_pretraining=True)` is called:
   - initializes `ModelPretrainer`
   - builds dataloaders via `prepare_dataloader(datatype=train/val, is_finetune=False)`
   - iterates `train_loader` by `run` until `args.total_samples` is reached
   - evaluates at `eval_every` intervals
4. Saves outputs and exits.

## 6. Saving location and outputs

Default save root is `config.yaml -> experiment.path.save_root`:

`{save_root}/pt00s/pt0001/{timestamp}/`

Typical outputs:
- `exp_config.yaml` (resolved config used for this run)
- `params.pkl` (run parameters)
- `net.pth` (best model checkpoint)
- `best_score.txt`
- `train_scores.json`, `eval_scores.json`
- `interims/net_*.pth` (checkpoint snapshots when `save_model_every` triggers)

## 7. Data assumptions

Based on `src/ssl_pt/codes/data/dataset.py`:

- pickles are expected under `experiment.path.data_root` in `config.yaml`
  - examples: `train_seed0001.pkl`, `val_seed0001.pkl`, `test.pkl`
- if `dataset` follows `gen-*`, paths are resolved under `generatives.data` in `config.yaml`

## 8. Notes and caveats

- Pretraining typically assumes `pretrained_weight_key` is `null`.
  If it is not `null`, `ExperimentManagerBase._get_pt_dir()` expects `./resources/settings.yaml`.
  This file is currently not included in this repository, so pretrained-weight reuse needs to be prepared manually.
- `run_train.py` runs `ModelPretrainer` with `run_pretraining=True`.
  The `freeze` argument is not implemented in the called path, but this does not affect standard pretraining-only runs.

## 9. Useful config checks

```bash
# Verify where YAML and outputs are expected
cd src/ssl_pt
python - <<'PY'
import yaml
with open('../../config.yaml') as f:
    c = yaml.safe_load(f)
print(c["experiment"]["path"]["pretrain_yaml_loc"])
print(c["experiment"]["path"]["save_root"])
print(c["experiment"]["seed"]["pretrain"])
PY

# Verify target YAML file exists
ls resources/pretrain_yamls/pt00s/pt0001.yaml
```

## 10. Check workflow (`generate -> run -> inspect logs`)

`src/ssl_pt/scripts/check_pretrain.sh` runs YAML generation, pretraining, and log capture in one command.

```bash
cd src/ssl_pt
bash scripts/check_pretrain.sh --pt 1 --gen-script gen_yamls_r01.py --device cuda:0 --debug
```

You can set `--log-dir` (for example, `--log-dir ./tmp_logs`), and use `bash scripts/check_pretrain.sh --help` for full options.
