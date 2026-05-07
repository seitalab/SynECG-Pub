# SSL classifier pretraining (transfer with SSL weights) guide (`src.ssl_clf`)

This directory runs ECG classifier experiments.
In this project, "pretraining" means:

1) SSL core pretraining by `src/ssl_pt/pretrain.py`  
2) SSL-weight transfer and classifier pretraining via `src/ssl_clf`

This document covers step 2.

## 1. Preparation

`src/ssl_clf` loads `../../config.yaml`; run from `src/ssl_clf`.

```bash
cd src/ssl_clf
```

Minimum required checks in `config.yaml`:

- `experiment.path.data_root` (pickle path for classifier data)
- `experiment.path.save_root` (experiment output root)
- `ssl.eval_pt_model.params.main01` (default training settings)
- `ssl.eval_pt_model.dataset_comb` (dataset / target label mapping)
- `ssl.eval_pt_model.fixed_setting.dataset_comb` (same intent, fixed setting variant)

## 2. Meaning of `finetune_target`

`finetune_target` in templates / generation YAML (`src/ssl_clf/resources`) is used when initializing classifier backbone.

Resolution rules in `prepare_model.py`:

- `ptXXXX` -> search latest checkpoint under `ptXXs/ptXXXX/<timestamp>` in `experiment.path.save_root`
- `pt-extra01` -> use `ssl.eval_pt_model.fixed_setting.extra_pt_model`
- `progress-ptXXXX-<step>` -> use progress checkpoint
- `random-init` -> no SSL weights; initialize with architecture metadata from `pt0006` (baseline)
- Variants like `gru` / `resnet` -> parse pt index and apply same resolution rules

Whether to fix or fine-tune the pretraining backbone is controlled by `freeze`:

- `freeze: False` -> fine-tune backbone and head
- `freeze: True` -> freeze backbone and update head only

## 3. Experiment YAML flow

Experiment templates are in `src/ssl_clf/resources/templates/*.yaml`.
Use `resources/README.md` and `memo.md` as references for existing experiment IDs (e.g., `exp01` etc.).

Run from `src/ssl_clf/resources`:

```bash
cd src/ssl_clf/resources
python exp_builder.py --all

# or build one set:
python exp_builder.py exp01_hps
```

Legacy runners such as `exp01_hps.py` are moved to `resources/legacy_exp_generators/`.

Validation YAMLs mainly replace:

- `VAL01`: `finetune_target` (example `pt0006`)
- `VAL02`: `dataset` (`ptbxl` / `g12ec` / `cpsc`)
- `VAL03`: `target_dx` (example `af`, `pvc`)
- `VAL04`: `hps_result` / `data_lim`
- `VAL05`: `result_path` (if needed)

## 4. Execution commands

- Run multiple IDs in batch:

```bash
cd src/ssl_clf
python bulk_execute.py --ids 1-20 --device cuda:0
```

- Run one ID:

```bash
cd src/ssl_clf
python bulk_execute.py --ids 1 --device cuda:0
```

`--show_error` prints full traceback for failures.

## 5. Output

`ExperimentManager._prepare_save_loc` stores results under `config.yaml -> save_root`:

- `ssl-clf-exp00s/exp0001/yyMMdd-HHmmss/<hostname>/`
- `params.pkl`, `params.txt`, `train_scores.json`, `eval_scores.json`, `net.pth`, `best_score.txt`
- `multirun/train/seed0001/...` and `multirun/eval/seed0001/...`
- `ResultTableMultiSeed.csv`, `ResultTableHPS.csv`, `exp_config_src.yaml`

## 6. About SSL pretraining execution (important)

`src/ssl_clf` does not include the SSL pretraining trainer.
If you need to create a new SSL pretraining run, use `src/ssl_pt/pretrain.py`.

```bash
cd src/ssl_pt
python pretrain.py --pt 1 --device cuda:0
python pretrain.py --pt 1 --device cpu --debug
```

After creating checkpoint files, pass the resulting ID as `finetune_target`.

## 7. Files to check before running classifier pretraining

- `src/ssl_clf/bulk_execute.py` (batch entry)
- `src/ssl_clf/experiment.py` (HPS/GS/seed main flow)
- `src/ssl_clf/codes/run_train.py` (weight loading when `finetune_target` exists)
- `src/ssl_clf/codes/models/prepare_model.py` (resolution of `ptXXXX`, `progress-pt`, `pt-extra`, `random-init`)
- `src/ssl_clf/codes/supports/set_weight.py` (`net.pth` resolution helper)
- `src/ssl_clf/codes/data/dataloader.py` and `dataset.py` (dataset, `data_lim`, `val_lim`, transforms)
- `src/ssl_clf/resources/README.md` (existing experiment and dataset mapping)
- `src/ssl_clf/memo.md` (past run history)

## 8. Troubleshooting

- `check_done.py` can quickly verify result completeness (`target_file` can change how completion is evaluated).
- If path resolution fails, first inspect `experiment.path` and `ssl.eval_pt_model.fixed_setting.extra_pt_model` in `config.yaml`.
- `finetune_target` not found often raises `ValueError`.
  Confirm `pt` string format and validity (for example `pt0006`).
