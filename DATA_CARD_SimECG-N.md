# Data Card: SimECG-N

## Dataset summary

SimECG-N is a synthetic ECG waveform corpus generated in this repository using a parametric simulator (`src/prep/SimECG-N/gen_ecg.py`) and augmentation utilities (`src/prep/SimECG-N/augment.py`).
The repository introduces this corpus as a single new dataset asset for NeurIPS 2026 Evaluations & Datasets.

## Intended use

- Pretraining and benchmarking single-lead ECG models.
- Non-clinical method comparison for representation learning and synthetic data baselines.
- Reproducibility studies and ablation work on training protocols.

## Non-recommended use

- Clinical diagnosis, triage, or direct deployment in care settings.
- Re-identification, identity attribution, or any attempt to link synthetic records to any source patient.
- Replacing real data collection for final clinical validation without independent human/clinical review.

## Dataset composition

### Subsets

The default release defines train/validation subsets only:

- `train`: subset in `train_seed0007.pkl`
- `val`: subset in `val_seed0007.pkl`

Both subsets are single-lead, fixed-length numeric sequences generated from:

- sampling rate: 500 Hz
- duration: 10 s
- split ratio: 10% validation (`settings.common.val_size = 0.1`)

### Generation settings captured for provenance

- script: `src/prep/SimECG-N/gen_ecg.py`
- simulator params: YAML under `src/prep/SimECG-N/resources/`
- default config root: repository `config.yaml` (`settings` section):
  - `settings.common.save_root = ../../../raw_data`
  - `settings.common.syncfg_root = resources`
  - `settings.syn_ecg.syncfg = syn_ecg-04`
  - `settings.common.n_syn.pt = 1_000`
  - `settings.common.duration = 10`
  - `settings.common.target_freq = 500`
  - `settings.common.val_size = 0.1`
  - `settings.common.max_process_time = 5`
  - `settings.common.seed` is not configured under `settings.common`; the generator default is `seed = 7` in `gen_ecg.py`.
- generation mode used by default in `gen_ecg.py`: `syn_mode = "pt"`

### Expected file list and manifest

- `syn_ecg-04/train_seed0007.pkl` (`train` split)
- `syn_ecg-04/val_seed0007.pkl` (`val` split)
- `syn_ecg-04/cfg_seed0007.txt` (generator snapshot)

Row-wise file-level metadata is maintained in `data/manifest_simecg_n.csv`.

## Limitations

- Synthetic data are generated from a parametric model and do not originate from individual patients in this repository.
- The generated morphology variety depends on seeded stochastic perturbations and may not cover all clinical edge cases.
- Timeout rejection in generation can reduce the realized sample count relative to target `n_syn`.

## Biases

- Default configuration may favor dominant rhythm-like patterns and underrepresent rare pathological rhythms.
- Signal appearance can vary strongly with seed and perturbation draws; model results can be sensitive to this random variation.
- No demographic attributes are attached, so subgroup-specific bias audits are limited by design.

## Sensitive information

- No patient identifiers or labels are present in these synthetic files.
- No protected health information is embedded in the repository-provided synthetic outputs.
- If you choose to add derived metadata, ensure it does not introduce linkable attributes.

## Social impact

Positive:
- Enables model development without redistributing real ECG recordings.
- Supports sharing methods under stronger privacy and licensing constraints.

Risks:
- If used as-is in healthcare settings, synthetic-only training may reduce clinical reliability.
- Over-trusting generated morphologies can produce inflated benchmark performance not mirrored in clinical streams.

## Provenance

- Corpus creator: this repository’s SimECG-N generation module.
- Raw source: publicly available real ECG datasets are not packaged or redistributed from this repository for this asset.
- File-level manifest: `data/manifest_simecg_n.csv`
- Dataset metadata (Croissant): `metadata/simecg_n_croissant.json`

## Licensing

- Code: see `LICENSE`
- SimECG-N corpus data: see `LICENSE_DATA`

## Reproduction

```bash
cd src/prep/SimECG-N
python gen_ecg.py
```

or:

```bash
bash scripts/prepare_simecg-n.sh
```

The outputs are placed under the directory configured by `settings.common.save_root`, with `syncfg` subfolder.

To customize generation, edit:
- `src/prep/SimECG-N/gen_ecg.py` (`seed`, `syn_mode`) or
- `src/prep/SimECG-N/resources/syn_ecg-04.yaml` (and other simulator config files).

## TODOs

- Set final, public version tags and release identifiers.
- Fill actual record counts in `data/manifest_simecg_n.csv` after full generation and validation.
- Finalize the public data license string in `metadata/simecg_n_croissant.json` and `LICENSE_DATA`.
- Finalize the final code license text in `LICENSE` if project policy requires a specific SPDX file.
