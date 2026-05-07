# SimECG-N ECG synthesis guide

This directory generates 1D ECG samples using `gen_ecg.py` and YAML settings.
By default, seed `7` and `syn_mode='pt'` are used to create PT-style data.

## 1. Prerequisites

- Required libraries:
  - `numpy`
  - `PyYAML`
  - `scipy`
  - `tqdm`
- Run from `src/prep/SimECG-N`.
  - `gen_ecg.py` loads `cfg_file = "../../../config.yaml"` via relative path.
- `config.yaml` used by default is the repository root file.

## 2. Configuration values affecting synthesis

The following keys in `config.yaml` are especially important:

- `settings.common.duration`
  - duration in seconds for one generated sample
  - default: `10`
- `settings.common.target_freq`
  - sampling frequency
  - default: `500` Hz
  - generated waveform length is `duration * target_freq` (default `10 * 500 = 5000`)
- `settings.common.val_size`
  - train/val split ratio (for example, `0.1` means 10% validation)
- `settings.common.max_process_time`
  - maximum processing time allowed per sample (over timeouts are dropped)
- `settings.common.n_syn.<clf|pt>`
  - number of samples per `syn_mode`
- `settings.syn_ecg.syncfg`
  - scenario setting key (for example `syn_ecg-01`)
- `settings.common.syncfg_root` and `settings.syn_ecg.syncfg`
  - YAML is expected under `src/prep/SimECG-N/resources/<syncfg>.yaml`
- `settings.common.save_root`
  - output root
  - saved output goes to `{save_root}/{syncfg}`

Example:
if `save_root=/path/to/dataset` and `syncfg=syn_ecg-01`, output is written to
`/path/to/dataset/syn_ecg-01/`.

## 3. Running synthesis

```bash
cd src/prep/SimECG-N
bash syn_data.sh
```

or

```bash
python gen_ecg.py
```

The `__main__` block defaults are:

- `seed = 7`
- `syn_mode = "pt"`

If you need different defaults, edit the bottom of `gen_ecg.py` directly.

## 4. Generated files

After running, files are created under `save_loc` (`{save_root}/{syncfg}`):

- `train_seed0007.pkl`
- `val_seed0007.pkl`
- `cfg_seed0007.txt`

If you change the seed, suffixes in file names change accordingly.

## 5. Process flow (implementation-aligned)

### 5.1 End-to-end flow in `gen_ecg.py`

1. Initialize `ECGsynthesizer(seed, syn_mode)`
   - set NumPy seed
   - load `duration`, `target_freq`, `val_size` from config
   - create `save_loc`
   - initialize `Augment`
2. `make_dataset()`
   - call `generate_ecg()` `n_syn[syn_mode]` times
   - require each sample to finish within `max_process_time` (timeout samples are skipped)
   - split generated data into train/val and save files

### 5.2 Parameter generation

#### `set_base_param()`

Creates base per-feature values from `params` in `resources/*.yaml`.

- start from `base.val`
- if `base.shift` exists, apply an offset from another key
- add random perturbation via `base_perturb` (`normal` or `uniform`)

#### `perturb_param(base_params)`

Adds `beat_perturb` for each heartbeat to introduce beat-level randomness.

### 5.3 Waveform synthesis

#### `generate_beat(start_val, beat_params)`

- build a 1-second axis `t` using `self.fs`
- generate Gaussian-shaped waves using `generate_peak_wave`:
  - P-wave: `p_peak`
  - QRS: `q_peak`, `r_peak`, `s_peak` (`q` and `s` are inverted)
  - T-wave: `t_peak`
- add two noise streams (`wn1`, `wn2`)
- random baseline trend compensation with `base_shift`
- adjust real-time length by `beat_params.length` through `change_sample`

#### `generate_ecg()`

- concatenate beats until generated length exceeds `1.5 * target_length`
- randomly crop a segment of exactly `target_length`
- sample augmentation count from `Poisson(lam=2)` and apply `Augment.rand_augment()` at least 0 times

### 5.4 Augmentation (`augment.py`)

`rand_augment` randomly selects one of:

- global linear scale
- global sinusoidal noise
- global rectangular noise
- global white noise
- local sinusoidal noise
- local rectangular noise
- local white noise

Each augmentation samples amplitude, frequency, and width from SciPy/NumPy random helpers.

## 6. Troubleshooting

- `FileNotFoundError: '../../../config.yaml'`
  - likely running from the wrong working directory; execute under `src/prep/SimECG-N`.
- fewer outputs than expected by `n_syn`
  - timeouts may drop samples
  - increase `settings.common.max_process_time` or optimize `generate_ecg()`.
- output directory not created
  - check whether `settings.common.save_root` exists and is writable.
- `cfg_seed0007.txt` is missing
  - likely because you changed seed in `__main__`; file name is seed-dependent.
