# SimECG-M ECG synthesis guide

This directory includes scripts that run the ECG simulator `ecgsyn` to produce `.dat` files
and convert them to pickles for training.

## 1. Prerequisites

- Required Python packages:
  - `numpy`
  - `tqdm`
  - `scikit-learn`
- Place `linux/ecgsyn` (from PhysioNet) under `src/prep/SimECG-M/linux` and make it executable.
  - Source: `https://www.physionet.org/content/ecgsyn/1.0.0/C/#files-panel`
- `linux/ecgsyn` is a Linux ELF binary.
  - Run on Linux (or WSL2 / Linux VM on macOS).
- `gen_sample.py` assumes it is run from `src/prep/SimECG-M/linux` because it executes `./ecgsyn`.

## 2. Generating `.dat` samples

1. Open `src/prep/SimECG-M/linux/gen_sample.py` and edit values in `if __name__ == "__main__":` if needed.
   - `num_simulations`: total number of outputs (default `1200000`)
   - `n_proc`: number of multiprocessing workers (default `60`)
   - `param_ver`: parameter preset (`v01` / `v02`)
   - `skip`: starting index for resume (default `717395`)
2. Edit `save_root` and configuration block as needed.

Example:
```bash
cd src/prep/SimECG-M/linux
python gen_sample.py
```

### 2.1 Implementation notes for `gen_sample.py`

- `cfg` defines two variants: `v01`, `v02`.
- For each simulation, random values are sampled and converted into command arguments such as:
  - `-n`, `-s`, `-S`
  - `-h`, `-H`, `-a`, `-v`, `-V`, `-f`, `-F`, `-q`, `-R`
- For each sample, it runs `./ecgsyn ... -O synXXXXXXXX.dat` and stores files in:
  - `"{save_root}/simulator/ecgsyn/{timestamp}/dat_files/id####/id########/syn########.dat"`
- Execution is parallelized with `multiprocessing.Pool`.
- Setting `skip` resumes from `range(skip, num_simulations)`.

## 3. Converting to pickle

`src/prep/SimECG-M/convert_to_pickle.py` converts `.dat` files to pickles.

1. Set `root_dir` to the directory containing `.dat` files.
   - Update the `root_dir` value passed to `prepare_pickle(root_dir, target_length)`.
2. Set `target_length` (default `5000`, i.e., 10 seconds at 500 Hz).
3. Run conversion:

```bash
cd src/prep/SimECG-M
python convert_to_pickle.py
```

### 3.1 Implementation notes for `convert_to_pickle.py`

- Read `.dat` files from:
  - `root_dir/dat_files/id*/*/*.dat`
  - read only the 2nd column: `np.loadtxt(dat_file, usecols=(1,))`
  - truncate to `target_length` from the start if longer
- Train/validation split:
  - `train_test_split(..., test_size=0.1, random_state=42)` (90/10 split)
- Outputs:
  - `train`: `root_dir/pickle_files/train/samples/idxXXXXXX.pkl`
  - `val`: `root_dir/pickle_files/val/samples/idxXXXXXX.pkl`

## 4. Notes

- Both scripts include hardcoded paths inside `__main__`; edit them to match your environment before running.
- `run_simulator` checks `subprocess.run` return code and prints stderr on failure.
- Use `skip` for resumable generation and avoid overwriting existing files under `dat_files`.
