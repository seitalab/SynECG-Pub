import os
import pickle
import shutil
import socket
import sys
import time
import traceback
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)


def _expand_path(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def _resolve_path(path: str, base_dir: str) -> str:
    path = _expand_path(path)
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base_dir, path))


def _to_int(value, key: str) -> int:
    if isinstance(value, str):
        v = value.strip()
        if "*" in v:
            left, right = v.split("*")
            value = float(left) * float(right)
        elif "." in v or "e" in v.lower():
            value = float(v)
        else:
            value = int(v)
    if value is None:
        raise ValueError(f"{key} must not be None.")
    return int(value)


def _to_bool(value, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y", "on"):
            return True
        if v in ("false", "0", "no", "n", "off"):
            return False
        raise ValueError(f"{key} must be boolean-like string. Got: {value}")
    return bool(value)


def _setup_torch_runtime() -> None:
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.suppress_errors = True


def _set_seed(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_gpu_ids(gpu_arg: str) -> List[int]:
    gpu_arg = gpu_arg.strip().lower()
    if gpu_arg == "all":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. '--gpu all' cannot be used.")
        gpu_ids = list(range(torch.cuda.device_count()))
        if not gpu_ids:
            raise RuntimeError("No visible CUDA device found for '--gpu all'.")
        return gpu_ids

    raw_ids = [v.strip() for v in gpu_arg.split(",") if v.strip()]
    if not raw_ids:
        raise ValueError("--gpu must be one of: all, -1, 0, 0,1,...")
    gpu_ids = [int(v) for v in raw_ids]

    if len(gpu_ids) == 1 and gpu_ids[0] < 0:
        return []
    if any(gid < 0 for gid in gpu_ids):
        raise ValueError("Negative GPU id can only be used as single value (-1 for CPU).")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --gpu -1 for CPU execution.")

    n_visible = torch.cuda.device_count()
    for gid in gpu_ids:
        if gid >= n_visible:
            raise ValueError(f"Invalid GPU id {gid}. Available ids: 0..{n_visible - 1}")

    # Keep the user-provided order while deduplicating.
    deduped = []
    for gid in gpu_ids:
        if gid not in deduped:
            deduped.append(gid)
    return deduped


def _split_counts(total: int, n_parts: int) -> List[int]:
    base = total // n_parts
    rem = total % n_parts
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def _load_fixed_params(yaml_file: str) -> Dict:
    with open(yaml_file) as f:
        params = yaml.safe_load(f)

    fixed_params = {}
    for key, value in params.items():
        if not isinstance(value, dict):
            continue
        param_type = value.get("param_type")
        if param_type != "fixed":
            raise NotImplementedError(
                f"Only fixed params are supported in generation yaml. key={key}, type={param_type}"
            )
        fixed_params[key] = value.get("param_val")
    return fixed_params


def _get_gen_yaml_path(gen_id: int) -> str:
    path_cfg = config.get("experiment", {}).get("path", {})
    gen_yaml_loc = path_cfg.get("gen_yaml_loc", "./resources/gen_yamls")
    gen_yaml_loc = _expand_path(gen_yaml_loc)
    return os.path.join(
        gen_yaml_loc,
        f"gen{gen_id // 100:02d}s",
        f"gen{gen_id:04d}.yaml",
    )


def _load_labels(labels_path: Optional[str], n_gen: int) -> Optional[np.ndarray]:
    if labels_path is None:
        return None

    ext = os.path.splitext(labels_path)[1].lower()
    if ext == ".npy":
        labels = np.load(labels_path)
    elif ext in (".pkl", ".pickle"):
        with open(labels_path, "rb") as f:
            labels = pickle.load(f)
    else:
        raise ValueError(
            f"Unsupported label file extension: {ext}. Use .npy or .pkl/.pickle."
        )

    labels = np.asarray(labels)
    if labels.ndim not in (1, 2):
        raise ValueError(f"labels ndim must be 1 or 2. Got shape={labels.shape}")
    if labels.shape[0] <= 0:
        raise ValueError(f"labels file is empty: {labels_path}")

    if labels.shape[0] < n_gen:
        n_labels = labels.shape[0]
        n_repeat = int(np.ceil(n_gen / n_labels))
        tile_shape = (n_repeat,) if labels.ndim == 1 else (n_repeat, 1)
        labels = np.tile(labels, tile_shape)
        print(
            f"[warn] labels length ({n_labels}) is smaller than n_gen ({n_gen}). "
            f"Repeated labels {n_repeat} times and truncated to n_gen."
        )

    labels = labels[:n_gen]
    return labels


def _import_sssd_ecg():
    sssd_dir = os.path.abspath("sssd_standalone")
    if sssd_dir not in sys.path:
        sys.path.insert(0, sssd_dir)
    from sssd_standalone import SSSDECG  # pylint: disable=import-outside-toplevel

    return SSSDECG


def _load_model(
    cfg_path: str,
    ckpt_path: str,
    device: str,
    compile_model: bool,
):
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))

    _setup_torch_runtime()
    SSSDECG = _import_sssd_ecg()
    model = SSSDECG(config_path=cfg_path, device=device)
    model.load_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    if compile_model and hasattr(torch, "compile"):
        model.model = torch.compile(model.model, mode="max-autotune", fullgraph=False)
    return model


class PklChunkWriter:

    def __init__(
        self,
        save_dir: str,
        n_per_file: int,
        start_idx: int = 1,
        fail_if_exists: bool = True,
    ):
        if n_per_file <= 0:
            raise ValueError(f"n_per_file must be > 0. Got {n_per_file}")

        self.save_dir = save_dir
        self.n_per_file = n_per_file
        self._next_idx = start_idx
        self.saved_files: List[str] = []
        self.total_saved = 0
        self._buffer: List[np.ndarray] = []
        self._buffer_size = 0

        os.makedirs(self.save_dir, exist_ok=True)
        existing = sorted(glob(os.path.join(self.save_dir, "idx*.pkl")))
        if fail_if_exists and existing:
            raise FileExistsError(
                f"Target directory already has idx*.pkl files: {self.save_dir}"
            )

    def _pop_n(self, n_take: int) -> np.ndarray:
        taken = []
        remaining = []
        rest = n_take

        for arr in self._buffer:
            n_arr = arr.shape[0]
            if rest <= 0:
                remaining.append(arr)
                continue
            if n_arr <= rest:
                taken.append(arr)
                rest -= n_arr
            else:
                taken.append(arr[:rest])
                remaining.append(arr[rest:])
                rest = 0

        if rest != 0:
            raise RuntimeError("Internal buffering error while popping chunk.")

        self._buffer = remaining
        self._buffer_size -= n_take
        return np.concatenate(taken, axis=0)

    def _save_chunk(self, chunk: np.ndarray) -> None:
        save_path = os.path.join(self.save_dir, f"idx{self._next_idx:06d}.pkl")
        if os.path.exists(save_path):
            raise FileExistsError(f"Output file already exists: {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.saved_files.append(save_path)
        self.total_saved += chunk.shape[0]
        self._next_idx += 1

    def add(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        self._buffer.append(samples)
        self._buffer_size += samples.shape[0]

        while self._buffer_size >= self.n_per_file:
            chunk = self._pop_n(self.n_per_file)
            self._save_chunk(chunk)

    def flush(self) -> None:
        if self._buffer_size == 0:
            return
        chunk = self._pop_n(self._buffer_size)
        self._save_chunk(chunk)


def _generate_to_writer(
    model,
    device: str,
    mode: str,
    n_samples: int,
    batch_size: int,
    amp_dtype: str,
    labels: Optional[np.ndarray],
    writer: PklChunkWriter,
) -> float:
    if n_samples <= 0:
        return 0.0

    if device.startswith("cuda"):
        torch.cuda.synchronize(device=device)

    start = time.perf_counter()
    offset = 0
    while offset < n_samples:
        current_bs = min(batch_size, n_samples - offset)

        label_tensor = None
        if labels is not None:
            label_batch = labels[offset:offset + current_bs]
            label_tensor = torch.from_numpy(label_batch).to(device)

        if mode == "normal":
            samples = model.generate(
                labels=label_tensor,
                num_samples=current_bs,
                return_numpy=True,
            )
        elif mode == "amp":
            samples = model.generate_amp(
                labels=label_tensor,
                num_samples=current_bs,
                return_numpy=True,
                use_amp=True,
                amp_dtype=amp_dtype,
            )
        elif mode == "jit":
            samples = model.generate_jit(
                labels=label_tensor,
                num_samples=current_bs,
                return_numpy=True,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        writer.add(np.asarray(samples))
        offset += current_bs

    writer.flush()

    if device.startswith("cuda"):
        torch.cuda.synchronize(device=device)
    return time.perf_counter() - start


def _worker_generate_and_save(
    rank: int,
    gpu_id: int,
    cfg_path: str,
    ckpt_path: str,
    mode: str,
    n_samples: int,
    batch_size: int,
    amp_dtype: str,
    compile_model: bool,
    labels_chunk: Optional[np.ndarray],
    n_per_file: int,
    save_dir: str,
    file_idx_start: int,
    seed: int,
    result_queue,
) -> None:
    try:
        if n_samples <= 0:
            result_queue.put((rank, gpu_id, 0, 0, 0.0, None))
            return

        device = f"cuda:{gpu_id}"
        torch.set_num_threads(1)
        _set_seed(seed + rank, device)
        model = _load_model(
            cfg_path=cfg_path,
            ckpt_path=ckpt_path,
            device=device,
            compile_model=compile_model,
        )
        writer = PklChunkWriter(
            save_dir,
            n_per_file=n_per_file,
            start_idx=file_idx_start,
            fail_if_exists=False,
        )
        elapsed = _generate_to_writer(
            model=model,
            device=device,
            mode=mode,
            n_samples=n_samples,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
            labels=labels_chunk,
            writer=writer,
        )
        result_queue.put((
            rank,
            gpu_id,
            writer.total_saved,
            len(writer.saved_files),
            elapsed,
            None,
        ))
    except Exception:
        result_queue.put((rank, gpu_id, 0, 0, 0.0, traceback.format_exc()))


def _run_split_multi_gpu(
    split_name: str,
    gpu_ids: List[int],
    cfg_path: str,
    ckpt_path: str,
    mode: str,
    n_samples: int,
    batch_size: int,
    amp_dtype: str,
    compile_model: bool,
    labels: Optional[np.ndarray],
    n_per_file: int,
    save_dir: str,
    seed: int,
):
    os.makedirs(save_dir, exist_ok=True)
    existing = sorted(glob(os.path.join(save_dir, "idx*.pkl")))
    if existing:
        raise FileExistsError(f"Target directory already has idx*.pkl files: {save_dir}")

    if n_samples <= 0:
        return 0, 0, 0.0, []

    counts = _split_counts(n_samples, len(gpu_ids))
    work_items = []
    for rank, (gpu_id, count) in enumerate(zip(gpu_ids, counts)):
        if count <= 0:
            continue
        n_files = (count + n_per_file - 1) // n_per_file
        work_items.append({
            "rank": rank,
            "gpu_id": gpu_id,
            "count": count,
            "n_files": n_files,
        })

    if labels is None:
        label_chunks = [None] * len(work_items)
    else:
        split_points = np.cumsum([w["count"] for w in work_items])[:-1]
        label_chunks = np.split(labels, split_points)

    next_idx = 1
    for item in work_items:
        item["file_idx_start"] = next_idx
        next_idx += item["n_files"]

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    wall_start = time.perf_counter()
    for idx, item in enumerate(work_items):
        p = ctx.Process(
            target=_worker_generate_and_save,
            args=(
                item["rank"],
                item["gpu_id"],
                cfg_path,
                ckpt_path,
                mode,
                item["count"],
                batch_size,
                amp_dtype,
                compile_model,
                label_chunks[idx],
                n_per_file,
                save_dir,
                item["file_idx_start"],
                seed,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    results = {}
    errors = []
    for _ in processes:
        rank, gpu_id, n_saved, n_files, elapsed, err = result_queue.get()
        if err is not None:
            errors.append((rank, gpu_id, err))
        else:
            results[rank] = {
                "gpu_id": gpu_id,
                "n_saved": n_saved,
                "n_files": n_files,
                "elapsed": elapsed,
            }

    for p in processes:
        p.join()
        if p.exitcode != 0:
            errors.append((-1, -1, f"worker exited with code {p.exitcode}"))

    if errors:
        print(f"{split_name} generation failed on at least one worker.")
        for rank, gpu_id, err in errors:
            print(f"[worker rank={rank}, gpu={gpu_id}] {err}")
        raise RuntimeError(f"Multi-GPU {split_name} generation failed.")

    wall_elapsed = time.perf_counter() - wall_start
    per_gpu_stats = []
    total_saved = 0
    total_files = 0
    for item in work_items:
        result = results[item["rank"]]
        per_gpu_stats.append({
            "gpu_id": item["gpu_id"],
            "n_saved": result["n_saved"],
            "n_files": result["n_files"],
            "elapsed": result["elapsed"],
        })
        total_saved += result["n_saved"]
        total_files += result["n_files"]

    return total_saved, total_files, wall_elapsed, per_gpu_stats


def main():
    parser = ArgumentParser()
    parser.add_argument("--gen", type=int, required=True)
    parser.add_argument(
        "--gpu",
        type=str,
        required=True,
        help="GPU selector: -1 (cpu), 0, 0,1,2, or all",
    )
    args = parser.parse_args()

    gpu_ids = _parse_gpu_ids(args.gpu)
    use_multi_gpu = len(gpu_ids) > 1
    device = "cpu" if len(gpu_ids) == 0 else f"cuda:{gpu_ids[0]}"

    gen_yaml_path = _get_gen_yaml_path(args.gen)
    if not os.path.exists(gen_yaml_path):
        raise FileNotFoundError(f"Generation yaml not found: {gen_yaml_path}")

    fixed = _load_fixed_params(gen_yaml_path)
    run_base_dir = os.getcwd()

    required_keys = ["ckpt_path", "output_root", "n_gen_train", "n_gen_val"]
    for key in required_keys:
        if key not in fixed or fixed[key] is None:
            raise ValueError(f"Missing required key in generation yaml: {key}")

    cfg_path = fixed.get("cfg_path", "./sssd_standalone/config/config_SSSD_ECG.json")
    cfg_path = _resolve_path(cfg_path, run_base_dir)
    ckpt_path = _resolve_path(fixed["ckpt_path"], run_base_dir)

    output_root = _resolve_path(fixed["output_root"], run_base_dir)
    append_timestamp = _to_bool(fixed.get("append_timestamp", True), "append_timestamp")
    if append_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_root = os.path.join(output_root, f"{timestamp}-{socket.gethostname()}")
    else:
        run_root = output_root

    train_samples_dir = os.path.join(run_root, "train", "samples")
    val_samples_dir = os.path.join(run_root, "val", "samples")

    n_gen_train = _to_int(fixed["n_gen_train"], "n_gen_train")
    n_gen_val = _to_int(fixed["n_gen_val"], "n_gen_val")
    n_per_file = _to_int(fixed.get("n_per_file", 50000), "n_per_file")
    batch_size = _to_int(fixed.get("batch_size", 256), "batch_size")
    mode = fixed.get("mode", "amp")
    amp_dtype = fixed.get("amp_dtype", "fp16")
    compile_model = _to_bool(fixed.get("compile", False), "compile")
    seed_default = config.get("experiment", {}).get("seed", {}).get("generate", 8)
    seed = _to_int(fixed.get("seed", seed_default), "seed")

    if mode not in ("normal", "amp", "jit"):
        raise ValueError(f"mode must be normal/amp/jit. Got: {mode}")
    if amp_dtype not in ("bf16", "fp16"):
        raise ValueError(f"amp_dtype must be bf16/fp16. Got: {amp_dtype}")

    train_labels_path = fixed.get("labels_train_path")
    val_labels_path = fixed.get("labels_val_path")
    if train_labels_path is not None:
        train_labels_path = _resolve_path(train_labels_path, run_base_dir)
    if val_labels_path is not None:
        val_labels_path = _resolve_path(val_labels_path, run_base_dir)

    train_labels = _load_labels(train_labels_path, n_gen_train)
    val_labels = _load_labels(val_labels_path, n_gen_val)

    os.makedirs(run_root, exist_ok=True)
    shutil.copy2(gen_yaml_path, os.path.join(run_root, "gen_config.yaml"))

    print(f"Using generation yaml: {gen_yaml_path}")
    if use_multi_gpu:
        print(f"Devices: {','.join([f'cuda:{gid}' for gid in gpu_ids])} (multi-gpu)")
    else:
        print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output root: {run_root}")
    print(f"Train samples: {n_gen_train}, Val samples: {n_gen_val}")

    t0 = time.perf_counter()
    if use_multi_gpu:
        train_saved, train_n_files, train_elapsed, train_per_gpu = _run_split_multi_gpu(
            split_name="train",
            gpu_ids=gpu_ids,
            cfg_path=cfg_path,
            ckpt_path=ckpt_path,
            mode=mode,
            n_samples=n_gen_train,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
            compile_model=compile_model,
            labels=train_labels,
            n_per_file=n_per_file,
            save_dir=train_samples_dir,
            seed=seed,
        )
        val_saved, val_n_files, val_elapsed, val_per_gpu = _run_split_multi_gpu(
            split_name="val",
            gpu_ids=gpu_ids,
            cfg_path=cfg_path,
            ckpt_path=ckpt_path,
            mode=mode,
            n_samples=n_gen_val,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
            compile_model=compile_model,
            labels=val_labels,
            n_per_file=n_per_file,
            save_dir=val_samples_dir,
            seed=seed + 1,
        )
    else:
        _set_seed(seed, device)
        model = _load_model(
            cfg_path=cfg_path,
            ckpt_path=ckpt_path,
            device=device,
            compile_model=compile_model,
        )

        train_writer = PklChunkWriter(train_samples_dir, n_per_file=n_per_file)
        train_elapsed = _generate_to_writer(
            model=model,
            device=device,
            mode=mode,
            n_samples=n_gen_train,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
            labels=train_labels,
            writer=train_writer,
        )
        train_saved = train_writer.total_saved
        train_n_files = len(train_writer.saved_files)
        train_per_gpu = [{
            "gpu_id": device,
            "n_saved": train_saved,
            "n_files": train_n_files,
            "elapsed": train_elapsed,
        }]

        _set_seed(seed + 1, device)
        val_writer = PklChunkWriter(val_samples_dir, n_per_file=n_per_file)
        val_elapsed = _generate_to_writer(
            model=model,
            device=device,
            mode=mode,
            n_samples=n_gen_val,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
            labels=val_labels,
            writer=val_writer,
        )
        val_saved = val_writer.total_saved
        val_n_files = len(val_writer.saved_files)
        val_per_gpu = [{
            "gpu_id": device,
            "n_saved": val_saved,
            "n_files": val_n_files,
            "elapsed": val_elapsed,
        }]
    total_elapsed = time.perf_counter() - t0

    if train_saved != n_gen_train:
        raise RuntimeError(
            f"train sample size mismatch: requested={n_gen_train}, saved={train_saved}"
        )
    if val_saved != n_gen_val:
        raise RuntimeError(
            f"val sample size mismatch: requested={n_gen_val}, saved={val_saved}"
        )

    summary = {
        "gen_id": args.gen,
        "gpu_arg": args.gpu,
        "gpu_ids": gpu_ids,
        "device": device,
        "cfg_path": cfg_path,
        "ckpt_path": ckpt_path,
        "mode": mode,
        "amp_dtype": amp_dtype,
        "compile": compile_model,
        "batch_size": batch_size,
        "n_per_file": n_per_file,
        "seed": seed,
        "run_root": run_root,
        "train": {
            "n_requested": n_gen_train,
            "n_saved": train_saved,
            "n_files": train_n_files,
            "elapsed_sec": train_elapsed,
            "per_gpu": train_per_gpu,
        },
        "val": {
            "n_requested": n_gen_val,
            "n_saved": val_saved,
            "n_files": val_n_files,
            "elapsed_sec": val_elapsed,
            "per_gpu": val_per_gpu,
        },
        "wall_time_sec": total_elapsed,
    }
    with open(os.path.join(run_root, "generation_summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    train_spd = n_gen_train / train_elapsed if train_elapsed > 0 else 0.0
    val_spd = n_gen_val / val_elapsed if val_elapsed > 0 else 0.0
    wall_spd = (n_gen_train + n_gen_val) / total_elapsed if total_elapsed > 0 else 0.0
    print(f"Train: {n_gen_train} samples, {train_n_files} files, {train_spd:.2f} samples/sec")
    for stat in train_per_gpu:
        spd = stat["n_saved"] / stat["elapsed"] if stat["elapsed"] > 0 else 0.0
        print(
            f"  train gpu {stat['gpu_id']}: "
            f"{stat['n_saved']} samples in {stat['elapsed']:.3f} sec ({spd:.2f} samples/sec)"
        )
    print(f"Val  : {n_gen_val} samples, {val_n_files} files, {val_spd:.2f} samples/sec")
    for stat in val_per_gpu:
        spd = stat["n_saved"] / stat["elapsed"] if stat["elapsed"] > 0 else 0.0
        print(
            f"  val   gpu {stat['gpu_id']}: "
            f"{stat['n_saved']} samples in {stat['elapsed']:.3f} sec ({spd:.2f} samples/sec)"
        )
    print(f"Wall : {total_elapsed:.3f} sec ({wall_spd:.2f} samples/sec)")
    print(f"Summary written to: {os.path.join(run_root, 'generation_summary.yaml')}")


if __name__ == "__main__":
    main()
