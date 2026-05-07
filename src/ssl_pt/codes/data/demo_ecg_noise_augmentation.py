from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SSL_PT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if SSL_PT_ROOT not in sys.path:
    sys.path.insert(0, SSL_PT_ROOT)

from codes.data.ecg_noise_augmentation import ECGNoiseAugmentation


def generate_dummy_ecg(
    sample_rate: int = 500,
    duration_sec: float = 10.0,
    heart_rate_bpm: float = 70.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate a simple clean ECG-like waveform with repeated P-QRS-T complexes."""
    rng = np.random.default_rng(seed)
    n_samples = int(sample_rate * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    ecg = np.zeros_like(t)

    rr = 60.0 / heart_rate_bpm
    beat_times = np.arange(0.5, duration_sec, rr)
    for bt in beat_times:
        ecg += 0.10 * np.exp(-0.5 * ((t - (bt - 0.20)) / 0.025) ** 2)  # P
        ecg += -0.12 * np.exp(-0.5 * ((t - (bt - 0.04)) / 0.010) ** 2)  # Q
        ecg += 1.20 * np.exp(-0.5 * ((t - bt) / 0.008) ** 2)  # R
        ecg += -0.25 * np.exp(-0.5 * ((t - (bt + 0.04)) / 0.012) ** 2)  # S
        ecg += 0.30 * np.exp(-0.5 * ((t - (bt + 0.26)) / 0.055) ** 2)  # T

    ecg += 0.015 * np.sin(2.0 * np.pi * 0.33 * t)  # gentle baseline rhythm
    ecg += rng.normal(0.0, 0.005, size=n_samples)  # small sensor noise
    return ecg.astype(np.float32)


def compute_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    signal_power = float(np.mean(clean.astype(np.float64) ** 2))
    noise_power = float(np.mean((noisy.astype(np.float64) - clean.astype(np.float64)) ** 2))
    if noise_power <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(max(signal_power, 1e-12) / noise_power))


def summarize(clean: np.ndarray, augmented: np.ndarray) -> Dict[str, float]:
    return {
        "mean_before": float(np.mean(clean)),
        "std_before": float(np.std(clean)),
        "mean_after": float(np.mean(augmented)),
        "std_after": float(np.std(augmented)),
        "snr_db": compute_snr_db(clean, augmented),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo ECG noise augmentation.")
    parser.add_argument("--sample_rate", type=int, default=500)
    parser.add_argument("--duration_sec", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save_plot", action="store_true")
    parser.add_argument("--plot_path", type=str, default="./ecg_noise_demo.png")
    args = parser.parse_args()

    clean = generate_dummy_ecg(
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        seed=args.seed,
    )
    modes = [
        "baseline_wander",
        "emg",
        "motion_artifact",
        "electrode_displacement",
        "combined",
    ]

    print("ECG Noise Augmentation Demo")
    print(f"Signal length: {clean.shape[0]} samples")
    print(f"Sample rate : {args.sample_rate} Hz")

    outputs = {}
    for i, mode in enumerate(modes):
        augmenter = ECGNoiseAugmentation(
            sample_rate=args.sample_rate,
            mode=mode,
            seed=args.seed + i,
        )
        # Force application in the demo so each mode is visibly tested.
        if mode == "combined":
            for key in augmenter.probabilities:
                augmenter.probabilities[key] = 1.0
        else:
            augmenter.probabilities[mode] = 1.0

        augmented = augmenter(clean)
        outputs[mode] = augmented
        stats = summarize(clean, augmented)
        cfg = augmenter.get_config()
        print(
            f"[{mode}] "
            f"mean: {stats['mean_before']:.5f} -> {stats['mean_after']:.5f}, "
            f"std: {stats['std_before']:.5f} -> {stats['std_after']:.5f}, "
            f"SNR: {stats['snr_db']:.2f} dB, "
            f"applied: {cfg['last_call'].get('applied_order', [])}"
        )

    if args.save_plot:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib is not available; skipping plot.")
            return

        fig, axes = plt.subplots(len(modes) + 1, 1, figsize=(14, 12), sharex=True)
        x = np.arange(clean.shape[0]) / float(args.sample_rate)

        axes[0].plot(x, clean, linewidth=1.0)
        axes[0].set_title("clean")
        axes[0].set_ylabel("mV")
        axes[0].grid(alpha=0.2)

        for idx, mode in enumerate(modes, start=1):
            axes[idx].plot(x, outputs[mode], linewidth=1.0)
            axes[idx].set_title(mode)
            axes[idx].set_ylabel("mV")
            axes[idx].grid(alpha=0.2)

        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(args.plot_path)), exist_ok=True)
        fig.savefig(args.plot_path, dpi=150)
        print(f"Saved plot: {args.plot_path}")


if __name__ == "__main__":
    main()
