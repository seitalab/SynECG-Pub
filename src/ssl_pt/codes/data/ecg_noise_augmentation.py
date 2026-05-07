from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt


class ECGNoiseAugmentation:
    """
    ECG noise augmentation for pretraining.

    Args:
        sample_rate (int): Sampling rate in Hz. Default: 500.
        mode (str): One of 'baseline_wander', 'emg', 'motion_artifact',
            'electrode_displacement', 'combined'.
        seed (int, optional): Random seed for reproducibility.
        rng (np.random.Generator, optional): External RNG. If provided,
            `seed` must be None.
    """

    VALID_MODES = {
        "baseline_wander",
        "emg",
        "motion_artifact",
        "electrode_displacement",
        "combined",
    }

    def __init__(
        self,
        sample_rate: int = 500,
        mode: str = "combined",
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown mode `{mode}`. Choose from {sorted(self.VALID_MODES)}.")
        if rng is not None and seed is not None:
            raise ValueError("Specify either `seed` or `rng`, not both.")
        if sample_rate <= 0:
            raise ValueError("`sample_rate` must be a positive integer.")

        self.sample_rate = int(sample_rate)
        self.mode = mode
        self._base_seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        self.probabilities: Dict[str, float] = {
            "baseline_wander": 0.8,
            "emg": 0.7,
            "motion_artifact": 0.5,
            "electrode_displacement": 0.5,
        }
        self.parameter_ranges: Dict[str, Dict[str, Any]] = {
            "baseline_wander": {
                "n_components": [1, 3],
                "amplitude_mv": [0.05, 0.3],
                "frequency_hz": [0.05, 0.5],
                "phase_rad": [0.0, float(2 * np.pi)],
            },
            "emg": {
                "filter_order": 4,
                "band_hz": [30.0, 245.0],
                "snr_db": [10.0, 30.0],
            },
            "motion_artifact": {
                "n_segments": [1, 3],
                "duration_ms": [50.0, 500.0],
                "dc_offset_mv": [-0.5, 0.5],
                "spike_amplitude_mv": [0.5, 2.0],
                "edge_taper_ratio": 0.1,
            },
            "electrode_displacement": {
                "n_events": [1, 2],
                "dc_offset_mv": [-0.3, 0.3],
                "scale_factor": [0.8, 1.2],
                "transition_ms": [10.0, 50.0],
            },
        }
        self._last_call_params: Dict[str, Any] = {}

    def set_seed(self, seed: int) -> None:
        """Reset internal RNG with a new seed (e.g., per dataloader worker)."""
        self._base_seed = int(seed)
        self.rng = np.random.default_rng(self._base_seed)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to a single ECG signal.

        Args:
            signal: 1D numpy array of shape (num_samples,), e.g. (5000,)

        Returns:
            Augmented signal of same shape.
        """
        input_signal = np.asarray(signal)
        if input_signal.ndim != 1:
            raise ValueError(f"`signal` must be 1D, got shape {input_signal.shape}.")
        if input_signal.size == 0:
            raise ValueError("`signal` must be non-empty.")

        out_dtype = input_signal.dtype if np.issubdtype(input_signal.dtype, np.floating) else np.float32
        augmented = input_signal.astype(np.float64, copy=True)

        self._last_call_params = {
            "mode": self.mode,
            "sample_rate": self.sample_rate,
            "num_samples": int(augmented.size),
            "applied_order": [],
            "augmentations": {},
        }

        if self.mode == "combined":
            ordered = [
                ("baseline_wander", self._apply_baseline_wander),
                ("emg", self._apply_emg_noise),
                ("motion_artifact", self._apply_motion_artifact),
                ("electrode_displacement", self._apply_electrode_displacement),
            ]
            for aug_name, aug_func in ordered:
                augmented = self._apply_with_probability(augmented, aug_name, aug_func)
        else:
            mode_to_func = {
                "baseline_wander": self._apply_baseline_wander,
                "emg": self._apply_emg_noise,
                "motion_artifact": self._apply_motion_artifact,
                "electrode_displacement": self._apply_electrode_displacement,
            }
            augmented = self._apply_with_probability(
                augmented,
                self.mode,
                mode_to_func[self.mode],
            )

        return augmented.astype(out_dtype, copy=False)

    def _apply_with_probability(self, signal: np.ndarray, aug_name: str, func) -> np.ndarray:
        apply_prob = self.probabilities[aug_name]
        is_applied = bool(self.rng.random() < apply_prob)
        log_info: Dict[str, Any] = {
            "probability": apply_prob,
            "applied": is_applied,
        }

        if is_applied:
            signal, sampled_params = func(signal)
            log_info.update(sampled_params)
            self._last_call_params["applied_order"].append(aug_name)

        self._last_call_params["augmentations"][aug_name] = log_info
        return signal

    def _apply_baseline_wander(self, signal: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        num_components = int(self.rng.integers(1, 4))
        time_axis = np.arange(signal.size, dtype=np.float64) / float(self.sample_rate)

        noise = np.zeros_like(signal)
        components: List[Dict[str, float]] = []
        for _ in range(num_components):
            amplitude = float(self.rng.uniform(0.05, 0.3))
            frequency = float(self.rng.uniform(0.05, 0.5))
            phase = float(self.rng.uniform(0.0, 2 * np.pi))

            noise += amplitude * np.sin(2 * np.pi * frequency * time_axis + phase)
            components.append(
                {
                    "amplitude_mv": amplitude,
                    "frequency_hz": frequency,
                    "phase_rad": phase,
                }
            )

        return signal + noise, {
            "num_components": num_components,
            "components": components,
        }

    def _apply_emg_noise(self, signal: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        nyquist = 0.5 * self.sample_rate
        low_hz = 30.0
        high_hz = min(245.0, nyquist - 1e-3)
        if high_hz <= low_hz:
            raise ValueError(
                f"EMG bandpass invalid for sample_rate={self.sample_rate}. "
                f"Need Nyquist > {low_hz} Hz."
            )

        white_noise = self.rng.normal(loc=0.0, scale=1.0, size=signal.size)
        sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=float(self.sample_rate), output="sos")
        band_limited = sosfiltfilt(sos, white_noise)

        target_snr_db = float(self.rng.uniform(10.0, 30.0))
        signal_power = float(np.mean(signal ** 2))
        noise_power = float(np.mean(band_limited ** 2))

        if signal_power <= 1e-12 or noise_power <= 1e-12:
            scaled_noise = np.zeros_like(signal)
            scale_factor = 0.0
            achieved_snr_db = np.inf
        else:
            target_noise_power = signal_power / (10.0 ** (target_snr_db / 10.0))
            scale_factor = float(np.sqrt(target_noise_power / noise_power))
            scaled_noise = band_limited * scale_factor
            final_noise_power = float(np.mean(scaled_noise ** 2))
            achieved_snr_db = float(10.0 * np.log10(signal_power / max(final_noise_power, 1e-12)))

        return signal + scaled_noise, {
            "target_snr_db": target_snr_db,
            "achieved_snr_db": float(achieved_snr_db),
            "scale_factor": scale_factor,
            "filter_order": 4,
            "band_hz": [low_hz, high_hz],
        }

    def _apply_motion_artifact(self, signal: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        if signal.size < 2:
            return signal, {"num_segments": 0, "segments": []}

        min_len = max(1, int(round(0.05 * self.sample_rate)))
        max_len = max(min_len, int(round(0.5 * self.sample_rate)))
        max_len = min(max_len, signal.size)

        num_segments = int(self.rng.integers(1, 4))
        augmented = signal.copy()
        segments: List[Dict[str, Any]] = []

        for _ in range(num_segments):
            seg_len = int(self.rng.integers(min_len, max_len + 1))
            start = int(self.rng.integers(0, signal.size - seg_len + 1))
            end = start + seg_len

            dc_offset = float(self.rng.uniform(-0.5, 0.5))
            spike_amp = float(self.rng.uniform(0.5, 2.0))
            seg_idx = np.arange(seg_len, dtype=np.float64)
            center = 0.5 * (seg_len - 1)
            sigma = max(seg_len / 4.0, 1.0)

            gaussian = spike_amp * np.exp(-0.5 * ((seg_idx - center) / sigma) ** 2)
            artifact = dc_offset + gaussian
            artifact = self._apply_hanning_taper(artifact)

            augmented[start:end] += artifact
            segments.append(
                {
                    "start_sample": start,
                    "end_sample": end,
                    "length_samples": seg_len,
                    "dc_offset_mv": dc_offset,
                    "spike_amplitude_mv": spike_amp,
                    "sigma_samples": sigma,
                }
            )

        return augmented, {
            "num_segments": num_segments,
            "segments": segments,
        }

    def _apply_hanning_taper(self, segment: np.ndarray) -> np.ndarray:
        length = segment.size
        if length < 4:
            return segment

        edge = int(round(0.1 * length))
        edge = min(max(edge, 1), length // 2)
        if edge == 0:
            return segment

        taper = np.ones(length, dtype=np.float64)
        hann = np.hanning(2 * edge)
        taper[:edge] = hann[:edge]
        taper[-edge:] = hann[edge:]
        return segment * taper

    def _apply_electrode_displacement(self, signal: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        num_events = int(self.rng.integers(1, 3))
        augmented = signal.copy()
        sample_idx = np.arange(signal.size, dtype=np.float64)

        min_transition = max(1, int(round(0.01 * self.sample_rate)))
        max_transition = max(min_transition, int(round(0.05 * self.sample_rate)))
        events: List[Dict[str, Any]] = []

        for _ in range(num_events):
            event_sample = int(self.rng.integers(0, signal.size))
            dc_offset = float(self.rng.uniform(-0.3, 0.3))
            scale_factor = float(self.rng.uniform(0.8, 1.2))
            transition_len = int(self.rng.integers(min_transition, max_transition + 1))

            # Smooth but sharp transition around the event point.
            slope = 10.0 / max(transition_len, 1)
            logits = np.clip(slope * (sample_idx - event_sample), -60.0, 60.0)
            transition = 1.0 / (1.0 + np.exp(-logits))

            transformed = augmented * scale_factor + dc_offset
            augmented = transformed * transition + augmented * (1.0 - transition)

            events.append(
                {
                    "event_sample": event_sample,
                    "dc_offset_mv": dc_offset,
                    "scale_factor": scale_factor,
                    "transition_samples": transition_len,
                }
            )

        return augmented, {
            "num_events": num_events,
            "events": events,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return augmentation configuration for logging."""
        return {
            "sample_rate": self.sample_rate,
            "mode": self.mode,
            "seed": self._base_seed,
            "probabilities": deepcopy(self.probabilities),
            "parameter_ranges": deepcopy(self.parameter_ranges),
            "last_call": deepcopy(self._last_call_params),
        }
