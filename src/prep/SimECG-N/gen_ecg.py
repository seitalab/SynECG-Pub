import os
import signal
import pickle
from argparse import Namespace

import yaml
import numpy as np
from tqdm import tqdm

from augment import Augment

cfg_file = "../../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

def generate_peak_wave(
    t: np.ndarray, 
    peak: float, 
    shift: float, 
    width: float, 
    flip_peak: bool=False
):
    wave = peak * np.exp(-0.5 * ((t - shift) / width) ** 2)
    if flip_peak:
        wave = -1 * wave
    return wave

def base_shift(
    wave: np.ndarray, 
    start_val: float,
    base_scale: float
):
    last = start_val + np.random.randn() * base_scale
    base = np.linspace(start_val, last, len(wave))
    return wave + base

def white_noise(
    wn_length, 
    noise_width, 
    scaler
):
    pad = int(wn_length - int(wn_length / noise_width) * noise_width)
    wn = np.random.randn(
        int(wn_length / noise_width)) / scaler
    wn = wn.repeat(noise_width)
    if pad > 0:
        wn = np.concatenate([wn, np.zeros(pad)])
    return wn

def change_sample(original_signal, factor):
    sampled_indices = np.arange(
        0, len(original_signal), factor)
    sampled_signal = np.interp(
        sampled_indices, 
        np.arange(len(original_signal)), 
        original_signal
    )
    return sampled_signal

def smooth_connection(array1, array2, window_size):
    if window_size <= 0:
        return np.concatenate((array1, array2))

    # Calculate the moving average for the last x steps of array1
    last_avg = np.mean(array1[-window_size:], axis=0)

    # Calculate the moving average for the first x steps of array2
    first_avg = np.mean(array2[:window_size], axis=0)

    # Calculate the weighted average for the smoothed connection
    smoothing_factor = np.linspace(0, 1, window_size)
    smoothed_connection = last_avg * (1 - smoothing_factor) + first_avg * smoothing_factor

    # Concatenate the arrays with the smoothed connection
    smoothed_result = np.concatenate((array1[:-window_size], smoothed_connection, array2[window_size:]))

    return smoothed_result

def handle_timeout(signum, frame):
    print("Too long -> reset processing")
    raise TimeoutError("Overtime")

class ECGsynthesizer:
    
    syn_type = "syn_ecg"

    def __init__(self, seed: int, syn_mode: str="clf"):
        np.random.seed(seed)
        self.seed = seed
        self.syn_mode = syn_mode

        self.base_len = 1
        self.val_size = cfg["settings"]["common"]["val_size"]
        self.fs = cfg["settings"]["common"]["target_freq"]
        self.target_length = self.fs * cfg["settings"]["common"]["duration"]

        self._prep_save_loc()
        self._set_cfg()

        self.augmentor = Augment(self.fs)

    def _prep_save_loc(self):

        self.save_loc = os.path.join(
            cfg["settings"]["common"]["save_root"], 
            cfg["settings"][self.syn_type]["syncfg"]
        )
        os.makedirs(self.save_loc, exist_ok=True)

    def _set_cfg(self):
        syncfg_file = os.path.join(
            cfg["settings"]["common"]["syncfg_root"],
            f'{cfg["settings"][self.syn_type]["syncfg"]}.yaml'
        )
        with open(syncfg_file, "r") as f:
            syncfg = yaml.safe_load(f)
        self.cfg = syncfg["params"]
        self._save_data(None, "cfg", ext="txt")

    def generate_beat(self, start_val: float, beat_params: Namespace):
        """
        Args:
            beat_params (Namespace): 
        Returns:
            pseudo_ecg (np.ndarray): 
        """
        t = np.linspace(
            0, 
            self.base_len, 
            int(self.base_len * self.fs), 
            endpoint=False
        )

        # P-wave
        p_wave = generate_peak_wave(
            t, 
            beat_params.p_peak, 
            beat_params.p_shift, 
            beat_params.p_width
        )

        # QRS complex
        q = generate_peak_wave(
            t,
            beat_params.q_peak, 
            beat_params.q_shift, 
            beat_params.q_width, True
        )
        r = generate_peak_wave(
            t, 
            beat_params.r_peak, 
            beat_params.r_shift, 
            beat_params.r_width
        )
        s = generate_peak_wave(
            t, 
            beat_params.s_peak, 
            beat_params.s_shift, 
            beat_params.s_width, 
            True
        )
        qrs = q + r + s

        # T-wave
        t_wave = generate_peak_wave(
            t, 
            beat_params.t_peak, 
            beat_params.t_shift, 
            beat_params.t_width
        )

        pseudo_ecg = p_wave + qrs + t_wave

        # change length
        rel_length = self.base_len / beat_params.length
        pseudo_ecg = change_sample(pseudo_ecg, rel_length)

        # Noise
        wn1 = white_noise(
            pseudo_ecg.shape[0],
            int(beat_params.wn1_width), 
            beat_params.wn1_scaler
        )
        wn2 = white_noise(
            pseudo_ecg.shape[0],
            int(beat_params.wn2_width), 
            beat_params.wn2_scaler
        )

        pseudo_ecg = pseudo_ecg + wn1 + wn2
        pseudo_ecg = base_shift(pseudo_ecg, start_val, beat_params.base_scale)
        return pseudo_ecg        
        
    def set_base_param(self):
        """
        Args:
            None
        Returns:
            base_param (Namespace): 
        """
        base_param = {}
        for key in self.cfg:
            param_val = self.cfg[key]["base"]["val"]
            if self.cfg[key]["base"]["shift"] is not None:
                param_val += base_param[self.cfg[key]["base"]["shift"]]

            # Add noise.
            noise_info = self.cfg[key]["base_perturb"]
            if noise_info["type"] == "normal":
                param_val += np.random.normal(scale=noise_info["sdev"])
            elif noise_info["type"] == "uniform":
                rand_val = np.random.random()
                scale = noise_info["max"] - noise_info["min"]
                rand_val = rand_val * scale + noise_info["min"]
                param_val += rand_val
            base_param[key] = param_val
        return Namespace(**base_param)
    
    def perturb_param(self, base_params: Namespace):
        """
        Args:
            base_param (Namespace): 
        Returns:
            beat_param (Namespace): 
        """
        perturbed_param = {}
        for key, value in vars(base_params).items():
            # Add noise.
            noise_info = self.cfg[key]["beat_perturb"]
            if noise_info["type"] == "normal":
                value += np.random.normal(scale=noise_info["sdev"])         
            elif noise_info["type"] == "uniform":
                rand_val = np.random.random()
                scale = noise_info["max"] - noise_info["min"]
                rand_val = rand_val * scale + noise_info["min"]
                value += rand_val
            perturbed_param[key] = value
        return Namespace(**perturbed_param)

    def generate_ecg(self):
        """
        Args:
        
        Returns:
        
        """
        p_ecg = np.array([0])
        base_params = self.set_base_param()
        while True:
            beat_params = self.perturb_param(base_params)
            _p_ecg = self.generate_beat(p_ecg[-1], beat_params)
            p_ecg = np.concatenate([p_ecg, _p_ecg])

            if len(p_ecg) > self.target_length*1.5:
                break
        # Randomly pick location.
        start_loc = np.random.choice(len(p_ecg) - self.target_length)
        p_ecg = p_ecg[start_loc:start_loc+self.target_length]

        n_aug = np.random.poisson(lam=2.)
        for _ in range(n_aug):
            p_ecg = self.augmentor.rand_augment(p_ecg)

        return p_ecg

    def _save_data(self, data, datatype: str, ext: str="pkl"):
        """
        Args:

        Returns:

        """
        savename = os.path.join(
            self.save_loc,
            f"{datatype}_seed{self.seed:04d}.{ext}"
        )
        if ext == "pkl":
            with open(savename, "wb") as fp:
                pickle.dump(data, fp)
        elif ext == "txt":
            cfg = ""
            for k, v in self.cfg.items():
                cfg += f"{k} : {v}\n"
            with open(savename, "w") as f:
                f.write(cfg.strip())
        else:
            raise

    def make_dataset(self):
        """
        Args:

        Returns:

        """
        # Generate.
        syn_ecg = []
        num_to_synthesize = cfg["settings"]["common"]["n_syn"][self.syn_mode]
        for _ in tqdm(range(num_to_synthesize)):
            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(cfg["settings"]["common"]["max_process_time"])
            try:
                ecg = self.generate_ecg()
                syn_ecg.append(ecg)
            except TimeoutError:
                pass        
        signal.alarm(0)
        syn_ecg = np.array(syn_ecg)
        
        # Split.
        n_train = int(len(syn_ecg) * (1 - self.val_size))
        train = syn_ecg[:n_train]
        val = syn_ecg[n_train:]
        
        # Store.
        print(f"Saving at `{self.save_loc}`")
        self._save_data(train, "train")
        self._save_data(val, "val")

if __name__ == "__main__":

    seed = 7
    syn_mode = "pt"
    syn = ECGsynthesizer(seed=seed, syn_mode=syn_mode)
    syn.make_dataset()