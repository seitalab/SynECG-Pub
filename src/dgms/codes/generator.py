import os
import pickle
from glob import glob
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from codes.models.model import prepare_model
from codes.utils.utils import make_ecg_plot

class GeneratedSampleManager:

    def __init__(self, n_per_file, save_dir, draw_samples: int=0):

        self.n_total_samples = 0
        self.n_per_file = n_per_file
        self.n_samples = 0
        self.samples = None

        self.save_dir = os.path.join(save_dir, "samples")
        os.makedirs(self.save_dir, exist_ok=True)

        self.draw_samples = draw_samples
        if draw_samples > 0:
            self.sample_image_dir = os.path.join(save_dir, "sample_images")
            os.makedirs(self.sample_image_dir, exist_ok=True)
        
        self.time_stamps = [(0, datetime.now())]
    
    def _concat_array(self, array):

        if self.samples is None:
            return array
        return np.concatenate([self.samples, array], axis=0)

    def keep_samples(self, samples):

        self.n_total_samples += samples.shape[0]
        self.n_samples += samples.shape[0]
        self.samples = self._concat_array(samples)

    def save_samples(self, force_save=False):

        if force_save:
            save_sample = True
        else:
            save_sample = self.n_samples >= self.n_per_file
        if save_sample:
            idx = self.n_total_samples // self.n_per_file
            savename = os.path.join(
                self.save_dir,
                f"idx{idx:06d}.pkl"
            )
            savesamples = self.samples[:self.n_per_file]
            with open(savename, "wb") as f:
                pickle.dump(savesamples, f)
            self.samples = self.samples[self.n_per_file:]
            self.n_samples -= self.n_per_file
        
        if self.draw_samples > 0:
            for i in range(self.draw_samples):
                savename = f"{self.sample_image_dir}/{i+1:04d}.png"
                make_ecg_plot(self.samples[i, 0], 10, 500, savename)
            self.draw_samples = 0

    def add_time_stamp(self):
            
        self.time_stamps.append((self.n_total_samples,datetime.now()))

    def save_time_stamps(self):

        with open(os.path.join(self.save_dir, "../time_stamps.txt"), "w") as f:
            for ndata, time_stamp in self.time_stamps:
                tstamp_line = f"{ndata},{time_stamp}"
                f.write(tstamp_line + "\n")

class Generator:

    def __init__(self, args, save_dir):

        self.args = args
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.model = None
        self.manager = GeneratedSampleManager(
            args.n_per_file, save_dir, draw_samples=5
        )
        self._store_args(args)

    def _store_args(self, args):
            
        with open(os.path.join(self.save_dir, "args.pkl"), "wb") as f:
            pickle.dump(args, f)
        
        args = vars(args)
        with open(os.path.join(self.save_dir, "args.txt"), "w") as f:
            for key, value in args.items():
                f.write(f"{key}: {value}\n")

    def set_model(self):
        model = prepare_model(self.args)
        model = model.to(self.args.device)

        self.model = model

    def set_weight(self, weight_file_dir: str):

        assert self.model is not None

        weight_file = glob(
            weight_file_dir + "/??????-??????-*/net.pth")[0]
        self.model.to("cpu")

        # Temporal solution.
        state_dict = dict(torch.load(weight_file, map_location="cpu")) # OrderedDict -> dict
        
        old_keys = list(state_dict.keys())
        for key in old_keys:
            state_dict[key] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)

        self.model.to(self.args.device)

    def _calc_epochs(self):

        return self.args.n_total_samples // self.args.batch_size + 1

    def run(self):

        epochs = self._calc_epochs()
        self.model.eval()
        for _ in tqdm(range(epochs)):
            z = torch.randn(
                self.args.batch_size, self.model.z_dim
            ).to(self.args.device)
            generated = self.model.generate(z)
            generated = generated.cpu().detach().numpy()
            self.manager.keep_samples(generated)
            self.manager.add_time_stamp()
            self.manager.save_samples()
        self.manager.save_time_stamps()
        self.manager.save_samples(force_save=True)
        print("Done")
