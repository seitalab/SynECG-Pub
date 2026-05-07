import numpy as np
import torch

np.random.seed(0)

class RandomMask:

    def __init__(self, mask_ratio: float):

        self.mask_ratio = mask_ratio

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]
        mask_width = int(data.shape[0] * self.mask_ratio)
        mask_start = np.random.choice(data.shape[0] - mask_width, 1)[0]

        masked_data = data.copy()
        masked_data[mask_start:mask_start+mask_width] = 0

        return {"data": masked_data}

class RandomShift:

    def __init__(self, max_shift_ratio: float):

        self.max_shift_ratio = max_shift_ratio

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]
        shift_ratio = np.random.rand() * self.max_shift_ratio
        shift_size = int(data.shape[0] * shift_ratio)

        # pad = np.zeros(shift_size)
        pad = np.zeros_like(data)[:shift_size]

        shifted_data = data.copy()
        if np.random.rand() < 0.5:
            shifted_data = np.concatenate([pad, shifted_data])[:len(data)]
        else:
            shifted_data = np.concatenate([shifted_data, pad])[-len(data):]
        assert len(data) == len(shifted_data)
        return {"data": shifted_data}

class AlignLength:

    def __init__(self, target_len: int):

        self.target_len = target_len

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]

        if len(data) < self.target_len:
            total_pad = self.target_len - len(data)
            pad_l = int(np.random.rand() * total_pad)
            pad_r = total_pad - pad_l
            data = np.concatenate([
                np.zeros(pad_l),
                data,
                np.zeros(pad_r)
            ])
        
        if len(data) > self.target_len:
            total_cut = len(data) - self.target_len
            cut_l = int(np.random.rand() * total_cut)
            data = data[cut_l:cut_l+self.target_len]

        return {"data": data}

class ScaleECG:

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]

        data = (data - data.mean(axis=0)) / data.std(axis=0)
        # data = (data - data.mean()) / data.std()

        # data = data[::10]

        return {"data": data}

class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    # def __init__(self, modelname: str):

    #     self.is_mae = modelname == "mae_base"

    def __call__(self, sample):

        data = sample["data"]
        data_tensor = torch.from_numpy(data)
        # if self.is_mae:
        data_tensor = data_tensor.unsqueeze(0)
        sample = {"data": data_tensor}
        return sample

class Subsample:
    """
    Subsample fixed length of ECG signals.
    Args:
        subsample_length (int): Length of subsampled data.
    """
    def __init__(self, subsample_length: int, n_lead: int=1):

        assert isinstance(subsample_length, int)
        self.subsample_length = subsample_length
        self.n_lead = n_lead

    def _pad(self, data):
        """
        Args:
            data (np.ndarray):
        Returns:
            padded_data (np.ndarray):
        """
        pad_length = self.subsample_length - data.shape[0]

        if pad_length == 0:
            return data
        pad = np.zeros([pad_length])
        pad_data = np.concatenate([data, pad], axis=-1)
        return pad_data

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (sequence_length)}
        Returns:
            sample (Dict): {"data": Array of shape (subsample_length)}
        """
        data = sample["data"]
        if len(data) > self.subsample_length:
            start = np.random.randint(0, data.shape[0] - self.subsample_length)        
            subsampled_data = data[start:start+self.subsample_length]
        else:
            subsampled_data = self._pad(data)

        return {"data": subsampled_data}

class SubsampleEval(Subsample):
    """
    Subsampling for evaluation mode.
    Args:
        subsample_length (int): Length of subsampled data.
    """

    def _pad_signal(self, data):
        """
        Args:
            data (np.ndarray):
        Returns:
            padded_data (np.ndarray):
        """
        chunk_length = self.subsample_length // 2
        pad_length = chunk_length - data.shape[0] % chunk_length

        if pad_length == 0:
            return data
        pad = np.zeros([pad_length])
        pad_data = np.concatenate([data, pad], axis=-1)
        return pad_data

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (sequence_length).}
        Returns:
            sample (Dict): {"data": Array of shape (num_split, subsample_length).}
        """
        data = sample["data"]
        slice_indices = np.arange(0, data.shape[0], self.subsample_length // 2)
        index_range = np.arange(self.subsample_length)
        target_locs = slice_indices[:, np.newaxis] + index_range[np.newaxis]

        padded_data = self._pad_signal(data)[np.newaxis]
        try:
            eval_subsamples = padded_data[:, target_locs]
        except:
            eval_subsamples = padded_data[:, target_locs[:-1]]
        return {"data": eval_subsamples[0]}