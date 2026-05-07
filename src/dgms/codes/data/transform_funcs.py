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
            sample (Dict): {
                "data": Array of shape (data_length, ),
                "mask": Array of shape (data_length, ) # 1: not masked, 0: masked
            }
        """
        data = sample["data"]
        mask_width = int(data.shape[0] * self.mask_ratio)
        mask_start = np.random.choice(data.shape[0] - mask_width, 1)[0]

        # masked_data = data.copy()
        # masked_data[mask_start:mask_start+mask_width] = 0
        mask = np.ones_like(data)
        mask[mask_start:mask_start+mask_width] = 0
        # masked_data = data * mask

        return {"data": data, "mask": mask}

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
        mask = sample["mask"]
        shift_ratio = np.random.rand() * self.max_shift_ratio
        shift_size = int(data.shape[0] * shift_ratio)

        pad = np.zeros(shift_size)

        shifted_data = data.copy()
        if np.random.rand() < 0.5:
            shifted_data = np.concatenate([pad, shifted_data])[:len(data)]
            shifted_mask = np.concatenate([pad.copy(), mask])[:len(data)]
        else:
            shifted_data = np.concatenate([shifted_data, pad])[-len(data):]
            shifted_mask = np.concatenate([mask, pad.copy()])[-len(data):]
        assert len(data) == len(shifted_data)
        sample.update({"data": shifted_data})
        sample.update({"mask": shifted_mask})
        return sample

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

        # return {"data": data}
        sample.update({"data": data})
        return sample

class ScaleECG:

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]

        data = (data - data.mean()) / data.std()

        # data = data[::10]

        sample.update({"data": data})

        return sample

class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        data_tensor = torch.from_numpy(sample["data"])
        data_tensor = data_tensor.unsqueeze(0)
        sample.update({"data": data_tensor})

        if "mask" in sample:
            mask_tensor = torch.from_numpy(sample["mask"])
            sample.update({"mask": mask_tensor})
        return sample
