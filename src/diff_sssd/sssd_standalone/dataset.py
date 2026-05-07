"""
ECG Dataset Classes
Provides PyTorch Dataset classes for loading ECG data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG data.

    Supports loading from:
    - Numpy arrays (.npy files)
    - Direct numpy arrays in memory

    Args:
        data_path (str or np.ndarray): Path to .npy file or numpy array of ECG signals
                                        shape=(num_samples, channels, length)
        labels_path (str or np.ndarray, optional): Path to .npy file or numpy array of labels
                                                    shape=(num_samples,) or (num_samples, num_classes)
        transform (callable, optional): Optional transform to apply to ECG signals
        target_transform (callable, optional): Optional transform to apply to labels
        lead_indices (list, optional): Indices of leads to select. E.g., [0,1,2,3,4,5,6,7] for 8 leads
        segment_length (int, optional): Length of ECG segments. If provided, will crop/pad to this length
    """

    def __init__(
        self,
        data_path,
        labels_path=None,
        transform=None,
        target_transform=None,
        lead_indices=None,
        segment_length=None
    ):
        super(ECGDataset, self).__init__()

        # Load data
        if isinstance(data_path, str):
            self.data = np.load(data_path)
        elif isinstance(data_path, np.ndarray):
            self.data = data_path
        else:
            raise ValueError("data_path must be a string (file path) or numpy array")

        # Load labels if provided
        if labels_path is not None:
            if isinstance(labels_path, str):
                self.labels = np.load(labels_path)
            elif isinstance(labels_path, np.ndarray):
                self.labels = labels_path
            else:
                raise ValueError("labels_path must be a string (file path) or numpy array")
        else:
            # Create dummy labels if not provided
            self.labels = np.zeros(len(self.data), dtype=np.int64)

        # Validate shapes
        assert len(self.data) == len(self.labels), \
            f"Data and labels must have same length: {len(self.data)} vs {len(self.labels)}"

        self.transform = transform
        self.target_transform = target_transform
        self.lead_indices = lead_indices
        self.segment_length = segment_length

    def __len__(self):
        return len(self.data)

    def _ensure_channel_first(self, ecg):
        """
        Normalize a single ECG sample to shape=(channels, length).
        """
        if ecg.ndim != 2:
            raise ValueError(
                f"Each ECG sample must be 2D (channels,length) or (length,channels). Got shape={ecg.shape}"
            )

        # Prefer explicit detection using expected segment length when available.
        if self.segment_length is not None:
            is_time_first = (
                ecg.shape[0] == self.segment_length and
                ecg.shape[1] != self.segment_length
            )
            is_channel_first = (
                ecg.shape[1] == self.segment_length and
                ecg.shape[0] != self.segment_length
            )
            if is_time_first:
                ecg = ecg.T
            elif (not is_channel_first) and ecg.shape[0] > ecg.shape[1] and ecg.shape[1] <= 32:
                # Fallback heuristic: (length, channels) like (1000, 12)
                ecg = ecg.T
        elif ecg.shape[0] > ecg.shape[1] and ecg.shape[1] <= 32:
            ecg = ecg.T

        return ecg

    def __getitem__(self, idx):
        # Get ECG signal and label
        ecg = self.data[idx]
        label = self.labels[idx]

        # Support both (channels, length) and (length, channels) source formats.
        ecg = self._ensure_channel_first(ecg)

        # Select specific leads if specified
        if self.lead_indices is not None:
            max_lead_idx = max(self.lead_indices)
            if max_lead_idx >= ecg.shape[0]:
                raise IndexError(
                    f"Lead index {max_lead_idx} out of bounds for ECG sample with shape {ecg.shape}"
                )
            ecg = ecg[self.lead_indices]

        # Crop or pad to segment_length if specified
        if self.segment_length is not None:
            current_length = ecg.shape[-1]
            if current_length > self.segment_length:
                # Crop
                ecg = ecg[..., :self.segment_length]
            elif current_length < self.segment_length:
                # Pad
                pad_length = self.segment_length - current_length
                ecg = np.pad(ecg, ((0, 0), (0, pad_length)), mode='constant')

        # Convert to tensors
        ecg = torch.from_numpy(ecg).float()

        # Handle label format
        if isinstance(label, np.ndarray) and len(label.shape) > 0:
            label = torch.from_numpy(label).float()
        else:
            label = torch.tensor(label, dtype=torch.float32)

        # Apply transforms if provided
        if self.transform is not None:
            ecg = self.transform(ecg)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return ecg, label


class ECGDatasetFromFiles(Dataset):
    """
    ECG Dataset that loads data with configurable file paths.

    This is a convenience class for the common use case of loading
    ECG data and labels from separate .npy files.

    Args:
        data_file (str): Path to .npy file containing ECG signals
        labels_file (str): Path to .npy file containing labels
        **kwargs: Additional arguments passed to ECGDataset
    """

    def __init__(self, data_file, labels_file, **kwargs):
        self.dataset = ECGDataset(
            data_path=data_file,
            labels_path=labels_file,
            **kwargs
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class PTBXLDataset(ECGDataset):
    """
    Specialized dataset for PTB-XL ECG data.

    PTB-XL specific features:
    - Defaults to 8-lead ECG (leads I, II, V1-V6)
    - Defaults to 1000 sample length (10 seconds at 100 Hz)
    - Multi-label classification (71 classes)

    Args:
        data_path (str): Path to PTB-XL data .npy file
        labels_path (str): Path to PTB-XL labels .npy file
        use_8_leads (bool): If True, select 8 leads (I, II, V1-V6). Default: True
        **kwargs: Additional arguments passed to ECGDataset
    """

    def __init__(
        self,
        data_path,
        labels_path,
        use_8_leads=True,
        **kwargs
    ):
        # Default lead indices for 8-lead ECG
        # PTB-XL has 12 leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        # We select: I, II, V1, V2, V3, V4, V5, V6 (indices 0, 1, 6, 7, 8, 9, 10, 11)
        lead_indices = [0, 1, 6, 7, 8, 9, 10, 11] if use_8_leads else None

        # Default segment length for PTB-XL
        segment_length = kwargs.pop('segment_length', 1000)

        super(PTBXLDataset, self).__init__(
            data_path=data_path,
            labels_path=labels_path,
            lead_indices=lead_indices,
            segment_length=segment_length,
            **kwargs
        )


def create_dataloaders(
    train_data_path,
    train_labels_path,
    val_data_path=None,
    val_labels_path=None,
    batch_size=8,
    num_workers=4,
    shuffle_train=True,
    lead_indices=None,
    segment_length=1000,
    dataset_class=ECGDataset
):
    """
    Convenience function to create train and validation dataloaders.

    Args:
        train_data_path (str): Path to training data .npy file
        train_labels_path (str): Path to training labels .npy file
        val_data_path (str, optional): Path to validation data .npy file
        val_labels_path (str, optional): Path to validation labels .npy file
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes for data loading
        shuffle_train (bool): Whether to shuffle training data
        lead_indices (list, optional): Indices of leads to select
        segment_length (int): Length of ECG segments
        dataset_class (class): Dataset class to use (ECGDataset or PTBXLDataset)

    Returns:
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader or None): Validation dataloader if val paths provided
    """
    from torch.utils.data import DataLoader

    # Create training dataset
    train_dataset = dataset_class(
        data_path=train_data_path,
        labels_path=train_labels_path,
        lead_indices=lead_indices,
        segment_length=segment_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    # Create validation dataset if provided
    val_loader = None
    if val_data_path is not None and val_labels_path is not None:
        val_dataset = dataset_class(
            data_path=val_data_path,
            labels_path=val_labels_path,
            lead_indices=lead_indices,
            segment_length=segment_length
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True
        )

    return train_loader, val_loader
