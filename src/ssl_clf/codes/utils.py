from datetime import datetime
from typing import List

import numpy as np
import torch
from torch import nn

def get_timestamp() -> str:
    """
    Get timestamp in `yymmdd-hhmmss` format.
    Args:
        None
    Returns:
        timestamp (str): Time stamp in string.
    """
    timestamp = datetime.now()
    timestamp = timestamp.strftime('%Y%m%d-%H%M%S')[2:]
    return timestamp

def calc_class_weight(
    labels: List
) -> np.ndarray:
    """
    Calculate class weight for multilabel task.
    Args:
        labels (np.ndarray): Label data array of shape [num_sample, num_classes]
    Returns:
        class_weight (np.ndarray): Array of shape [num_classes].
    """
    labels = np.array(labels)[:, np.newaxis]
    num_samples = labels.shape[0]

    positive_per_class = labels.sum(axis=0)
    negative_per_class = num_samples - positive_per_class

    class_weight = negative_per_class / positive_per_class

    return class_weight

def aggregator(model: nn.Module, X: torch.Tensor):
    """
    ```
    These predictions are then aggregated using the element-wise maximum
    (or mean in case of age and gender prediction)
    ```
    From paper's appendix I => We use maximum.
    Args:
        model (nn.Module):
        X (torch.Tensor): Tensor of shape (batch_size, num_split, sequence_length).
    Returns:
        aggregated_preds (torch.Tensor): Tensor of shape (batch_size, num_classes).
    """
    aggregated_preds = []
    for i in range(X.size(0)):
        if X[i].dim() == 3:
            data = torch.permute(X[i], (1, 0, 2))
        else:
            data = X[i]
        y_preds = model(data) # data: (num_split, sequence_length)
        _aggregated_preds, _ = torch.max(y_preds, axis=0)
        aggregated_preds.append(_aggregated_preds)
    aggregated_preds = torch.stack(aggregated_preds)
    return aggregated_preds