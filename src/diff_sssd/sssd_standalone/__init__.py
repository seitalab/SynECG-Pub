"""
SSSD-ECG: Structured State Space Diffusion Model for ECG Generation

This is a standalone implementation that can be easily copied to other projects.
"""

from .model_wrapper import SSSDECG
from .dataset import ECGDataset, PTBXLDataset, create_dataloaders

__version__ = "1.0.0"
__all__ = [
    "SSSDECG",
    "ECGDataset",
    "PTBXLDataset",
    "create_dataloaders",
]
