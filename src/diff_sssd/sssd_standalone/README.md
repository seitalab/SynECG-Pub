# SSSD-ECG Standalone Implementation

This is a standalone implementation of SSSD-ECG (Structured State Space Diffusion Model for ECG Generation) that can be easily copied to other projects.

## Features

- **Clean API**: Simple and intuitive interface for training and inference
- **Modular Design**: Separated data loading, model implementation, and training logic
- **Easy to Use**: Minimal setup required to get started
- **Flexible**: Supports custom datasets and configurations
- **Well Documented**: Comprehensive documentation and examples

## Directory Structure

```
sssd_standalone/
├── __init__.py              # Package initialization
├── model_wrapper.py         # SSSDECG wrapper class
├── dataset.py              # Dataset classes (ECGDataset, PTBXLDataset)
├── models/
│   ├── __init__.py
│   ├── SSSD_ECG.py         # Main diffusion model
│   └── S4Model.py          # S4 (Structured State Space) layer
├── utils/
│   ├── __init__.py
│   └── util.py             # Utility functions
├── config/
│   └── config_SSSD_ECG.json  # Model configuration
├── examples/
│   ├── train_new.py        # Training script
│   └── example_usage.py    # Usage examples
└── README.md               # This file
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.10
- NumPy
- (Optional) CUDA for GPU acceleration

Install dependencies:

```bash
pip install -r requirements.txt
```

### Quick Setup

Simply copy the `sssd_standalone` folder to your project:

```bash
cp -r sssd_standalone /path/to/your/project/
```

## Quick Start

### Basic Usage

```python
import torch
from sssd_standalone import SSSDECG, ECGDataset
from torch.utils.data import DataLoader

# 1. Create dataset
dataset = ECGDataset(
    data_path="your_ecg_data.npy",  # Shape: (num_samples, channels, length)
    labels_path="your_labels.npy",   # Shape: (num_samples,) or (num_samples, num_classes)
    segment_length=1000
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 2. Initialize model
model = SSSDECG(config_path="sssd_standalone/config/config_SSSD_ECG.json")

# 3. Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# 4. Training loop
for x, y in dataloader:
    loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
```

### Generate Samples

```python
from sssd_standalone import SSSDECG
import torch

# Initialize model
model = SSSDECG(config_path="sssd_standalone/config/config_SSSD_ECG.json")

# Load trained checkpoint
model.load_checkpoint("checkpoints/100000.pkl")

# Generate samples with random labels
samples = model.generate(num_samples=10, return_numpy=True)
print(f"Generated shape: {samples.shape}")  # (10, 8, 1000)

# Generate samples with specific labels
labels = torch.tensor([0, 5, 10, 15, 20])  # Class indices
samples = model.generate(labels=labels, return_numpy=True)
```

## API Reference

### SSSDECG Class

The main model wrapper class providing a clean interface.

#### `__init__(config_path=None, device=None)`

Initialize the model.

**Parameters:**
- `config_path` (str): Path to configuration JSON file. Default: `"config/config_SSSD_ECG.json"`
- `device` (str): Device to use. Default: `"cuda"` if available, else `"cpu"`

#### `forward(x, y)`

Compute training loss.

**Parameters:**
- `x` (torch.Tensor): ECG signals, shape=(batch_size, channels, length)
- `y` (torch.Tensor): Labels, shape=(batch_size, num_classes) or (batch_size,)

**Returns:**
- `loss` (torch.Tensor): Scalar loss value

#### `generate(labels=None, num_samples=1, return_numpy=False)`

Generate ECG samples.

**Parameters:**
- `labels` (torch.Tensor, optional): Conditioning labels
- `num_samples` (int): Number of samples to generate
- `return_numpy` (bool): If True, return numpy array

**Returns:**
- `samples` (torch.Tensor or np.ndarray): Generated ECG signals

#### `load_checkpoint(checkpoint_path)`

Load model weights from a checkpoint file.

#### `save_checkpoint(checkpoint_path, optimizer=None, epoch=None)`

Save model weights to a checkpoint file.

### ECGDataset Class

PyTorch Dataset for ECG data.

**Parameters:**
- `data_path` (str or np.ndarray): Path to .npy file or numpy array
- `labels_path` (str or np.ndarray, optional): Path to labels or numpy array
- `transform` (callable, optional): Transform for ECG signals
- `target_transform` (callable, optional): Transform for labels
- `lead_indices` (list, optional): Select specific leads
- `segment_length` (int, optional): ECG segment length

### PTBXLDataset Class

Specialized dataset for PTB-XL data (8-lead ECG, 1000 samples).

**Parameters:**
- `data_path` (str): Path to PTB-XL data file
- `labels_path` (str): Path to PTB-XL labels file
- `use_8_leads` (bool): If True, select 8 leads. Default: True

### create_dataloaders Function

Convenience function to create train and validation dataloaders.

```python
from sssd_standalone import create_dataloaders

train_loader, val_loader = create_dataloaders(
    train_data_path="train_data.npy",
    train_labels_path="train_labels.npy",
    val_data_path="val_data.npy",
    val_labels_path="val_labels.npy",
    batch_size=8,
    num_workers=4,
    segment_length=1000
)
```

## Configuration

Edit `config/config_SSSD_ECG.json` to customize model parameters:

```json
{
  "diffusion_config": {
    "T": 200,              // Number of diffusion steps
    "beta_0": 0.0001,      // Noise schedule start
    "beta_T": 0.02         // Noise schedule end
  },
  "wavenet_config": {
    "in_channels": 8,      // Input channels
    "out_channels": 8,     // Output channels
    "num_res_layers": 36,  // Number of residual blocks
    "res_channels": 256,   // Residual channels
    // ... other model parameters
  },
  "train_config": {
    "learning_rate": 2e-4,
    "batch_size": 8,
    "n_iters": 100000
  }
}
```

## Examples

See the `examples/` directory for detailed usage examples:

- `train_new.py`: Full training script with command-line arguments
- `example_usage.py`: Various usage examples including training and generation

Run the training script:

```bash
cd examples
python train_new.py --config ../config/config_SSSD_ECG.json \
                    --data_path /path/to/data.npy \
                    --labels_path /path/to/labels.npy \
                    --batch_size 8 \
                    --lr 2e-4 \
                    --n_iters 100000
```

## Data Format

### ECG Data
- Format: NumPy array (.npy)
- Shape: `(num_samples, num_channels, signal_length)`
- Example: `(10000, 8, 1000)` for 10000 samples, 8 leads, 1000 time steps

### Labels
- Format: NumPy array (.npy)
- Shape: `(num_samples,)` for single-label or `(num_samples, num_classes)` for multi-label
- Example: `(10000,)` for single-label or `(10000, 71)` for 71-class multi-label

## Tips

### Memory Optimization
- Adjust `batch_size` based on GPU memory
- Use `num_workers` in DataLoader for parallel data loading
- Enable `pin_memory=True` for faster GPU transfer

### Training Stability
- Use learning rate schedulers
- Consider gradient clipping
- Monitor loss and validation metrics

### Multi-GPU Training
- Wrap model with `torch.nn.DataParallel` or `DistributedDataParallel`

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
train_loader, _ = create_dataloaders(..., batch_size=4)

# Or use gradient accumulation
accumulation_steps = 4
for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Data Shape Errors
Ensure your data has the correct shape:
- ECG data: `(num_samples, channels, length)`
- Labels: `(num_samples,)` or `(num_samples, num_classes)`

## Citation

If you use this code in your research, please cite:

```bibtex
@article{sssd_ecg,
  title={SSSD-ECG: Structured State Space Diffusion Model for ECG Generation},
  author={...},
  journal={...},
  year={2024}
}
```

## License

[Specify your license here]

## Acknowledgments

This implementation is based on the SSSD (Structured State Space Diffusion) model and adapted for ECG generation tasks.
