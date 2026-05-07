"""
SSSD-ECG Model Wrapper
Provides a clean interface for training and inference with the SSSD-ECG diffusion model.

Usage:
    # Training
    model = SSSDECG(config_path="config/config_SSSD_ECG.json")
    loss = model(x, y)

    # Inference
    pred = model.generate(labels=y, num_samples=10)
"""

import os
import json
import torch
import torch.nn as nn
from models.SSSD_ECG import SSSD_ECG as SSSD_ECG_Base
from utils.util import calc_diffusion_hyperparams, training_loss_label, sampling_label


class SSSDECG(nn.Module):
    """
    Wrapper class for SSSD-ECG diffusion model with clean interface.

    This class handles:
    - Loading configuration
    - Initializing diffusion hyperparameters
    - Training loss computation
    - Sample generation

    Args:
        config_path (str): Path to configuration JSON file. Default: "config/config_SSSD_ECG.json"
        device (str): Device to run the model on. Default: "cuda" if available, else "cpu"
    """

    def __init__(self, config_path=None, device=None):
        super(SSSDECG, self).__init__()

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load configuration
        if config_path is None:
            # Default config path relative to this file
            config_path = os.path.join(
                os.path.dirname(__file__),
                "config",
                "config_SSSD_ECG.json"
            )

        with open(config_path) as f:
            self.config = json.load(f)

        # Extract configs
        self.diffusion_config = self.config["diffusion_config"]
        self.model_config = self.config["wavenet_config"]

        # Calculate diffusion hyperparameters
        self.diffusion_hyperparams = calc_diffusion_hyperparams(
            T=self.diffusion_config["T"],
            beta_0=self.diffusion_config["beta_0"],
            beta_T=self.diffusion_config["beta_T"]
        )

        # Move diffusion hyperparameters to device
        for key in self.diffusion_hyperparams:
            if key != "T":
                self.diffusion_hyperparams[key] = self.diffusion_hyperparams[key].to(self.device)

        # Initialize the base model
        self.model = SSSD_ECG_Base(**self.model_config).to(self.device)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # For tracking training mode
        self._is_training = True

        # Cache for JIT compiled model
        self._jit_model = None
        self._jit_available = self._check_jit_availability()

    def forward(self, x, y):
        """
        Compute training loss for a batch of data.

        Args:
            x (torch.Tensor): Input ECG signals, shape=(batch_size, channels, length)
            y (torch.Tensor): Labels, shape=(batch_size, num_classes) for multi-label
                             or (batch_size,) for single-label (will be converted to one-hot)

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Move to device
        x = x.to(self.device)
        y = y.to(self.device)

        # Handle label format - convert to float if needed
        if y.dtype != torch.float32:
            y = y.float()

        # If y is 1D, convert to one-hot encoding
        if len(y.shape) == 1:
            num_classes = self.model_config.get("label_embed_classes", 71)
            y_onehot = torch.zeros(y.size(0), num_classes, device=self.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1)
            y = y_onehot

        # Compute loss
        X = (x, y)
        loss = training_loss_label(
            self.model,
            self.loss_fn,
            X,
            self.diffusion_hyperparams,
            device=self.device
        )

        return loss

    def generate(self, labels=None, num_samples=1, return_numpy=False):
        """
        Generate ECG samples using the diffusion model.

        Args:
            labels (torch.Tensor, optional): Conditioning labels
                - shape=(num_samples, num_classes) for multi-label
                - shape=(num_samples,) for class indices (will be converted to one-hot)
                - If None, random labels will be generated
            num_samples (int): Number of samples to generate. Default: 1
                              Only used if labels is None
            return_numpy (bool): If True, return numpy array instead of torch tensor

        Returns:
            torch.Tensor or np.ndarray: Generated ECG signals
                shape=(num_samples, channels, length)
        """
        # Ensure model is in eval mode
        was_training = self.model.training
        self.model.eval()

        # Handle labels
        if labels is None:
            # Generate random labels
            num_classes = self.model_config.get("label_embed_classes", 71)
            labels = torch.randint(0, num_classes, (num_samples,), device=self.device)
        else:
            labels = labels.to(self.device)
            num_samples = labels.size(0)

        # If labels is 1D, convert class indices to one-hot.
        if len(labels.shape) == 1:
            labels = labels.long()
            num_classes = self.model_config.get("label_embed_classes", 71)
            labels_onehot = torch.zeros(num_samples, num_classes, device=self.device)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels = labels_onehot
        else:
            labels = labels.float()

        # Determine output size
        channels = self.model_config["out_channels"]
        length = self.config.get("trainset_config", {}).get("segment_length", 1000)
        size = (num_samples, channels, length)

        # Generate samples
        with torch.inference_mode():
            samples = sampling_label(
                self.model,
                size,
                self.diffusion_hyperparams,
                cond=labels,
                device=self.device
            )

        # Restore training mode
        if was_training:
            self.model.train()

        # Convert to numpy if requested
        if return_numpy:
            return samples.cpu().numpy()

        return samples

    def generate_amp(
        self,
        labels=None,
        num_samples=1,
        return_numpy=False,
        use_amp=True,
        amp_dtype="bf16",
    ):
        """
        Generate ECG samples using the diffusion model.

        Args:
            labels (torch.Tensor, optional): Conditioning labels
                - shape=(num_samples, num_classes) for multi-label
                - shape=(num_samples,) for class indices (will be converted to one-hot)
                - If None, random labels will be generated
            num_samples (int): Number of samples to generate. Default: 1
                            Only used if labels is None
            return_numpy (bool): If True, return numpy array instead of torch tensor
            use_amp (bool): If True, use automatic mixed precision for faster generation
            amp_dtype (str): AMP dtype, "bf16" or "fp16"

        Returns:
            torch.Tensor or np.ndarray: Generated ECG signals
                shape=(num_samples, channels, length)
        """
        # Ensure model is in eval mode
        was_training = self.model.training
        self.model.eval()

        # Handle labels
        if labels is None:
            # Generate random labels
            num_classes = self.model_config.get("label_embed_classes", 71)
            labels = torch.randint(0, num_classes, (num_samples,), device=self.device)
        else:
            labels = labels.to(self.device)
            num_samples = labels.size(0)

        # If labels is 1D, convert class indices to one-hot.
        if len(labels.shape) == 1:
            labels = labels.long()
            num_classes = self.model_config.get("label_embed_classes", 71)
            labels_onehot = torch.zeros(num_samples, num_classes, device=self.device)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels = labels_onehot
        else:
            labels = labels.float()

        # Determine output size
        channels = self.model_config["out_channels"]
        length = self.config.get("trainset_config", {}).get("segment_length", 1000)
        size = (num_samples, channels, length)

        # Generate samples with optional AMP
        with torch.inference_mode():
            if use_amp and str(self.device).startswith('cuda'):
                if amp_dtype == "bf16":
                    autocast_dtype = torch.bfloat16
                elif amp_dtype == "fp16":
                    autocast_dtype = torch.float16
                else:
                    raise ValueError(f"Unknown amp_dtype: {amp_dtype}. Choose from ['bf16', 'fp16'].")

                try:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        samples = sampling_label(
                            self.model,
                            size,
                            self.diffusion_hyperparams,
                            cond=labels,
                            device=self.device
                        )
                except RuntimeError as e:
                    err_msg = str(e)
                    # Some torch versions do not support BF16 in FFT kernels used by S4.
                    if amp_dtype == "bf16" and "Unsupported dtype BFloat16" in err_msg:
                        print(
                            "BF16 autocast is not supported for this model path on this torch build. "
                            "Retrying with FP16."
                        )
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            samples = sampling_label(
                                self.model,
                                size,
                                self.diffusion_hyperparams,
                                cond=labels,
                                device=self.device
                            )
                    else:
                        raise
            else:
                samples = sampling_label(
                    self.model,
                    size,
                    self.diffusion_hyperparams,
                    cond=labels,
                    device=self.device
                )

        # Restore training mode
        if was_training:
            self.model.train()

        # Convert to numpy if requested
        if return_numpy:
            return samples.cpu().numpy()

        return samples 

    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint .pkl file
        """
        state_dict = dict(torch.load(checkpoint_path, map_location="cpu")) # OrderedDict -> dict

        old_keys = list(state_dict.keys())
        for key in old_keys:
            new_key = key.replace("model.", "")
            # if key.find("mae.") != -1:
            #     new_key = key.replace("mae.", "backbone.")
            state_dict[new_key] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)

        print(f"Loaded checkpoint from {checkpoint_path}")

    def save_checkpoint(self, checkpoint_path, optimizer=None, epoch=None):
        """
        Save model weights to a checkpoint file.

        Args:
            checkpoint_path (str): Path to save the checkpoint
            optimizer (torch.optim.Optimizer, optional): Optimizer state to save
            epoch (int, optional): Current epoch/iteration number
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def get_model_size(self):
        """
        Get the number of parameters in the model.

        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def print_model_size(self):
        """Print model size information."""
        num_params = self.get_model_size()
        print(f"Model Parameters: {num_params / 1e6:.2f}M")

    def _check_jit_availability(self):
        """Check if PyTorch JIT compilation is available."""
        try:
            # Check for torch.compile (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                return 'compile'
            # Fall back to torch.jit.script
            elif hasattr(torch, 'jit'):
                return 'jit'
            else:
                return None
        except Exception:
            return None

    def _sampling_label_jit(self, net, size, diffusion_hyperparams, cond):
        """
        JIT-compatible version of sampling_label function.
        Performs the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

        Args:
            net: the model network
            size (tuple): size of tensor to be generated (batch_size, channels, length)
            diffusion_hyperparams (dict): dictionary of diffusion hyperparameters
            cond: conditioning labels

        Returns:
            torch.Tensor: Generated samples
        """
        _dh = diffusion_hyperparams
        T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]

        # Initialize with noise
        x = torch.randn(size, device=self.device)

        # Reverse diffusion process
        for t in range(T-1, -1, -1):
            diffusion_steps = (t * torch.ones((size[0], 1), device=self.device))

            # Predict epsilon
            epsilon_theta = net((x, cond, diffusion_steps))

            # Update x_{t-1}
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])

            # Add variance term if not at the last step
            if t > 0:
                noise = torch.randn(size, device=self.device)
                x = x + Sigma[t] * noise

        return x

    def generate_jit(self, labels=None, num_samples=1, return_numpy=False, force_recompile=False):
        """
        Generate ECG samples using JIT-compiled model for faster inference.

        This method uses PyTorch's JIT compilation (torch.compile or torch.jit) to speed up
        the generation process. The model is compiled on the first call and cached for subsequent calls.

        Args:
            labels (torch.Tensor, optional): Conditioning labels
                - shape=(num_samples, num_classes) for multi-label
                - shape=(num_samples,) for class indices (will be converted to one-hot)
                - If None, random labels will be generated
            num_samples (int): Number of samples to generate. Default: 1
                              Only used if labels is None
            return_numpy (bool): If True, return numpy array instead of torch tensor
            force_recompile (bool): If True, force recompilation of the model

        Returns:
            torch.Tensor or np.ndarray: Generated ECG signals
                shape=(num_samples, channels, length)

        Note:
            - First call will be slower due to compilation overhead
            - Subsequent calls will be significantly faster
            - If JIT is not available, falls back to regular generate()
        """
        # Check if JIT is available
        if not self._jit_available:
            print("Warning: JIT compilation not available. Falling back to regular generate().")
            return self.generate(labels=labels, num_samples=num_samples, return_numpy=return_numpy)

        # Ensure model is in eval mode
        was_training = self.model.training
        self.model.eval()

        # Compile model if not already compiled
        if self._jit_model is None or force_recompile:
            print(f"Compiling model with {self._jit_available}...")
            try:
                if self._jit_available == 'compile':
                    # Use torch.compile (PyTorch 2.0+)
                    self._jit_model = torch.compile(self.model, mode='reduce-overhead')
                elif self._jit_available == 'jit':
                    # Use torch.jit.script as fallback
                    # Note: torch.jit.script may not work with all models
                    self._jit_model = self.model  # Keep original model
                print("Model compilation completed.")
            except Exception as e:
                print(f"Warning: Model compilation failed: {e}")
                print("Falling back to regular generate().")
                self._jit_available = None
                return self.generate(labels=labels, num_samples=num_samples, return_numpy=return_numpy)

        # Handle labels
        if labels is None:
            # Generate random labels
            num_classes = self.model_config.get("label_embed_classes", 71)
            labels = torch.randint(0, num_classes, (num_samples,), device=self.device)
        else:
            labels = labels.to(self.device)
            num_samples = labels.size(0)

        # If labels is 1D, convert class indices to one-hot.
        if len(labels.shape) == 1:
            labels = labels.long()
            num_classes = self.model_config.get("label_embed_classes", 71)
            labels_onehot = torch.zeros(num_samples, num_classes, device=self.device)
            labels_onehot.scatter_(1, labels.long().unsqueeze(1), 1)
            labels = labels_onehot
        else:
            labels = labels.float()

        # Determine output size
        channels = self.model_config["out_channels"]
        length = self.config.get("trainset_config", {}).get("segment_length", 1000)
        size = (num_samples, channels, length)

        # Generate samples using JIT-compiled model
        with torch.no_grad():
            samples = self._sampling_label_jit(
                self._jit_model,
                size,
                self.diffusion_hyperparams,
                labels
            )

        # Restore training mode
        if was_training:
            self.model.train()

        # Convert to numpy if requested
        if return_numpy:
            return samples.cpu().numpy()

        return samples
