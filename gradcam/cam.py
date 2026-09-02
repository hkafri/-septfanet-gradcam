"""
PyTorch implementation of Grad-CAM for Sep-TFAnet-VAD.

Grad-CAM (Gradient-weighted Class Activation Mapping) computes attention maps
by backpropagating gradients through a target layer and computing weighted
combinations of the activations.

For audio spectrograms, the CAM highlights which time-frequency regions
contributed most to a decision (e.g., "why did the network think this
region belonged to speaker 1?").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, Dict, Any


class GradCAMHook:
    """
    Registers hooks on a model layer to capture activations and gradients
    for Grad-CAM computation.
    """
    
    def __init__(self, model: nn.Module, target_layer_name: str):
        """
        Args:
            model: The PyTorch model to hook
            target_layer_name: Name/path to the target layer (e.g., 'TCN.TCN.23')
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.target_module = self._get_module_by_name(model, target_layer_name)
        
        if self.target_module is None:
            raise ValueError(f"Layer '{target_layer_name}' not found in model")
        
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        
        self._register_hooks()
    
    def _get_module_by_name(self, model: nn.Module, module_name: str) -> Optional[nn.Module]:
        """Get a module by its name (e.g., 'layer1.0.conv')"""
        parts = module_name.split('.')
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part, None)
                if module is None:
                    return None
        return module
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        # Forward hook to save activations
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # Backward hook to save gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        handle1 = self.target_module.register_forward_hook(forward_hook)
        handle2 = self.target_module.register_full_backward_hook(backward_hook)
        
        self.hook_handles = [handle1, handle2]
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
    
    def __del__(self):
        """Clean up hooks on deletion"""
        self.remove_hooks()


class GradCAM:
    """
    Grad-CAM implementation for Sep-TFAnet-VAD.
    
    Computes activation maps for a given target (e.g., VAD probability,
    separation mask value) by computing weighted combinations of layer
    activations using backpropagated gradients.
    """
    
    def __init__(self, model: nn.Module, target_layer_name: str, device: str = 'cpu'):
        """
        Args:
            model: SeparationModel from Sep-TFAnet-VAD
            target_layer_name: Name of the layer to visualize (e.g., 'TCN.TCN.23')
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.hook = GradCAMHook(model, target_layer_name)
    
    def compute_cam(self, input_audio: torch.Tensor, target_scalar: torch.Tensor,
                    inference_kw: Dict[str, Any] = None) -> np.ndarray:
        """
        Compute Grad-CAM for a given target scalar.
        
        Args:
            input_audio: Input waveform (B, T), should have requires_grad=True
            target_scalar: Scalar to backprop through (e.g., VAD prob at frame t)
            inference_kw: Inference kwargs dict (passed to model forward)
        
        Returns:
            CAM: Activation map (freq, time) normalized to [0, 1]
        """
        if inference_kw is None:
            inference_kw = {}
        
        # Ensure input has gradients enabled
        if not input_audio.requires_grad:
            input_audio = input_audio.clone().detach().requires_grad_(True)
        
        input_audio = input_audio.to(self.device).float()
        
        # Backward pass to fill gradients
        try:
            target_scalar.backward(retain_graph=True)
        except RuntimeError as e:
            print(f"Warning: backward failed: {e}")
        
        # Compute Grad-CAM
        activations = self.hook.activations  # (B, C, F, T) or (B, C, T)
        gradients = self.hook.gradients       # same shape
        
        if activations is None or gradients is None:
            raise RuntimeError("Failed to capture activations or gradients")
        
        # Compute weights: global average pool gradients over spatial/temporal dims
        # Keep channel dimension
        weights = gradients.mean(dim=tuple(range(2, gradients.ndim)), keepdim=True)  # (B, C, 1, ..., 1)
        
        # Weighted activation sum
        cam = (weights * activations).sum(dim=1, keepdim=False)  # (B, F, T) or (B, T)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam.cpu().numpy()
    
    def __del__(self):
        """Clean up hooks"""
        if hasattr(self, 'hook'):
            self.hook.remove_hooks()


def compute_gradcam_for_vad(model: nn.Module, input_audio: torch.Tensor,
                            target_layer_name: str, vad_frame_idx: int,
                            speaker_idx: int = 0, device: str = 'cpu',
                            inference_kw: Dict[str, Any] = None) -> np.ndarray:
    """
    Compute Grad-CAM for a specific VAD frame probability.
    
    Args:
        model: SeparationModel
        input_audio: Input waveform (B, T)
        target_layer_name: Layer to visualize
        vad_frame_idx: Which time frame to explain
        speaker_idx: Which speaker (0 or 1)
        device: Device to run on
        inference_kw: Inference kwargs
    
    Returns:
        CAM: Activation map (T,) or (F, T) depending on target layer
    """
    if inference_kw is None:
        inference_kw = {}
    
    model.eval()
    model.to(device)
    input_audio = input_audio.to(device).float()
    input_audio.requires_grad_(True)
    
    # Get VAD output
    with torch.enable_grad():
        _, output_vad, _ = model(input_audio, inference_kw)
    
    # Target: VAD probability at specific frame and speaker
    if output_vad is not None and torch.is_tensor(output_vad):
        target_vad = output_vad[0, speaker_idx, vad_frame_idx]
    else:
        raise ValueError("Model did not produce VAD output")
    
    # Compute Grad-CAM
    gradcam = GradCAM(model, target_layer_name, device=device)
    cam = gradcam.compute_cam(input_audio, target_vad, inference_kw)
    
    return cam.squeeze()  # Remove batch dimension


def compute_gradcam_for_separation(model: nn.Module, input_audio: torch.Tensor,
                                   target_layer_name: str, freq_bin: int,
                                   time_frame: int, speaker_idx: int = 0,
                                   device: str = 'cpu',
                                   inference_kw: Dict[str, Any] = None) -> np.ndarray:
    """
    Compute Grad-CAM for a specific mask value (separation target).
    
    Args:
        model: SeparationModel
        input_audio: Input waveform (B, T)
        target_layer_name: Layer to visualize
        freq_bin: Which frequency bin (0 to 256)
        time_frame: Which time frame
        speaker_idx: Which speaker (0 or 1)
        device: Device to run on
        inference_kw: Inference kwargs
    
    Returns:
        CAM: Activation map
    """
    if inference_kw is None:
        inference_kw = {}
    
    model.eval()
    model.to(device)
    input_audio = input_audio.to(device).float()
    input_audio.requires_grad_(True)
    
    # Get mask output
    with torch.enable_grad():
        _, _, _ = model(input_audio, inference_kw)
    
    # Target: mask value at specific bin and frame
    if hasattr(model, 'mask_per_speaker'):
        target_mask = model.mask_per_speaker[0, speaker_idx, freq_bin, time_frame]
    else:
        raise ValueError("Model did not produce masks")
    
    # Compute Grad-CAM
    gradcam = GradCAM(model, target_layer_name, device=device)
    cam = gradcam.compute_cam(input_audio, target_mask, inference_kw)
    
    return cam.squeeze()
