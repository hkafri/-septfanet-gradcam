"""
Visualization utilities for Grad-CAM on spectrograms.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Optional, Tuple


def plot_spectrogram_with_cam(spec: np.ndarray, cam: np.ndarray, 
                              title: str = "Spectrogram with Grad-CAM",
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (14, 5)) -> None:
    """
    Plot magnitude spectrogram with overlaid Grad-CAM heatmap.
    
    Args:
        spec: Magnitude spectrogram (freq, time)
        cam: Grad-CAM activation map (freq, time) or (time,)
        title: Plot title
        save_path: Path to save figure (if None, just display)
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Normalize spectrogram for display
    spec_normalized = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    
    # Left: Original spectrogram
    ax = axes[0]
    im0 = ax.imshow(spec_normalized, aspect='auto', origin='lower', cmap='gray')
    ax.set_title('Input Spectrogram')
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Frequency Bins')
    plt.colorbar(im0, ax=ax, label='dB')
    
    # Right: Spectrogram with CAM overlay
    ax = axes[1]
    
    # Handle different CAM shapes
    if cam.ndim == 1:
        # Time-only CAM (e.g., VAD branch output)
        cam_2d = np.tile(cam, (spec.shape[0], 1))
    else:
        # Freq-time CAM (e.g., TCN layer)
        cam_2d = cam
    
    # Ensure CAM has same shape as spec
    if cam_2d.shape != spec.shape:
        # Interpolate if needed
        from scipy.interpolate import interp2d
        x_old = np.arange(cam_2d.shape[1])
        y_old = np.arange(cam_2d.shape[0])
        x_new = np.linspace(0, cam_2d.shape[1]-1, spec.shape[1])
        y_new = np.linspace(0, cam_2d.shape[0]-1, spec.shape[0])
        f = interp2d(x_old, y_old, cam_2d, kind='linear')
        cam_2d = f(x_new, y_new)
    
    # Display spectrogram
    im1 = ax.imshow(spec_normalized, aspect='auto', origin='lower', cmap='gray', alpha=1.0)
    
    # Overlay CAM heatmap
    cmap_cam = cm.get_cmap('inferno')
    im2 = ax.imshow(cam_2d, aspect='auto', origin='lower', cmap=cmap_cam, alpha=0.6)
    
    ax.set_title(f'{title} (Heatmap Overlay)')
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Frequency Bins')
    plt.colorbar(im2, ax=ax, label='CAM Weight')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved figure to {save_path}")
    
    plt.show()


def plot_vad_with_importance(vad_output: np.ndarray, cam: np.ndarray,
                             title: str = "VAD with Grad-CAM Importance",
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Plot VAD probabilities with Grad-CAM importance curve overlay.
    
    Args:
        vad_output: VAD probabilities (time,) in [0, 1]
        cam: Grad-CAM importance weights (time,) or (freq, time)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    time_frames = np.arange(len(vad_output))
    
    # Handle different CAM shapes
    if cam.ndim > 1:
        # Aggregate freq dimension
        cam_1d = cam.mean(axis=0)
    else:
        cam_1d = cam
    
    # Normalize CAM
    cam_1d = (cam_1d - cam_1d.min()) / (cam_1d.max() - cam_1d.min() + 1e-8)
    
    # Plot VAD
    ax.plot(time_frames, vad_output, 'b-', linewidth=2, label='VAD Probability', alpha=0.7)
    ax.fill_between(time_frames, 0, vad_output, alpha=0.2, color='blue')
    
    # Plot importance curve
    ax.plot(time_frames, cam_1d, 'r-', linewidth=2, label='Grad-CAM Importance', alpha=0.7)
    ax.fill_between(time_frames, 0, cam_1d, alpha=0.2, color='red')
    
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Probability / Importance')
    ax.set_ylim([0, 1])
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved figure to {save_path}")
    
    plt.show()


def plot_comparison_grid(specs: dict, cams: dict, titles: dict,
                        save_path: Optional[str] = None,
                        figsize: Optional[Tuple[int, int]] = None) -> None:
    """
    Plot a grid of spectrograms with CAM overlays for comparison.
    
    Args:
        specs: Dict of spectrogram arrays {name: spec_array}
        cams: Dict of CAM arrays {name: cam_array}
        titles: Dict of titles {name: title_str}
        save_path: Path to save figure
        figsize: Figure size (auto-computed if None)
    """
    n_plots = len(specs)
    if figsize is None:
        figsize = (5 * n_plots, 4)
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    for i, (name, spec) in enumerate(specs.items()):
        ax = axes[i]
        cam = cams.get(name, np.zeros_like(spec))
        title = titles.get(name, name)
        
        # Normalize
        spec_normalized = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) if cam.max() > cam.min() else cam
        
        # Display
        ax.imshow(spec_normalized, aspect='auto', origin='lower', cmap='gray', alpha=1.0)
        cmap_cam = cm.get_cmap('inferno')
        ax.imshow(cam_normalized, aspect='auto', origin='lower', cmap=cmap_cam, alpha=0.6)
        
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved comparison grid to {save_path}")
    
    plt.show()


def get_peaks_in_cam(cam: np.ndarray, num_peaks: int = 3) -> list:
    """
    Find top-k peaks (time-frequency regions) in CAM.
    
    Args:
        cam: Grad-CAM array (freq, time) or (time,)
        num_peaks: Number of top peaks to return
    
    Returns:
        List of tuples (time_frame, freq_bin, importance_value)
    """
    if cam.ndim == 1:
        # Only time dimension
        indices = np.argsort(cam)[-num_peaks:][::-1]
        return [(idx, 0, cam[idx]) for idx in indices]
    else:
        # Freq and time dimensions
        flat_indices = np.argsort(cam.flatten())[-num_peaks:][::-1]
        peaks = []
        for idx in flat_indices:
            freq_bin = idx // cam.shape[1]
            time_frame = idx % cam.shape[1]
            value = cam[freq_bin, time_frame]
            peaks.append((time_frame, freq_bin, float(value)))
        return peaks
