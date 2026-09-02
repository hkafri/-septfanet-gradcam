"""Grad-CAM module for Sep-TFAnet-VAD"""

from .cam import GradCAM, GradCAMHook, compute_gradcam_for_vad, compute_gradcam_for_separation
from .visualization import plot_spectrogram_with_cam, plot_vad_with_importance, plot_comparison_grid, get_peaks_in_cam

__all__ = [
    'GradCAM',
    'GradCAMHook',
    'compute_gradcam_for_vad',
    'compute_gradcam_for_separation',
    'plot_spectrogram_with_cam',
    'plot_vad_with_importance',
    'plot_comparison_grid',
    'get_peaks_in_cam',
]