"""Physics-based augmentation for EBSD pattern embedding training.

Applies realistic degradations that mimic experimental conditions:
Poisson noise, Gaussian noise, background gradients, blur, contrast
variation, and partial occlusion.  Each augmentation is applied
independently with probability ``p``.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


class PhysicsAugmentation:
    """Physics-based augmentation pipeline for EBSD patterns.

    Parameters
    ----------
    p : float
        Probability of applying each individual augmentation.
        Set to 0.0 to disable all augmentations (passthrough).
    poisson_gain_range : tuple
        (min, max) gain for Poisson noise simulation.
    gaussian_sigma_range : tuple
        (min, max) sigma for additive Gaussian noise.
    gradient_strength : float
        Maximum strength of linear background gradient.
    blur_sigma_max : float
        Maximum sigma for Gaussian blur (strain simulation).
    contrast_alpha_range : tuple
        (min, max) multiplicative contrast factor.
    contrast_beta_range : tuple
        (min, max) additive brightness shift.
    occlusion_frac_range : tuple
        (min, max) fraction of pixels to occlude.
    """

    def __init__(
        self,
        p: float = 0.5,
        poisson_gain_range: tuple[float, float] = (50.0, 500.0),
        gaussian_sigma_range: tuple[float, float] = (0.01, 0.1),
        gradient_strength: float = 0.3,
        blur_sigma_max: float = 2.0,
        contrast_alpha_range: tuple[float, float] = (0.7, 1.3),
        contrast_beta_range: tuple[float, float] = (-0.1, 0.1),
        occlusion_frac_range: tuple[float, float] = (0.05, 0.15),
        anisotropic_blur: bool = False,
        aniso_sigma_range: tuple[float, float] = (0.5, 3.0),
        aniso_angle_range: tuple[float, float] = (0.0, 180.0),
    ) -> None:
        self.p = p
        self.poisson_gain_range = poisson_gain_range
        self.gaussian_sigma_range = gaussian_sigma_range
        self.gradient_strength = gradient_strength
        self.blur_sigma_max = blur_sigma_max
        self.contrast_alpha_range = contrast_alpha_range
        self.contrast_beta_range = contrast_beta_range
        self.occlusion_frac_range = occlusion_frac_range
        self.anisotropic_blur = anisotropic_blur
        self.aniso_sigma_range = aniso_sigma_range
        self.aniso_angle_range = aniso_angle_range

    def __call__(self, pattern: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a pattern.

        Parameters
        ----------
        pattern : np.ndarray
            2-D float32 pattern, ideally in [0, 1] range.

        Returns
        -------
        np.ndarray
            Augmented pattern, same shape, float32.
        """
        out = pattern.copy().astype(np.float32)

        if self.p <= 0.0:
            return out

        # 1. Poisson noise (camera statistics)
        if np.random.random() < self.p:
            gain = np.random.uniform(*self.poisson_gain_range)
            noisy = np.random.poisson(np.clip(out, 0, None) * gain) / gain
            out = noisy.astype(np.float32)

        # 2. Gaussian noise (electronics)
        if np.random.random() < self.p:
            sigma = np.random.uniform(*self.gaussian_sigma_range)
            out = out + np.random.normal(0, sigma, out.shape).astype(np.float32)

        # 3. Background gradient (linear ramp)
        if np.random.random() < self.p:
            h, w = out.shape
            a = np.random.uniform(-self.gradient_strength, self.gradient_strength)
            b = np.random.uniform(-self.gradient_strength, self.gradient_strength)
            y_ramp = np.linspace(-1, 1, h).reshape(-1, 1)
            x_ramp = np.linspace(-1, 1, w).reshape(1, -1)
            gradient = (a * y_ramp + b * x_ramp).astype(np.float32)
            out = out + gradient

        # 4. Gaussian blur (strain simulation)
        if np.random.random() < self.p:
            sigma = np.random.uniform(0, self.blur_sigma_max)
            if sigma > 0.1:
                out = gaussian_filter(out, sigma=sigma).astype(np.float32)

        # 5. Contrast variation
        if np.random.random() < self.p:
            alpha = np.random.uniform(*self.contrast_alpha_range)
            beta = np.random.uniform(*self.contrast_beta_range)
            out = alpha * out + beta

        # 6. Partial occlusion (dead pixels)
        if np.random.random() < self.p:
            frac = np.random.uniform(*self.occlusion_frac_range)
            mask = np.random.random(out.shape) < frac
            out[mask] = 0.0

        # 7. Anisotropic blur (strain simulation)
        if self.anisotropic_blur and np.random.random() < self.p:
            from scipy.ndimage import rotate
            sigma = np.random.uniform(*self.aniso_sigma_range)
            angle = np.random.uniform(*self.aniso_angle_range)
            # Apply directional blur: rotate, blur along x, rotate back
            rotated = rotate(out, angle, reshape=False, order=1)
            blurred = gaussian_filter(rotated, sigma=[0, sigma])
            out = rotate(blurred, -angle, reshape=False, order=1).astype(np.float32)

        # 8. Final normalization: zero-mean, unit-variance
        std = out.std()
        if std > 1e-8:
            out = (out - out.mean()) / std

        return out.astype(np.float32)
