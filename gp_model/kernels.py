from dataclasses import dataclass
from typing import Tuple

from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel


@dataclass
class RBFKernelParams:
    """RBF kernel hyperparameters (smoothness)."""
    length_scale: float = 0.6
    length_scale_bounds: Tuple[float, float] = (1e-2, 10.0)


@dataclass
class PeriodicKernelParams:
    """Periodic kernel hyperparameters (seasonality)."""
    periodicity: float = 1.0  # 1 year seasonality
    periodicity_bounds: Tuple[float, float] = (0.5, 2.0)
    length_scale: float = 0.6
    length_scale_bounds: Tuple[float, float] = (1e-2, 10.0)


@dataclass
class NoiseKernelParams:
    """White noise kernel hyperparameters (observation noise)."""
    noise_level: float = 0.05
    noise_level_bounds: Tuple[float, float] = (1e-5, 1e1)


def build_rbf_noise_kernel(
    rbf: RBFKernelParams | None = None,
    noise: NoiseKernelParams | None = None,
):
    """Kernel = RBF + WhiteKernel."""
    rbf = rbf or RBFKernelParams()
    noise = noise or NoiseKernelParams()

    k_rbf = RBF(length_scale=rbf.length_scale, length_scale_bounds=rbf.length_scale_bounds)
    k_noise = WhiteKernel(noise_level=noise.noise_level, noise_level_bounds=noise.noise_level_bounds)
    return k_rbf + k_noise


def build_rbf_periodic_noise_kernel(
    rbf: RBFKernelParams | None = None,
    periodic: PeriodicKernelParams | None = None,
    noise: NoiseKernelParams | None = None,
):
    """Kernel = RBF + ExpSineSquared (periodic) + WhiteKernel."""
    rbf = rbf or RBFKernelParams()
    periodic = periodic or PeriodicKernelParams()
    noise = noise or NoiseKernelParams()

    k_rbf = RBF(length_scale=rbf.length_scale, length_scale_bounds=rbf.length_scale_bounds)

    k_periodic = ExpSineSquared(
        length_scale=periodic.length_scale,
        periodicity=periodic.periodicity,
        length_scale_bounds=periodic.length_scale_bounds,
        periodicity_bounds=periodic.periodicity_bounds,
    )

    k_noise = WhiteKernel(noise_level=noise.noise_level, noise_level_bounds=noise.noise_level_bounds)

    return k_rbf + k_periodic + k_noise
