from dataclasses import dataclass, field
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal

from sklearn.gaussian_process import GaussianProcessRegressor

from gp_model.kernels import (
    RBFKernelParams,
    PeriodicKernelParams,
    NoiseKernelParams,
    build_rbf_noise_kernel,
    build_rbf_periodic_noise_kernel,
)

KernelType = Literal["rbf", "rbf_periodic"]


@dataclass
class GasForwardGPConfig:
    """
    GP config.

    kernel_type:
      - "rbf"         : Step 2 model (RBF + noise)
      - "rbf_periodic": Step 3 model (RBF + periodic + noise)
    """
    kernel_type: KernelType = "rbf"

    # GP settings
    n_restarts_optimizer: int = 5
    normalize_y: bool = True
    random_state: int = 42

    # Kernel params (USE default_factory!)
    rbf_params: RBFKernelParams = field(default_factory=RBFKernelParams)
    periodic_params: PeriodicKernelParams = field(default_factory=PeriodicKernelParams)
    noise_params: NoiseKernelParams = field(default_factory=NoiseKernelParams)



class GasForwardGP:
    """Small wrapper around sklearn GaussianProcessRegressor."""

    def __init__(self, config: Optional[GasForwardGPConfig] = None):
        self.config = config or GasForwardGPConfig()

        if self.config.kernel_type == "rbf":
            kernel = build_rbf_noise_kernel(self.config.rbf_params, self.config.noise_params)
        elif self.config.kernel_type == "rbf_periodic":
            kernel = build_rbf_periodic_noise_kernel(
                self.config.rbf_params,
                self.config.periodic_params,
                self.config.noise_params,
            )
        else:
            raise ValueError(f"Unknown kernel_type: {self.config.kernel_type}")

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,  # noise handled by WhiteKernel
            n_restarts_optimizer=self.config.n_restarts_optimizer,
            normalize_y=self.config.normalize_y,
            random_state=self.config.random_state,
        )

    def fit(self, ttm_years: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(ttm_years).reshape(-1, 1)
        y = np.asarray(y)
        self.gp.fit(X, y)

    def predict(self, ttm_years: np.ndarray, return_std: bool = True):
        X = np.asarray(ttm_years).reshape(-1, 1)
        if return_std:
            mean, std = self.gp.predict(X, return_std=True)
            return mean, std
        mean = self.gp.predict(X, return_std=False)
        return mean, None

    @property
    def kernel_(self):
        return self.gp.kernel_
