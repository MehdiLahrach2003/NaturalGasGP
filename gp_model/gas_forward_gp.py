import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


@dataclass
class GasForwardGPConfig:
    """
    Configuration for the Gaussian Process model.

    Step 2: we use
    - RBF kernel (smooth term structure)
    - WhiteKernel (observation noise)
    """
    length_scale: float = 0.5
    length_scale_bounds: Tuple[float, float] = (1e-2, 10.0)
    noise_level: float = 0.05
    noise_level_bounds: Tuple[float, float] = (1e-5, 1e1)
    n_restarts_optimizer: int = 5
    normalize_y: bool = True
    random_state: int = 42


class GasForwardGP:
    """
    Wrapper around sklearn GaussianProcessRegressor for gas forward curves.

    Input X is always time-to-maturity in years, shaped as (n_samples, 1).
    Output y is the forward price.
    """

    def __init__(self, config: Optional[GasForwardGPConfig] = None):
        if config is None:
            config = GasForwardGPConfig()
        self.config = config

        # RBF kernel (smooth) + WhiteKernel (noise)
        rbf_kernel = RBF(
            length_scale=config.length_scale,
            length_scale_bounds=config.length_scale_bounds,
        )
        noise_kernel = WhiteKernel(
            noise_level=config.noise_level,
            noise_level_bounds=config.noise_level_bounds,
        )

        kernel = rbf_kernel + noise_kernel

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,  # noise handled by WhiteKernel
            n_restarts_optimizer=config.n_restarts_optimizer,
            normalize_y=config.normalize_y,
            random_state=config.random_state,
        )

    def fit(self, ttm_years: np.ndarray, forward_prices: np.ndarray) -> None:
        """
        Fit the GP model to observed forward prices.

        Parameters
        ----------
        ttm_years : np.ndarray
            Time to maturity in years (1D array).
        forward_prices : np.ndarray
            Observed forward prices (1D array).
        """
        X = np.asarray(ttm_years).reshape(-1, 1)
        y = np.asarray(forward_prices)
        self.gp.fit(X, y)

    def predict(
        self,
        ttm_years: np.ndarray,
        return_std: bool = True,
    ):
        """
        Predict the forward curve at a grid of maturities.

        Parameters
        ----------
        ttm_years : np.ndarray
            Time to maturity in years (1D array).
        return_std : bool
            If True, also returns the standard deviation of the GP posterior.

        Returns
        -------
        mean : np.ndarray
        std : np.ndarray or None
        """
        X = np.asarray(ttm_years).reshape(-1, 1)
        if return_std:
            mean, std = self.gp.predict(X, return_std=True)
            return mean, std
        else:
            mean = self.gp.predict(X, return_std=False)
            return mean, None

    @property
    def kernel_(self):
        """Return the fitted kernel after hyperparameter optimization."""
        return self.gp.kernel_
