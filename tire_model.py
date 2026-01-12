"""
Tire longitudinal force model with saturation.
Implements Fx(kappa, Fz, params) using a saturated curve.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class TireParams:
    """Tire model parameters."""
    mu: float = 1.0  # Peak friction coefficient
    Ck: float = 50000.0  # Longitudinal stiffness (N per unit slip)


def compute_fx(kappa: np.ndarray, Fz: float, params: TireParams) -> np.ndarray:
    """
    Compute longitudinal tire force Fx using a saturated model.
    
    Uses a smooth saturation function that preserves linearity for small kappa
    and saturates at mu*Fz for large kappa. Fx has the same sign as kappa.
    
    Model: Fx = sign(kappa) * Fx_max * tanh(Ck*kappa/Fx_max)
    For small kappa: Fx ≈ Ck*kappa (linear)
    For large kappa: Fx → ±Fx_max (saturated)
    
    Args:
        kappa: Longitudinal slip ratio (dimensionless, can be array)
        Fz: Normal load (N)
        params: Tire model parameters
        
    Returns:
        Fx: Longitudinal tire force (N), same shape as kappa
    """
    # Maximum force (saturation level)
    Fx_max = params.mu * Fz
    
    # Compute normalized slip
    kappa_norm = params.Ck * kappa / Fx_max
    
    # Use tanh for smooth saturation: Fx = sign(kappa) * Fx_max * tanh(|kappa_norm|)
    # This preserves linearity: tanh(x) ≈ x for small x
    Fx = np.sign(kappa) * Fx_max * np.tanh(np.abs(kappa_norm))
    
    # Handle kappa = 0 case explicitly
    Fx = np.where(kappa == 0, 0.0, Fx)
    
    return Fx


def validate_fx_model(kappa: np.ndarray, Fx: np.ndarray, Fz: float, params: TireParams):
    """
    Sanity checks for the Fx model.
    
    Raises AssertionError if checks fail.
    """
    # Check: Fx should be ~0 when kappa=0
    zero_slip_idx = np.where(np.abs(kappa) < 1e-6)[0]
    if len(zero_slip_idx) > 0:
        assert np.all(np.abs(Fx[zero_slip_idx]) < 1e-3), \
            f"Fx should be ~0 when kappa=0, but got {Fx[zero_slip_idx]}"
    
    # Check: Fx should saturate near +/- mu*Fz at large |kappa|
    Fx_max_expected = params.mu * Fz
    large_slip_idx = np.where(np.abs(kappa) > 0.2)[0]
    if len(large_slip_idx) > 0:
        Fx_large = Fx[large_slip_idx]
        saturation_ratio = np.abs(Fx_large) / Fx_max_expected
        assert np.all(saturation_ratio > 0.8), \
            f"Fx should saturate near ±{Fx_max_expected}N at large |kappa|, " \
            f"but got {np.max(np.abs(Fx_large))}N"
    
    # Check: For small |kappa|, Fx should be approximately linear in kappa
    # Use a smaller range to ensure we're in the truly linear region
    small_slip_idx = np.where((np.abs(kappa) > 0.0001) & (np.abs(kappa) < 0.005))[0]
    if len(small_slip_idx) > 0:
        kappa_small = kappa[small_slip_idx]
        Fx_small = Fx[small_slip_idx]
        # Linear approximation: Fx ≈ Ck * kappa
        Fx_expected_linear = params.Ck * kappa_small
        linear_error = np.abs(Fx_small - Fx_expected_linear) / (np.abs(Fx_expected_linear) + 1e-10)
        assert np.all(linear_error < 0.15), \
            f"For small |kappa|, Fx should be approximately linear, " \
            f"but relative error is {np.max(linear_error):.2%}"
    
    # Check: Fx should have the same sign as kappa
    assert np.all(np.sign(Fx) == np.sign(kappa)) or np.all(kappa == 0), \
        "Fx should have the same sign as kappa"

