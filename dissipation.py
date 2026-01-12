"""
Tire energy dissipation calculations.
Computes power dissipation and cumulative energy.
"""

import numpy as np


def compute_pdiss(Fx: np.ndarray, kappa: np.ndarray, V: float) -> np.ndarray:
    """
    Compute dissipated power: Pdiss = Fx * (kappa * V)
    
    Args:
        Fx: Longitudinal tire force (N)
        kappa: Longitudinal slip ratio (dimensionless)
        V: Forward speed at the tire (m/s)
        
    Returns:
        Pdiss: Dissipated power (W), same shape as Fx and kappa
    """
    return Fx * (kappa * V)


def compute_pdiss_dissipated(Fx: np.ndarray, kappa: np.ndarray, V: float) -> np.ndarray:
    """
    Compute absolute dissipated power (always non-negative).
    
    This represents the actual energy being dissipated as heat.
    Pdiss_dissipated = abs(Fx * (kappa * V))
    
    Args:
        Fx: Longitudinal tire force (N)
        kappa: Longitudinal slip ratio (dimensionless)
        V: Forward speed at the tire (m/s)
        
    Returns:
        Pdiss_dissipated: Non-negative dissipated power (W)
    """
    return np.abs(compute_pdiss(Fx, kappa, V))


def compute_ediss(time: np.ndarray, Pdiss: np.ndarray) -> np.ndarray:
    """
    Compute cumulative dissipated energy via numerical integration.
    
    Uses cumulative trapezoidal integration.
    
    Args:
        time: Time array (s)
        Pdiss: Dissipated power array (W)
        
    Returns:
        Ediss: Cumulative dissipated energy (J), same shape as time
    """
    if len(time) < 2:
        return np.zeros_like(time)
    
    # Use cumulative trapezoidal integration
    dt = np.diff(time)
    # Average power over each interval
    Pdiss_avg = (Pdiss[:-1] + Pdiss[1:]) / 2.0
    # Energy in each interval
    dE = Pdiss_avg * dt
    # Cumulative sum
    Ediss = np.concatenate([[0.0], np.cumsum(dE)])
    
    return Ediss


def validate_dissipation(kappa: np.ndarray, Pdiss: np.ndarray):
    """
    Sanity check: Pdiss should be ~0 when kappa=0.
    
    Raises AssertionError if check fails.
    """
    zero_slip_idx = np.where(np.abs(kappa) < 1e-6)[0]
    if len(zero_slip_idx) > 0:
        assert np.all(np.abs(Pdiss[zero_slip_idx]) < 1e-3), \
            f"Pdiss should be ~0 when kappa=0, but got {Pdiss[zero_slip_idx]}"

