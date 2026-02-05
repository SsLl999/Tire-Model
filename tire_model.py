"""
Steady-state longitudinal brush tire model.
Implements Fx(kappa, Fz, params) using a brush model with bristle deformation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TireParams:
    """
    Brush tire model parameters.
    
    Attributes:
        mu: Friction coefficient (dimensionless), peak limit
        a: Half contact patch length (meters). Total patch length L = 2*a
        kx: Bristle stiffness per unit length (Newtons per meter^2) in longitudinal shear.
            Explanation: local shear force per unit patch length = kx * q(s),
            where q(s) is bristle shear displacement (meters).
    """
    mu: float = 1.0  # Peak friction coefficient
    a: float = 0.1  # Half contact patch length (m)
    kx: float = 1e6  # Bristle stiffness per unit length (N/m^2)


def brush_model_fx(kappa: np.ndarray, Fz: float, params: TireParams) -> np.ndarray:
    """
    Compute longitudinal tire force Fx using a steady-state brush model.
    
    Model mechanics:
    - kappa (κ) = longitudinal slip ratio (dimensionless). Use sign(κ) to set driving vs braking direction.
    - Fz = normal load (Newtons).
    - V = forward speed at the tire (m/s).
    - Fx = longitudinal tire force (Newtons), with sign consistent with κ.
    
    The model computes force from bristle deformation through the contact patch:
    - Uniform line load: w = Fz / (2*a) [Newtons per meter]
    - Local friction limit: f_limit = mu * w [Newtons per meter]
    - Bristle shear displacement: q(s) = |κ| * s [meters]
    - Transition point s_star where adhesion breaks: s_star = f_limit / (kx * |κ|)
    
    Args:
        kappa: Longitudinal slip ratio (dimensionless, can be array)
        Fz: Normal load (N)
        params: Tire model parameters (mu, a, kx)
        
    Returns:
        Fx: Longitudinal tire force (N), same shape as kappa
    """
    # Convert to array if scalar
    kappa = np.asarray(kappa)
    is_scalar = kappa.ndim == 0
    kappa = np.atleast_1d(kappa)
    
    # Contact patch parameters
    L = 2.0 * params.a  # Total patch length (m)
    w = Fz / L  # Uniform line load (N/m)
    f_limit = params.mu * w  # Local friction limit per unit length (N/m)
    
    # Initialize output
    Fx = np.zeros_like(kappa)
    
    # Handle edge case: very small |kappa| to avoid division by zero
    kappa_abs = np.abs(kappa)
    small_kappa_mask = kappa_abs < 1e-6
    
    # For non-small kappa, compute brush model
    valid_mask = ~small_kappa_mask
    
    if np.any(valid_mask):
        kappa_valid = kappa_abs[valid_mask]
        kappa_valid_indices = np.where(valid_mask)[0]
        
        # Transition point where adhesion breaks
        # s_star = f_limit / (kx * |κ|)
        s_star = f_limit / (params.kx * kappa_valid)
        
        # Case 1: Full adhesion (no sliding) if s_star >= L
        full_adhesion_mask = s_star >= L
        if np.any(full_adhesion_mask):
            # Fx magnitude = 0.5 * kx * |κ| * L^2
            Fx_magnitude_full = 0.5 * params.kx * kappa_valid[full_adhesion_mask] * L**2
            full_adhesion_indices = kappa_valid_indices[full_adhesion_mask]
            Fx[full_adhesion_indices] = (
                np.sign(kappa[full_adhesion_indices]) * Fx_magnitude_full
            )
        
        # Case 2: Partial sliding if s_star < L
        partial_sliding_mask = s_star < L
        if np.any(partial_sliding_mask):
            s_star_partial = s_star[partial_sliding_mask]
            kappa_partial = kappa_valid[partial_sliding_mask]
            
            # Adhesion region contribution
            # Fx_adh = 0.5 * kx * |κ| * s_star^2
            Fx_adh = 0.5 * params.kx * kappa_partial * s_star_partial**2
            
            # Sliding region contribution
            # Fx_sld = f_limit * (L - s_star)
            Fx_sld = f_limit * (L - s_star_partial)
            
            # Total magnitude
            Fx_magnitude_partial = Fx_adh + Fx_sld
            
            partial_sliding_indices = kappa_valid_indices[partial_sliding_mask]
            Fx[partial_sliding_indices] = (
                np.sign(kappa[partial_sliding_indices]) * Fx_magnitude_partial
            )
    
    # For small kappa, Fx ≈ 0 (already initialized to zero)
    
    # Return scalar if input was scalar
    if is_scalar:
        return Fx[0]
    return Fx


def brush_model_field(
    kappa: float, Fz: float, params: TireParams, n_points: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute internal bristle deformation field over the contact patch.
    
    Returns arrays over position s along the patch:
    - s_array: Position along patch from leading edge (m), s in [0, 2*a]
    - q_array: Bristle shear displacement (m), q(s) = |κ| * s
    - f_array: Local shear force per unit length (N/m), f(s) = min(kx * q(s), f_limit)
    - adhesion_mask: Boolean array, True where in adhesion region, False where sliding
    
    Args:
        kappa: Longitudinal slip ratio (dimensionless, scalar)
        Fz: Normal load (N)
        params: Tire model parameters (mu, a, kx)
        n_points: Number of points to sample along patch length
        
    Returns:
        Tuple of (s_array, q_array, f_array, adhesion_mask)
    """
    # Contact patch parameters
    L = 2.0 * params.a  # Total patch length (m)
    w = Fz / L  # Uniform line load (N/m)
    f_limit = params.mu * w  # Local friction limit per unit length (N/m)
    
    # Position array along patch
    s_array = np.linspace(0, L, n_points)
    
    # Handle edge case: very small |kappa|
    kappa_abs = abs(kappa)
    if kappa_abs < 1e-6:
        # All adhesion, but force is essentially zero
        q_array = kappa_abs * s_array
        f_array = params.kx * q_array
        adhesion_mask = np.ones_like(s_array, dtype=bool)
        return s_array, q_array, f_array, adhesion_mask
    
    # Bristle shear displacement: q(s) = |κ| * s
    q_array = kappa_abs * s_array
    
    # Elastic shear force per unit length: f_elastic(s) = kx * q(s)
    f_elastic = params.kx * q_array
    
    # Transition point where adhesion breaks
    s_star = f_limit / (params.kx * kappa_abs)
    
    # Determine adhesion vs sliding regions
    adhesion_mask = s_array <= s_star
    
    # Local shear force: clamped at friction limit in sliding region
    f_array = np.minimum(f_elastic, f_limit)
    
    return s_array, q_array, f_array, adhesion_mask


def brush_model_diagnostics(
    kappa: float, Fz: float, V: float, params: TireParams
) -> dict:
    """
    Compute brush model diagnostics for a given operating condition.
    
    Returns:
        Dictionary with keys:
        - Fx: Longitudinal tire force (N)
        - s_star: Transition point where adhesion breaks (m), or +inf if full adhesion
        - L: Total contact patch length (m) = 2*a
        - adhesion_length: Length of adhesion region (m)
        - sliding_length: Length of sliding region (m)
        - E_elastic: Total elastic energy stored in adhesion region (J)
        - Pdiss_sliding: Sliding power dissipation from Coulomb slip (W)
        - Pdiss_global: Global dissipation estimate Pdiss = abs(Fx * (κ * V)) (W)
    
    Args:
        kappa: Longitudinal slip ratio (dimensionless, scalar)
        Fz: Normal load (N)
        V: Forward speed at the tire (m/s)
        params: Tire model parameters (mu, a, kx)
    """
    # Contact patch parameters
    L = 2.0 * params.a  # Total patch length (m)
    w = Fz / L  # Uniform line load (N/m)
    f_limit = params.mu * w  # Local friction limit per unit length (N/m)
    
    # Compute Fx
    Fx = brush_model_fx(kappa, Fz, params)
    
    # Handle edge case: very small |kappa|
    kappa_abs = abs(kappa)
    if kappa_abs < 1e-6:
        return {
            'Fx': Fx,
            's_star': np.inf,
            'L': L,
            'adhesion_length': L,
            'sliding_length': 0.0,
            'E_elastic': 0.0,
            'Pdiss_sliding': 0.0,
            'Pdiss_global': 0.0
        }
    
    # Transition point where adhesion breaks
    s_star = f_limit / (params.kx * kappa_abs)
    
    # Region lengths
    adhesion_length = min(s_star, L)
    sliding_length = max(0.0, L - adhesion_length)
    
    # Elastic energy stored in adhesion region
    # E_elastic = (1/6) * kx * |κ|^2 * min(s_star, L)^3
    E_elastic = (1.0 / 6.0) * params.kx * kappa_abs**2 * adhesion_length**3
    
    # Sliding power dissipation
    # Pdiss_sliding = (f_limit * L_sliding) * (|κ| * V)
    Pdiss_sliding = (f_limit * sliding_length) * (kappa_abs * V)
    
    # Global dissipation estimate
    Pdiss_global = abs(Fx * (kappa * V))
    
    return {
        'Fx': Fx,
        's_star': s_star,
        'L': L,
        'adhesion_length': adhesion_length,
        'sliding_length': sliding_length,
        'E_elastic': E_elastic,
        'Pdiss_sliding': Pdiss_sliding,
        'Pdiss_global': Pdiss_global
    }


# Alias for backward compatibility (main interface function)
def compute_fx(kappa: np.ndarray, Fz: float, params: TireParams) -> np.ndarray:
    """
    Compute longitudinal tire force Fx using brush model.
    
    This is the main interface function that maintains compatibility with existing code.
    It calls brush_model_fx internally.
    
    Args:
        kappa: Longitudinal slip ratio (dimensionless, can be array)
        Fz: Normal load (N)
        params: Tire model parameters
        
    Returns:
        Fx: Longitudinal tire force (N), same shape as kappa
    """
    return brush_model_fx(kappa, Fz, params)


def validate_fx_model(kappa: np.ndarray, Fx: np.ndarray, Fz: float, params: TireParams):
    """
    Sanity checks for the brush Fx model.
    
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
        # For brush model, saturation should approach mu*Fz at large slip
        # Use 0.90 threshold to account for partial sliding at moderate kappa
        # At very large kappa (>>0.2), it should approach 1.0
        assert np.all(saturation_ratio > 0.90), \
            f"Fx should saturate near ±{Fx_max_expected}N at large |kappa|, " \
            f"but got max {np.max(np.abs(Fx_large))}N (ratio: {np.max(saturation_ratio):.3f})"
        # Also check that at the maximum kappa, we're closer to saturation
        max_kappa_idx = np.argmax(np.abs(kappa))
        if np.abs(kappa[max_kappa_idx]) > 0.2:
            max_saturation = np.abs(Fx[max_kappa_idx]) / Fx_max_expected
            assert max_saturation > 0.92, \
                f"Fx at max |kappa|={np.abs(kappa[max_kappa_idx]):.3f} should be closer to saturation, " \
                f"but ratio is {max_saturation:.3f}"
    
    # Check: Fx should have the same sign as kappa
    assert np.all(np.sign(Fx) == np.sign(kappa)) or np.all(np.abs(kappa) < 1e-6), \
        "Fx should have the same sign as kappa"
    
    # Check: In partial sliding regime, verify f_array never exceeds f_limit
    # Sample a few kappa values in the partial sliding regime
    test_kappa_values = kappa[np.where((np.abs(kappa) > 0.01) & (np.abs(kappa) < 0.15))[0]]
    if len(test_kappa_values) > 0:
        # Test first few values
        for kappa_test in test_kappa_values[:3]:
            _, _, f_array, _ = brush_model_field(kappa_test, Fz, params)
            L = 2.0 * params.a
            w = Fz / L
            f_limit = params.mu * w
            assert np.all(f_array <= f_limit * 1.01), \
                f"f_array should never exceed f_limit, but max is {np.max(f_array):.2f} vs limit {f_limit:.2f}"
