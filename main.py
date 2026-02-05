"""
Main script for tire energy dissipation experiments.
Runs experiments and generates plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from tire_model import (
    compute_fx, TireParams, validate_fx_model,
    brush_model_field, brush_model_diagnostics
)
from dissipation import (
    compute_pdiss, compute_pdiss_dissipated, compute_ediss, validate_dissipation
)


# Default parameters for brush model
# mu: friction coefficient, a: half contact patch length (m), kx: bristle stiffness (N/m^2)
DEFAULT_PARAMS = TireParams(mu=1.0, a=0.1, kx=1e6)
DEFAULT_V = 20.0  # m/s


def experiment_a_sweep_kappa():
    """
    Experiment A: Sweep kappa for multiple Fz values.
    Plots Fx vs kappa and Pdiss vs kappa.
    """
    print("Running Experiment A: Sweep kappa for multiple Fz values")
    
    # Parameters
    kappa_range = np.linspace(-0.25, 0.25, 200)
    Fz_values = [600, 900, 1200]  # N
    V = DEFAULT_V
    params = DEFAULT_PARAMS
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Compute and plot for each Fz
    for Fz in Fz_values:
        # Compute Fx
        Fx = compute_fx(kappa_range, Fz, params)
        
        # Compute Pdiss (absolute value for dissipation)
        Pdiss = compute_pdiss_dissipated(Fx, kappa_range, V)
        
        # Plot Fx vs kappa
        ax1.plot(kappa_range, Fx, label=f'Fz = {Fz} N', linewidth=2)
        
        # Plot Pdiss vs kappa
        ax2.plot(kappa_range, Pdiss, label=f'Fz = {Fz} N', linewidth=2)
        
        # Validate model
        validate_fx_model(kappa_range, Fx, Fz, params)
        validate_dissipation(kappa_range, Pdiss)
    
    # Format Fx plot
    ax1.set_xlabel('Longitudinal Slip Ratio κ', fontsize=12)
    ax1.set_ylabel('Longitudinal Force Fx (N)', fontsize=12)
    ax1.set_title('Fx vs κ for Different Normal Loads', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax1.axvline(0, color='k', linestyle='--', linewidth=0.5)
    
    # Format Pdiss plot
    ax2.set_xlabel('Longitudinal Slip Ratio κ', fontsize=12)
    ax2.set_ylabel('Dissipated Power Pdiss (W)', fontsize=12)
    ax2.set_title('Dissipated Power vs κ for Different Normal Loads', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axvline(0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('experiment_a_sweep_kappa.png', dpi=150)
    print("  Saved plot: experiment_a_sweep_kappa.png")
    plt.close()


def experiment_b_time_domain():
    """
    Experiment B: Time-domain example with kappa(t) profile.
    Plots kappa(t), Pdiss(t), and Ediss(t).
    """
    print("Running Experiment B: Time-domain example")
    
    # Parameters
    time_array = np.linspace(0, 3.0, 300)  # 0 to 3 seconds
    Fz = 900.0  # N
    V = DEFAULT_V
    params = DEFAULT_PARAMS
    
    # Define kappa(t) profile: ramp from 0 to 0.15 then hold
    kappa_t = np.zeros_like(time_array)
    ramp_end_time = 1.5  # seconds
    ramp_end_idx = np.argmin(np.abs(time_array - ramp_end_time))
    kappa_max = 0.15
    
    # Ramp phase
    kappa_t[:ramp_end_idx] = kappa_max * (time_array[:ramp_end_idx] / ramp_end_time)
    # Hold phase
    kappa_t[ramp_end_idx:] = kappa_max
    
    # Compute Fx(t)
    Fx_t = compute_fx(kappa_t, Fz, params)
    
    # Compute Pdiss(t) and Ediss(t)
    Pdiss_t = compute_pdiss_dissipated(Fx_t, kappa_t, V)
    Ediss_t = compute_ediss(time_array, Pdiss_t)
    
    # Validate
    validate_fx_model(kappa_t, Fx_t, Fz, params)
    validate_dissipation(kappa_t, Pdiss_t)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Plot kappa(t)
    axes[0].plot(time_array, kappa_t, 'b-', linewidth=2)
    axes[0].set_ylabel('Slip Ratio κ', fontsize=12)
    axes[0].set_title('Time-Domain Tire Energy Dissipation', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='k', linestyle='--', linewidth=0.5)
    
    # Plot Pdiss(t)
    axes[1].plot(time_array, Pdiss_t, 'r-', linewidth=2)
    axes[1].set_ylabel('Dissipated Power Pdiss (W)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='k', linestyle='--', linewidth=0.5)
    
    # Plot Ediss(t)
    axes[2].plot(time_array, Ediss_t, 'g-', linewidth=2)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Cumulative Energy Ediss (J)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('experiment_b_time_domain.png', dpi=150)
    print("  Saved plot: experiment_b_time_domain.png")
    plt.close()
    
    # Print summary
    print("\nTime-Domain Experiment Summary:")
    print(f"  Max Fx: {np.max(np.abs(Fx_t)):.2f} N")
    print(f"  Max Pdiss: {np.max(Pdiss_t):.2f} W")
    print(f"  Total Ediss: {Ediss_t[-1]:.2f} J")
    print(f"  Final kappa: {kappa_t[-1]:.3f}")
    
    return {
        'max_Fx': np.max(np.abs(Fx_t)),
        'max_Pdiss': np.max(Pdiss_t),
        'total_Ediss': Ediss_t[-1]
    }


def experiment_c_brush_field_diagnostics():
    """
    Experiment C: Brush model field and diagnostics for a representative case.
    Plots f_array(s) vs s and q_array(s) vs s, and prints diagnostics.
    """
    print("Running Experiment C: Brush model field and diagnostics")
    
    # Representative operating condition
    kappa = 0.10  # Longitudinal slip ratio
    Fz = 900.0  # N
    V = DEFAULT_V
    params = DEFAULT_PARAMS
    
    # Compute field arrays
    s_array, q_array, f_array, adhesion_mask = brush_model_field(kappa, Fz, params, n_points=200)
    
    # Compute diagnostics
    diag = brush_model_diagnostics(kappa, Fz, V, params)
    
    # Contact patch parameters for plotting
    L = 2.0 * params.a
    w = Fz / L
    f_limit = params.mu * w
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot f_array(s) vs s showing adhesion then friction clamp
    ax1 = axes[0]
    # Plot adhesion region
    ax1.plot(s_array[adhesion_mask], f_array[adhesion_mask], 'b-', 
             linewidth=2, label='Adhesion region', alpha=0.7)
    # Plot sliding region
    if not np.all(adhesion_mask):
        ax1.plot(s_array[~adhesion_mask], f_array[~adhesion_mask], 'r-', 
                 linewidth=2, label='Sliding region', alpha=0.7)
    # Mark transition point
    if diag['s_star'] < np.inf and diag['s_star'] <= L:
        ax1.axvline(diag['s_star'], color='k', linestyle='--', 
                   linewidth=1.5, label=f"s* = {diag['s_star']:.4f} m")
    # Mark friction limit
    ax1.axhline(f_limit, color='g', linestyle=':', 
               linewidth=1.5, label=f'f_limit = {f_limit:.2f} N/m')
    ax1.set_xlabel('Position along patch s (m)', fontsize=12)
    ax1.set_ylabel('Local shear force f(s) (N/m)', fontsize=12)
    ax1.set_title(f'Brush Model: Local Shear Force vs Position (κ={kappa:.3f}, Fz={Fz} N)', 
                  fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, L)
    
    # Plot q_array(s) vs s
    ax2 = axes[1]
    ax2.plot(s_array, q_array, 'b-', linewidth=2, label='Bristle displacement q(s)')
    # Mark transition point if applicable
    if diag['s_star'] < np.inf and diag['s_star'] <= L:
        q_star = abs(kappa) * diag['s_star']
        ax2.plot(diag['s_star'], q_star, 'ro', markersize=8, 
                label=f'Transition point (s*, q*)')
        ax2.axvline(diag['s_star'], color='k', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Position along patch s (m)', fontsize=12)
    ax2.set_ylabel('Bristle shear displacement q(s) (m)', fontsize=12)
    ax2.set_title(f'Brush Model: Bristle Displacement vs Position (κ={kappa:.3f})', 
                  fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, L)
    
    plt.tight_layout()
    plt.savefig('experiment_c_brush_field.png', dpi=150)
    print("  Saved plot: experiment_c_brush_field.png")
    plt.close()
    
    # Print diagnostics summary
    print("\nBrush Model Diagnostics Summary:")
    print(f"  Operating condition: κ = {kappa:.3f}, Fz = {Fz:.1f} N, V = {V:.1f} m/s")
    print(f"  Fx: {diag['Fx']:.2f} N")
    print(f"  Contact patch length L: {diag['L']:.4f} m")
    print(f"  Transition point s*: {diag['s_star']:.4f} m" if diag['s_star'] < np.inf else 
          f"  Transition point s*: +inf (full adhesion)")
    print(f"  Adhesion length: {diag['adhesion_length']:.4f} m")
    print(f"  Sliding length: {diag['sliding_length']:.4f} m")
    print(f"  Elastic energy stored E_elastic: {diag['E_elastic']:.2f} J")
    print(f"  Sliding power dissipation Pdiss_sliding: {diag['Pdiss_sliding']:.2f} W")
    print(f"  Global power dissipation Pdiss_global: {diag['Pdiss_global']:.2f} W")
    
    return diag


def main():
    """Run all experiments."""
    print("=" * 60)
    print("Tire Energy Dissipation Experiments (Brush Model)")
    print("=" * 60)
    print()
    
    # Run experiments
    experiment_a_sweep_kappa()
    print()
    summary = experiment_b_time_domain()
    print()
    diag = experiment_c_brush_field_diagnostics()
    
    print()
    print("=" * 60)
    print("All experiments completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

