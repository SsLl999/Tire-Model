**THIS IS A WORK IN PROGRESS AND EXPERIMENTAL, AI TOOLS ARE BEING USED IN DEVELOPEMENT**
# Tire Energy Dissipation Model - Brush Model Implementation

A Python simulation for modeling tire longitudinal forces and energy dissipation using a steady-state longitudinal brush tire model.

## Overview

This project implements a brush tire model that computes:
- **Longitudinal tire force (Fx)** as a function of slip ratio (κ) and normal load (Fz)
- **Internal bristle deformation field** showing adhesion and sliding regions
- **Elastic energy storage** in the adhesion region
- **Power dissipation (Pdiss)** from both sliding and global estimates
- **Cumulative energy dissipation (Ediss)** over time

The tire model uses a brush mechanics approach where the tread is modeled as bristles that enter the contact patch and can transition from adhesion to sliding regions based on local friction limits.

## Features

- **Experiment A**: Sweep slip ratio (κ) for multiple normal loads, plotting Fx vs κ and Pdiss vs κ
- **Experiment B**: Time-domain analysis with a custom κ(t) profile, showing κ(t), Pdiss(t), and Ediss(t)
- **Experiment C**: Brush model field diagnostics showing internal bristle deformation, adhesion/sliding regions, and detailed energy/power metrics

## Requirements

- Python 3.7+
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0

## Installation

1. Clone this repository:
```bash
git clone git@github.com:SsLl999/Tire-Model.git
cd Tire-Model
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to execute all experiments:

```bash
python main.py
```

This will:
- Run Experiment A (sweep kappa) and save `experiment_a_sweep_kappa.png`
- Run Experiment B (time-domain) and save `experiment_b_time_domain.png`
- Run Experiment C (brush field diagnostics) and save `experiment_c_brush_field.png`
- Print validation results and experiment summaries

## Project Structure

```
.
├── main.py              # Main script with experiment functions
├── tire_model.py        # Brush tire force model (Fx computation, field, diagnostics)
├── dissipation.py       # Energy dissipation calculations
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Model Details

### Brush Tire Model

The model treats the tire tread as bristles that enter the contact patch at the leading edge and move through it. The key mechanics are:

**Contact Patch Parameters:**
- `L = 2*a`: Total contact patch length (meters)
- `w = Fz / L`: Uniform line load (Newtons per meter)
- `f_limit = μ * w`: Local friction limit per unit length (Newtons per meter)

**Bristle Deformation:**
- Position along patch: `s ∈ [0, L]` (meters from leading edge)
- Bristle shear displacement: `q(s) = |κ| * s` (meters)
- Local elastic shear force: `f_elastic(s) = kx * q(s)` (Newtons per meter)

**Adhesion/Sliding Transition:**
- Transition point: `s* = f_limit / (kx * |κ|)` (meters)
- Adhesion region: `s ∈ [0, s*]` where `f(s) = kx * q(s)`
- Sliding region: `s ∈ [s*, L]` where `f(s) = f_limit` (friction limit)

**Force Computation:**
- **Full adhesion** (if `s* ≥ L`):
  - `Fx = sign(κ) * 0.5 * kx * |κ| * L²`
- **Partial sliding** (if `s* < L`):
  - Adhesion contribution: `Fx_adh = 0.5 * kx * |κ| * s*²`
  - Sliding contribution: `Fx_sld = f_limit * (L - s*)`
  - Total: `Fx = sign(κ) * (Fx_adh + Fx_sld)`

**Energy and Power:**
- Elastic energy stored: `E_elastic = (1/6) * kx * |κ|² * min(s*, L)³`
- Sliding power dissipation: `Pdiss_sliding = f_limit * L_sliding * (|κ| * V)`
- Global power dissipation: `Pdiss_global = |Fx * (κ * V)|`

### Model Parameters

The brush model uses:
- `μ` = friction coefficient (dimensionless), default: 1.0
- `a` = half contact patch length (meters), default: 0.1 m
- `kx` = bristle stiffness per unit length (N/m²), default: 1e6 N/m²

## Default Parameters

- Friction coefficient (μ): 1.0
- Half contact patch length (a): 0.1 m
- Bristle stiffness (kx): 1e6 N/m²
- Forward speed (V): 20 m/s
- Normal loads tested: 600 N, 900 N, 1200 N

## Output

The script generates three PNG plots:
- `experiment_a_sweep_kappa.png`: Fx and Pdiss vs κ for different normal loads
- `experiment_b_time_domain.png`: Time-domain plots of κ(t), Pdiss(t), and Ediss(t)
- `experiment_c_brush_field.png`: Brush model field plots showing f(s) vs s and q(s) vs s with adhesion/sliding regions

## Example Output

When you run `python main.py`, you should see output similar to:

```
============================================================
Tire Energy Dissipation Experiments (Brush Model)
============================================================

Running Experiment A: Sweep kappa for multiple Fz values
  Saved plot: experiment_a_sweep_kappa.png

Running Experiment B: Time-domain example
  Saved plot: experiment_b_time_domain.png

Time-Domain Experiment Summary:
  Max Fx: 832.50 N
  Max Pdiss: 2497.50 W
  Total Ediss: 5482.77 J
  Final kappa: 0.150

Running Experiment C: Brush model field and diagnostics
  Saved plot: experiment_c_brush_field.png

Brush Model Diagnostics Summary:
  Operating condition: κ = 0.100, Fz = 900.0 N, V = 20.0 m/s
  Fx: 798.75 N
  Contact patch length L: 0.2000 m
  Transition point s*: 0.0450 m
  Adhesion length: 0.0450 m
  Sliding length: 0.1550 m
  Elastic energy stored E_elastic: 0.15 J
  Sliding power dissipation Pdiss_sliding: 1395.00 W
  Global power dissipation Pdiss_global: 1597.50 W

============================================================
All experiments completed successfully!
============================================================
```

## Key Symbols

- **κ (kappa)**: Longitudinal slip ratio (dimensionless). Use sign(κ) to set driving vs braking direction.
- **Fz**: Normal load (Newtons)
- **V**: Forward speed at the tire (m/s)
- **Fx**: Longitudinal tire force (Newtons), with sign consistent with κ
- **Pdiss**: Dissipated power (Watts)
- **s**: Position along contact patch from leading edge (meters)
- **q(s)**: Bristle shear displacement at position s (meters)
- **f(s)**: Local shear force per unit length at position s (N/m)
- **s***: Transition point where adhesion breaks and sliding begins (meters)

## License

This project is open source and available for educational and research purposes.
