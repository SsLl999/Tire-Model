# Tire Energy Dissipation Model

A Python simulation for modeling tire longitudinal forces and energy dissipation during slip conditions.

## Overview

This project implements a tire model that computes:
- **Longitudinal tire force (Fx)** as a function of slip ratio (κ) and normal load (Fz)
- **Power dissipation (Pdiss)** due to tire slip
- **Cumulative energy dissipation (Ediss)** over time

The tire model uses a smooth saturation function that preserves linearity for small slip ratios and saturates at the maximum friction force for large slip ratios.

## Features

- **Experiment A**: Sweep slip ratio (κ) for multiple normal loads, plotting Fx vs κ and Pdiss vs κ
- **Experiment B**: Time-domain analysis with a custom κ(t) profile, showing κ(t), Pdiss(t), and Ediss(t)

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

Run the main script to execute both experiments:

```bash
python main.py
```

This will:
- Run Experiment A (sweep kappa) and save `experiment_a_sweep_kappa.png`
- Run Experiment B (time-domain) and save `experiment_b_time_domain.png`
- Print validation results and experiment summaries

## Project Structure

```
.
├── main.py              # Main script with experiment functions
├── tire_model.py        # Tire force model (Fx computation)
├── dissipation.py       # Energy dissipation calculations
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Model Details

### Tire Force Model

The longitudinal force is computed using:
```
Fx = sign(κ) * Fx_max * tanh(Ck * κ / Fx_max)
```

Where:
- `Fx_max = μ * Fz` (maximum friction force)
- `μ` = peak friction coefficient (default: 1.0)
- `Ck` = longitudinal stiffness (default: 50000 N/unit slip)
- `κ` = longitudinal slip ratio

### Energy Dissipation

- **Power dissipation**: `Pdiss = |Fx * (κ * V)|`
- **Cumulative energy**: `Ediss = ∫ Pdiss dt`

## Default Parameters

- Friction coefficient (μ): 1.0
- Longitudinal stiffness (Ck): 50000 N/unit slip
- Forward speed (V): 20 m/s
- Normal loads tested: 600 N, 900 N, 1200 N

## Output

The script generates two PNG plots:
- `experiment_a_sweep_kappa.png`: Fx and Pdiss vs κ for different normal loads
- `experiment_b_time_domain.png`: Time-domain plots of κ(t), Pdiss(t), and Ediss(t)

## License

This project is open source and available for educational and research purposes.



