# 12-parameter fitting using lmfit with PARAMETER STANDARDIZATION
# Added normalization factor as a fitting parameter

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lmfit import Parameters, minimize

## CONFIGURATION

# Column name
REFLECTANCE_COLUMN = 'EpiReflect1_9.Current Value'

# Weighting configuration
ENABLE_OSCILLATION_WEIGHTING = True
OSCILLATION_WEIGHT = 1.0
INTERFACE_WEIGHT = 1.0
REGULAR_WEIGHT = 1.0
INITIAL_POINT_WEIGHT = 1.0
APPLY_INTERFACE_TO_NEIGHBORS = False

# Constants
WAVELENGTH = 633.0e-9
C = 2.998e8
PI = np.pi
eta_0 = 1 / C

## PREPROCESSING

df_layer = pd.read_csv('layers.csv', index_col='index')
df_log = pd.read_csv('2025.2025-05.C2721_HL13B5-WG_L2_C2721_HL13B5-WG_L2_2913.csv')

def preprocess_log_data(df_log, reflectance_column):
    print("\n" + "="*70)
    print("MESSAGE CORRECTION")
    print("="*70)

    trigger_msg = 'end 1170nm InGaAsP, set flow for 1050nm InGaAsP'
    trigger_indices = df_log[df_log['Message'] == trigger_msg].index

    if len(trigger_indices) > 0:
        trigger_idx = trigger_indices[0]
        print(f"Found trigger message at index {trigger_idx}: '{trigger_msg}'")

        mask = (df_log['Message'] == '1050 nm InGaAsP') & (df_log.index > trigger_idx)
        corrected_count = mask.sum()

        if corrected_count > 0:
            df_log.loc[mask, 'Message'] = '1050nm InGaAsP'
            print(f"[OK] Corrected {corrected_count} instances of '1050 nm InGaAsP' -> '1050nm InGaAsP'")
        else:
            print("No '1050 nm InGaAsP' messages found after trigger.")
    else:
        print(f"[WARNING] Trigger message not found: '{trigger_msg}'")

    print("\nMessage distribution after correction:")
    for msg in ['introduce AsH3', '1050 nm InGaAsP', '1178nm InGaAsP',
                '1050nm InGaAsP', 'InP cap']:
        count = (df_log['Message'] == msg).sum()
        if count > 0:
            print(f"  '{msg}': {count} occurrences")

    msg_list = ['introduce AsH3', '1050 nm InGaAsP', '1178nm InGaAsP',
                '1050nm InGaAsP', 'InP cap']
    col_list = ['Time (rel)', reflectance_column, 'Message']

    df_filtered = df_log[df_log['Message'].isin(msg_list)][col_list].reset_index(drop=True)

    print(f"\nFiltered data: {len(df_filtered)} rows retained")
    print("="*70)

    return df_filtered

df_log = preprocess_log_data(df_log, REFLECTANCE_COLUMN)

# Store UNNORMALIZED original data
raw_data_original = df_log[REFLECTANCE_COLUMN].copy()
raw_max_original = raw_data_original.max()
raw_min_original = raw_data_original.min()
raw_first_original = raw_data_original.iloc[0]
raw_mean_original = raw_data_original.mean()

## SIMULATION FUNCTION

def simulate_reflectance(GR_1050=0.4664, GR_1178=0.5135, GR_InP=0.4246,
                        N_1050_real=3.8642, N_1050_imag=0.4279,
                        N_1178_real=3.9774, N_1178_imag=0.4498,
                        N_sub_real=3.7076, N_sub_imag=0.4138,
                        N_sub_real_0=3.7076, N_sub_imag_0=0.4138):

    M = np.eye(2, dtype=complex)
    reflectance_values = []

    for idx in range(len(df_log)):
        msg = df_log.loc[idx, 'Message']

        if '1050' in msg:
            N_j = N_1050_real - 1j * N_1050_imag
            GR_j = GR_1050
        elif '1178' in msg:
            N_j = N_1178_real - 1j * N_1178_imag
            GR_j = GR_1178
        elif 'InP' in msg:
            N_j = N_sub_real - 1j * N_sub_imag
            GR_j = GR_InP
        else:
            N_j = 1.0 + 0.0j
            GR_j = 0.0

        d = GR_j * 1.0e-9
        delta_j = 2 * PI * N_j * d / WAVELENGTH
        eta_j = N_j / C
        eta_m = (N_sub_real_0 - 1j * N_sub_imag_0) / C

        M_j = np.array([
            [np.cos(delta_j), 1j * np.sin(delta_j) / eta_j],
            [1j * eta_j * np.sin(delta_j), np.cos(delta_j)]
        ])

        M = M_j @ M

        BC = M @ np.array([1, eta_m])
        rho = (BC[0] * eta_0 - BC[1]) / (BC[0] * eta_0 + BC[1])
        R = np.vdot(rho, rho).real * 100

        reflectance_values.append(R)

    return np.array(reflectance_values)

print("\nCalculating initial simulation with default parameters...")
initial_simulation = simulate_reflectance()

# Calculate initial normalization estimate (but DON'T apply it to data yet)
scaling_estimate = raw_first_original / initial_simulation[0]
print(f"Initial normalization estimate: {scaling_estimate:.6f}")
print(f"This will be used as the initial value for the normalization parameter")

# Keep the ORIGINAL unnormalized data in df_log
# The normalization will be applied inside the objective function

## IDENTIFY POINTS
# Use ORIGINAL data to identify first occurrences
df_log['is_first_occurrence'] = df_log[REFLECTANCE_COLUMN] != df_log[REFLECTANCE_COLUMN].shift(1)
first_occurrences = df_log[df_log['is_first_occurrence']].copy()

first_occurrences['message_changed'] = first_occurrences['Message'] != first_occurrences['Message'].shift(1)
first_occurrences.iloc[0, first_occurrences.columns.get_loc('message_changed')] = False

interface_points = first_occurrences[first_occurrences['message_changed']].copy()

## WEIGHTING FUNCTIONS

def calculate_curvature_weights(measured_values, curvature_weight=100.0):
    first_deriv = np.gradient(measured_values)
    second_deriv = np.gradient(first_deriv)
    abs_curvature = np.abs(second_deriv)

    if abs_curvature.max() > 0:
        normalized_curvature = abs_curvature / abs_curvature.max()
    else:
        normalized_curvature = abs_curvature

    weights = 1.0 + normalized_curvature * (curvature_weight - 1.0)

    return weights, abs_curvature

def apply_interface_weights_with_neighbors(weights, fitting_points, interface_weight):
    interface_mask = fitting_points['message_changed'].values
    interface_positions = np.where(interface_mask)[0]
    interface_indices = []

    for pos in interface_positions:
        weights[pos] = interface_weight
        interface_indices.append(pos)

        if APPLY_INTERFACE_TO_NEIGHBORS:
            if pos > 0:
                weights[pos - 1] = interface_weight
                interface_indices.append(pos - 1)

            if pos < len(weights) - 1:
                weights[pos + 1] = interface_weight
                interface_indices.append(pos + 1)

    interface_indices = sorted(list(set(interface_indices)))

    return weights, interface_indices

## STANDARDIZATION FUNCTIONS

def scale_to_unit(value, bounds):
    """Scale parameter from original range to [0, 1]"""
    min_val, max_val = bounds
    return (value - min_val) / (max_val - min_val)

def unscale_from_unit(scaled_value, bounds):
    """Unscale parameter from [0, 1] back to original range"""
    min_val, max_val = bounds
    return scaled_value * (max_val - min_val) + min_val

## PARAMETER BOUNDS (NOW WITH NORMALIZATION FACTOR)

param_bounds = {
    'GR_1050':      (0.42, 0.8),
    'GR_1178':      (0.4, 0.8),
    'GR_InP':       (0.4, 0.8),
    'N_1050_real':  (3.5, 4.2),
    'N_1050_imag':  (0.2, 0.9),
    'N_1178_real':  (3.5, 4.2),
    'N_1178_imag':  (0.2, 0.9),
    'N_sub_real':   (3.5, 4.2),
    'N_sub_imag':   (0.2, 0.9),
    'N_sub_real_0': (3.5, 4.2),
    'N_sub_imag_0': (0.2, 0.9),
    'norm_factor':  (scaling_estimate * 0.8, scaling_estimate * 1.2),  # ±20% range
}

initial_values = {
    'GR_1050': 0.4664,
    'GR_1178': 0.5135,
    'GR_InP': 0.4246,
    'N_1050_real': 3.8642,
    'N_1050_imag': 0.4279,
    'N_1178_real': 3.9774,
    'N_1178_imag': 0.4498,
    'N_sub_real': 3.7076,
    'N_sub_imag': 0.4138,
    'N_sub_real_0': 3.7076,
    'N_sub_imag_0': 0.4138,
    'norm_factor': scaling_estimate,  # Initial estimate
}

## STANDARDIZED OBJECTIVE FUNCTION (MODIFIED)

def lmfit_objective_std(params_scaled):
    """Objective function with standardized parameters including normalization"""
    params_original = {}
    for name in param_bounds.keys():
        scaled_val = params_scaled[name].value
        original_val = unscale_from_unit(scaled_val, param_bounds[name])
        params_original[name] = original_val

    # Simulate reflectance (theoretical values)
    simulated = simulate_reflectance(
        params_original['GR_1050'],
        params_original['GR_1178'],
        params_original['GR_InP'],
        params_original['N_1050_real'],
        params_original['N_1050_imag'],
        params_original['N_1178_real'],
        params_original['N_1178_imag'],
        params_original['N_sub_real'],
        params_original['N_sub_imag'],
        params_original['N_sub_real_0'],
        params_original['N_sub_imag_0']
    )

    # Apply normalization to SIMULATED data (multiply by norm_factor)
    # This way the simulated data is scaled to match the measured data
    simulated_normalized = simulated * params_original['norm_factor']

    fitting_points = first_occurrences
    # Use ORIGINAL measured data (unnormalized)
    measured = fitting_points[REFLECTANCE_COLUMN].values
    simulated_fit = simulated_normalized[fitting_points.index.values]
    residuals = simulated_fit - measured

    weights = np.ones(len(fitting_points)) * REGULAR_WEIGHT

    if ENABLE_OSCILLATION_WEIGHTING:
        curvature_weights, _ = calculate_curvature_weights(measured, OSCILLATION_WEIGHT)
        weights *= curvature_weights

    weights, interface_indices = apply_interface_weights_with_neighbors(
        weights, fitting_points, INTERFACE_WEIGHT
    )

    weights[0] = max(weights[0], INITIAL_POINT_WEIGHT)
    weighted_residuals = residuals * np.sqrt(weights)

    return weighted_residuals

## SETUP STANDARDIZED PARAMETERS

params_scaled = Parameters()

for name, original_value in initial_values.items():
    scaled_value = scale_to_unit(original_value, param_bounds[name])
    params_scaled.add(name, value=scaled_value, min=0.0, max=1.0)

## DISPLAY CONFIGURATION

print("\n" + "="*70)
print("PARAMETER STANDARDIZATION (12 PARAMETERS)")
print("="*70)
print(f"{'Parameter':<20s} {'Original':<12s} {'Scaled':<12s} {'Range':<30s}")
print("-"*70)

for name in param_bounds.keys():
    orig_val = initial_values[name]
    scaled_val = params_scaled[name].value
    orig_range = f"[{param_bounds[name][0]:.4f}, {param_bounds[name][1]:.4f}]"
    print(f"{name:<20s} {orig_val:<12.6f} {scaled_val:<12.4f} {orig_range:<30s}")
print("="*70)

print("\n" + "="*70)
print("FITTING CONFIGURATION")
print("="*70)
print(f"Total parameters: 12")
print(f"  Growth rates: 3 (GR_1050, GR_1178, GR_InP)")
print(f"  Refractive indices (layers): 4 (N_1050, N_1178)")
print(f"  Refractive indices (InP substrate): 2 (N_sub)")
print(f"  Refractive indices (InP at t=0): 2 (N_sub_0)")
print(f"  Normalization factor: 1 (norm_factor)")
print(f"Optimization: STANDARDIZED parameters [0, 1]")
print("="*70)

print("\n" + "="*70)
print("WEIGHTING CONFIGURATION")
print("="*70)
print(f"Initial point penalty: {INITIAL_POINT_WEIGHT:.1f}x")
print(f"Oscillation weighting: {'ENABLED' if ENABLE_OSCILLATION_WEIGHTING else 'DISABLED'}")
if ENABLE_OSCILLATION_WEIGHTING:
    print(f"  Oscillation weight: {OSCILLATION_WEIGHT:.1f}x")
print(f"Interface weight: {INTERFACE_WEIGHT:.2f}x")
print(f"Regular weight: {REGULAR_WEIGHT:.1f}x")

fitting_points_config = first_occurrences
n_interface = fitting_points_config['message_changed'].sum()
n_regular = len(fitting_points_config) - n_interface

print(f"\nPoint distribution:")
print(f"  Total fitting points: {len(fitting_points_config)}")
print(f"  Interface points: {n_interface}")
print(f"  Regular points: {n_regular}")
print("="*70)

## PERFORM FITTING

print("\nStarting optimization with STANDARDIZED parameters...")
print("Method: BFGS (gradient-based quasi-Newton)")

result_scaled = minimize(lmfit_objective_std, params_scaled, method='bfgs')

## UNSCALE RESULTS

print("\n" + "="*70)
print("UNSCALING RESULTS TO ORIGINAL PARAMETER SPACE")
print("="*70)

# Create a new Parameters object with unscaled values
result_params = Parameters()
for name in param_bounds.keys():
    scaled_val = result_scaled.params[name].value
    original_val = unscale_from_unit(scaled_val, param_bounds[name])
    bounds = param_bounds[name]
    result_params.add(name, value=original_val, min=bounds[0], max=bounds[1])

# Store optimization metadata separately
optimization_success = result_scaled.success
optimization_nfev = result_scaled.nfev
optimization_message = result_scaled.message

print("\n" + "="*50)
print("FITTING RESULTS (Unscaled)")
print("="*50)

# Print parameters manually
print(f"{'Parameter':<20s} {'Value':<15s} {'Min':<12s} {'Max':<12s}")
print("-"*60)
for name in param_bounds.keys():
    param = result_params[name]
    print(f"{name:<20s} {param.value:<15.6f} {param.min:<12.6f} {param.max:<12.6f}")
print("-"*60)

print(f"\nOptimization summary:")
print(f"  Success: {optimization_success}")
print(f"  Function evaluations: {optimization_nfev}")
print(f"  Message: {optimization_message}")

## GENERATE OPTIMIZED REFLECTANCE

optimized_reflectance_raw = simulate_reflectance(
    result_params['GR_1050'].value,
    result_params['GR_1178'].value,
    result_params['GR_InP'].value,
    result_params['N_1050_real'].value,
    result_params['N_1050_imag'].value,
    result_params['N_1178_real'].value,
    result_params['N_1178_imag'].value,
    result_params['N_sub_real'].value,
    result_params['N_sub_imag'].value,
    result_params['N_sub_real_0'].value,
    result_params['N_sub_imag_0'].value
)

# Apply optimized normalization factor
optimized_reflectance = optimized_reflectance_raw * result_params['norm_factor'].value

## INITIAL POINT MATCHING ANALYSIS

print("\n" + "="*70)
print("INITIAL POINT MATCHING ANALYSIS")
print("="*70)

measured_first = first_occurrences[REFLECTANCE_COLUMN].iloc[0]
simulated_first = optimized_reflectance[first_occurrences.index[0]]
initial_error = simulated_first - measured_first
initial_error_pct = (initial_error / measured_first) * 100

print(f"First point comparison:")
print(f"  Measured:  {measured_first:.6f}")
print(f"  Simulated: {simulated_first:.6f}")
print(f"  Error:     {initial_error:+.6f} ({initial_error_pct:+.4f}% relative)")
print(f"  Weight applied: {INITIAL_POINT_WEIGHT:.0f}x")

if abs(initial_error_pct) < 0.01:
    print(f"  ✓ Excellent match! (< 0.01% relative error)")
elif abs(initial_error_pct) < 0.1:
    print(f"  ✓ Very good match (< 0.1% relative error)")
elif abs(initial_error_pct) < 1.0:
    print(f"  ✓ Good match (< 1% relative error)")
else:
    print(f"  ⚠ Warning: Significant mismatch at initial point")

print("="*70)

## NORMALIZATION FACTOR ANALYSIS

print("\n" + "="*70)
print("NORMALIZATION FACTOR ANALYSIS")
print("="*70)

norm_initial = initial_values['norm_factor']
norm_optimized = result_params['norm_factor'].value
norm_change = ((norm_optimized - norm_initial) / norm_initial) * 100

print(f"Initial normalization factor: {norm_initial:.6f}")
print(f"Optimized normalization factor: {norm_optimized:.6f}")
print(f"Change: {norm_change:+.2f}%")
print(f"Bounds: [{param_bounds['norm_factor'][0]:.6f}, {param_bounds['norm_factor'][1]:.6f}]")
print("="*70)

## SCALING FACTOR ANALYSIS

print("\n" + "="*70)
print("SUBSTRATE PROPERTIES ANALYSIS")
print("="*70)

n_real_0_change = ((result_params['N_sub_real_0'].value - initial_values['N_sub_real_0']) /
                   initial_values['N_sub_real_0']) * 100
n_imag_0_change = ((result_params['N_sub_imag_0'].value - initial_values['N_sub_imag_0']) /
                   initial_values['N_sub_imag_0']) * 100

print(f"Optimized substrate properties at t=0:")
print(f"  N_sub_real_0 = {result_params['N_sub_real_0'].value:.6f} ({n_real_0_change:+.2f}%)")
print(f"  N_sub_imag_0 = {result_params['N_sub_imag_0'].value:.6f} ({n_imag_0_change:+.2f}%)")

print(f"\nSubstrate properties during growth:")
print(f"  N_sub_real = {result_params['N_sub_real'].value:.6f}")
print(f"  N_sub_imag = {result_params['N_sub_imag'].value:.6f}")

print(f"\nDifference (growth - t=0):")
diff_real = result_params['N_sub_real'].value - result_params['N_sub_real_0'].value
diff_imag = result_params['N_sub_imag'].value - result_params['N_sub_imag_0'].value
print(f"  Δn (real) = {diff_real:+.6f}")
print(f"  Δκ (imag) = {diff_imag:+.6f}")
print("="*70)

## BOUNDARY CHECK

def check_parameter_boundaries(result_params, param_bounds, boundary_tolerance=1e-6, warning_threshold=5.0):
    """Check if fitted parameters are at or near their boundary limits."""
    print("\n" + "="*50)
    print("BOUNDARY CHECK")
    print("="*50)

    params_at_boundary = []

    for name, param in result_params.items():
        if param.vary:
            param_min = param_bounds[name][0]
            param_max = param_bounds[name][1]

            at_min = abs(param.value - param_min) < boundary_tolerance
            at_max = abs(param.value - param_max) < boundary_tolerance

            param_range = param_max - param_min
            distance_from_min = (param.value - param_min) / param_range * 100
            distance_from_max = (param_max - param.value) / param_range * 100

            if at_min or at_max:
                params_at_boundary.append(name)
                if at_min:
                    print(f"[!] {name:20s} HIT LOWER BOUND: {param.value:.6f} (min={param_min:.6f})")
                if at_max:
                    print(f"[!] {name:20s} HIT UPPER BOUND: {param.value:.6f} (max={param_max:.6f})")
            elif distance_from_min < warning_threshold or distance_from_max < warning_threshold:
                if distance_from_min < warning_threshold:
                    print(f"[~] {name:20s} CLOSE TO LOWER BOUND: {param.value:.6f} ({distance_from_min:.1f}% from min)")
                if distance_from_max < warning_threshold:
                    print(f"[~] {name:20s} CLOSE TO UPPER BOUND: {param.value:.6f} ({distance_from_max:.1f}% from max)")

    if not params_at_boundary:
        print("[OK] All parameters are within bounds (not at limits)")
    else:
        print(f"\n[WARNING] {len(params_at_boundary)} parameter(s) hit boundary limits!")
        print("   Consider widening the bounds for these parameters:")
        for name in params_at_boundary:
            print(f"   - {name}")

    print("\n" + "-"*80)
    print("DETAILED BOUNDARY ANALYSIS")
    print("-"*80)
    print(f"{'Parameter':<20s} {'Value':<12s} {'Min':<12s} {'Max':<12s} {'%Min':<8s} {'%Max':<8s}")
    print("-"*80)

    for name, param in result_params.items():
        if param.vary:
            param_min = param_bounds[name][0]
            param_max = param_bounds[name][1]
            param_range = param_max - param_min
            pct_from_min = (param.value - param_min) / param_range * 100
            pct_from_max = (param_max - param.value) / param_range * 100

            warning = ""
            if abs(param.value - param_min) < boundary_tolerance:
                warning = "[!] MIN"
            elif abs(param.value - param_max) < boundary_tolerance:
                warning = "[!] MAX"
            elif pct_from_min < warning_threshold:
                warning = "[~] ~MIN"
            elif pct_from_max < warning_threshold:
                warning = "[~] ~MAX"

            print(f"{name:<20s} {param.value:<12.6f} {param_min:<12.6f} {param_max:<12.6f} "
                  f"{pct_from_min:<8.1f} {pct_from_max:<8.1f} {warning}")

    print("-"*80)
    print("Legend: [!] = At boundary limit, [~] = Within 5% of boundary")
    print("="*50)

    return params_at_boundary

params_at_boundary = check_parameter_boundaries(result_params, param_bounds)

## STATISTICS

print(f"\n" + "="*50)
print("SIMULATION STATISTICS")
print("="*50)

# Initial simulation (with initial normalization)
initial_simulation_normalized = initial_simulation * scaling_estimate

print(f"Initial Simulation (normalized):")
print(f"  Min: {initial_simulation_normalized.min():.4f}")
print(f"  Max: {initial_simulation_normalized.max():.4f}")
print(f"  Mean: {initial_simulation_normalized.mean():.4f}")

print(f"\nOptimized Simulation (with optimized normalization):")
print(f"  Min: {optimized_reflectance.min():.4f}")
print(f"  Max: {optimized_reflectance.max():.4f}")
print(f"  Mean: {optimized_reflectance.mean():.4f}")
print(f"  First point: {optimized_reflectance[first_occurrences.index[0]]:.6f}")

print(f"\nMeasured Data (original, unnormalized):")
print(f"  Min: {df_log[REFLECTANCE_COLUMN].min():.4f}")
print(f"  Max: {df_log[REFLECTANCE_COLUMN].max():.4f}")
print(f"  Mean: {df_log[REFLECTANCE_COLUMN].mean():.4f}")
print(f"  First point: {first_occurrences[REFLECTANCE_COLUMN].iloc[0]:.6f}")

## VISUALIZATION

fitting_points_plot = first_occurrences

if ENABLE_OSCILLATION_WEIGHTING:
    vis_curv_weights, vis_abs_curvature = calculate_curvature_weights(
        fitting_points_plot[REFLECTANCE_COLUMN].values, OSCILLATION_WEIGHT
    )
    high_curv_threshold_vis = 1.0 + (OSCILLATION_WEIGHT - 1.0) * 0.5
    high_curv_points = fitting_points_plot[vis_curv_weights > high_curv_threshold_vis]
else:
    high_curv_points = pd.DataFrame()  # Empty dataframe if weighting disabled

fig = go.Figure()

# Measured data
fig.add_trace(go.Scatter(
    x=df_log.index,
    y=df_log[REFLECTANCE_COLUMN],
    mode='markers',
    name='Measured (original)',
    marker=dict(color='lightgrey', size=4),
    hovertext=df_log['Message'],
    hovertemplate='%{hovertext}<br>Reflectance: %{y:.4f}<extra></extra>'
))

# Optimized fit
fig.add_trace(go.Scatter(
    x=df_log.index,
    y=optimized_reflectance,
    mode='lines',
    name='Optimized Fit (12-param with norm)',
    line=dict(color='red', width=2)
))

# Fitting points
fig.add_trace(go.Scatter(
    x=fitting_points_plot.index,
    y=fitting_points_plot[REFLECTANCE_COLUMN],
    mode='markers',
    marker=dict(size=8, color='green', symbol='circle'),
    name='Fitting Data Points',
    hovertext=fitting_points_plot['Message'],
    hovertemplate='%{hovertext}<br>Reflectance: %{y:.4f}<extra></extra>'
))

# Initial point
initial_point_idx = first_occurrences.index[0]
initial_point_val = first_occurrences[REFLECTANCE_COLUMN].iloc[0]
fig.add_trace(go.Scatter(
    x=[initial_point_idx],
    y=[initial_point_val],
    mode='markers',
    marker=dict(size=16, color='red', symbol='star',
               line=dict(width=2, color='darkred')),
    name=f'Initial Point (weight={INITIAL_POINT_WEIGHT:.0f}x)',
    hovertemplate=f'Initial Point<br>Reflectance: {initial_point_val:.4f}<extra></extra>'
))

# High-curvature points
if ENABLE_OSCILLATION_WEIGHTING and len(high_curv_points) > 0:
    fig.add_trace(go.Scatter(
        x=high_curv_points.index,
        y=high_curv_points[REFLECTANCE_COLUMN],
        mode='markers',
        marker=dict(size=12, color='purple', symbol='diamond',
                   line=dict(width=2, color='darkviolet')),
        name=f'High Curvature (weight>{high_curv_threshold_vis:.0f}x)',
        hovertext=high_curv_points['Message'],
        hovertemplate='%{hovertext}<br>Reflectance: %{y:.4f}<extra></extra>'
    ))

# Transition lines
colors = ["orange", "cyan", "magenta", "purple", "brown"]
for i, (idx, row) in enumerate(interface_points.iterrows()):
    color = colors[i % len(colors)]
    prev_msg = first_occurrences.loc[first_occurrences.index < idx, 'Message'].iloc[-1] if len(
        first_occurrences.loc[first_occurrences.index < idx]) > 0 else "Start"
    curr_msg = row['Message']
    annotation_text = f"{prev_msg} → {curr_msg}"
    annotation_position = "top left" if i % 2 == 0 else "top right"
    fig.add_vline(x=idx, line_dash="dash", line_color=color,
                  annotation_text=annotation_text,
                  annotation_position=annotation_position)

title_text = '12-Parameter Fitting with Normalization Factor'
fig.update_layout(
    width=1400, height=600,
    xaxis_title='Time Index',
    yaxis_title='Reflectance',
    title=title_text,
    legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top')
)

# Save as HTML file
html_filename = 'fitting_visualization_12param.html'
fig.write_html(html_filename)
print(f"\n[OK] Interactive figure saved to: {html_filename}")
print(f"     Open this file in your web browser to view the plot")

fig.show()

## PERFORMANCE METRICS

print(f"\n" + "="*50)
print("FITTING PERFORMANCE METRICS")
print("="*50)

fitting_points = first_occurrences
measured = fitting_points[REFLECTANCE_COLUMN].values
predicted_initial = initial_simulation_normalized[fitting_points.index.values]
predicted_optimized = optimized_reflectance[fitting_points.index.values]

n = len(measured)
p = 12  # Number of parameters

residuals_initial = predicted_initial - measured
residuals_optimized = predicted_optimized - measured

SS_res_initial = np.sum(residuals_initial**2)
SS_res_optimized = np.sum(residuals_optimized**2)
SS_tot = np.sum((measured - np.mean(measured))**2)

R2_initial = 1 - (SS_res_initial / SS_tot)
R2_optimized = 1 - (SS_res_optimized / SS_tot)
R2_adj_optimized = 1 - (1 - R2_optimized) * (n - 1) / (n - p - 1)

RMSE_initial = np.sqrt(SS_res_initial / n)
RMSE_optimized = np.sqrt(SS_res_optimized / n)
MAE_optimized = np.mean(np.abs(residuals_optimized))
MAPE_optimized = np.mean(np.abs(residuals_optimized / measured)) * 100

data_range = np.max(measured) - np.min(measured)
NRMSE_optimized = (RMSE_optimized / data_range) * 100

print(f"{'Metric':<30} {'Initial':<15} {'Optimized':<15} {'Improvement (%)'}")
print("="*75)
R2_improvement = ((R2_optimized - R2_initial) / abs(R2_initial)) * 100 if R2_initial != 0 else 0
RMSE_improvement = ((RMSE_initial - RMSE_optimized) / RMSE_initial) * 100
SSE_improvement = ((SS_res_initial - SS_res_optimized) / SS_res_initial) * 100

print(f"{'R²':<30} {R2_initial:<15.6f} {R2_optimized:<15.6f} {R2_improvement:+.2f}%")
print(f"{'Adjusted R²':<30} {'-':<15} {R2_adj_optimized:<15.6f} {'-'}")
print(f"{'RMSE':<30} {RMSE_initial:<15.4f} {RMSE_optimized:<15.4f} {RMSE_improvement:+.2f}%")
print(f"{'MAE':<30} {'-':<15} {MAE_optimized:<15.4f} {'-'}")
print(f"{'MAPE (%)':<30} {'-':<15} {MAPE_optimized:<15.4f} {'-'}")
print(f"{'NRMSE (%)':<30} {'-':<15} {NRMSE_optimized:<15.4f} {'-'}")
print(f"{'SSE':<30} {SS_res_initial:<15.4f} {SS_res_optimized:<15.4f} {SSE_improvement:+.2f}%")

print(f"\nR² = {R2_optimized:.4f} (explains {R2_optimized*100:.2f}% of variance)")
print(f"NRMSE = {NRMSE_optimized:.2f}% ({'excellent' if NRMSE_optimized < 10 else 'good' if NRMSE_optimized < 20 else 'fair' if NRMSE_optimized < 30 else 'poor'})")

## PARAMETER CHANGES

print(f"\n" + "="*50)
print("PARAMETER CHANGES (from initial values)")
print("="*50)

for name, initial_val in initial_values.items():
    final_val = result_params[name].value
    change = ((final_val - initial_val) / initial_val) * 100
    print(f"  {name}: {initial_val:.6f} → {final_val:.6f} ({change:+.1f}%)")

## EXPORT RESULTS TO CSV

print(f"\n" + "="*50)
print("EXPORTING RESULTS TO CSV")
print("="*50)

growth_periods = ['1050 nm InGaAsP', '1178nm InGaAsP', '1050nm InGaAsP', 'InP cap']
period_lengths = {}
for i, msg in enumerate(growth_periods):
    period_data = df_log[df_log['Message'] == msg]
    period_lengths[f'period_{i+1}'] = len(period_data)
    print(f"Period {i+1} ({msg}): {len(period_data)} data points")

results_dict = {
    'n_parameters': 12,
    'optimization_method': 'bfgs_standardized_with_norm',
    'initial_point_weight': INITIAL_POINT_WEIGHT,
    'initial_point_error': initial_error,
    'initial_point_error_pct': initial_error_pct,
    'oscillation_weighting_enabled': ENABLE_OSCILLATION_WEIGHTING,
    'oscillation_weight': OSCILLATION_WEIGHT if ENABLE_OSCILLATION_WEIGHTING else 0,
    'Rsquare': R2_optimized,
    'Rsquare_adj': R2_adj_optimized,
    'RMSE': RMSE_optimized,
    'MAE': MAE_optimized,
    'NRMSE': NRMSE_optimized,
    'optimization_success': optimization_success,
    'n_function_evals': optimization_nfev,
    'norm_factor_initial': initial_values['norm_factor'],
    'norm_factor_final': result_params['norm_factor'].value,
    'norm_factor_change_pct': norm_change,
    '1050_growth_rate_final': result_params['GR_1050'].value,
    '1178_growth_rate_final': result_params['GR_1178'].value,
    'InP_growth_rate_final': result_params['GR_InP'].value,
    'period_1': period_lengths['period_1'],
    'period_2': period_lengths['period_2'],
    'period_3': period_lengths['period_3'],
    'period_4': period_lengths['period_4'],
    '1050_real': result_params['N_1050_real'].value,
    '1050_imag': result_params['N_1050_imag'].value,
    '1178_real': result_params['N_1178_real'].value,
    '1178_imag': result_params['N_1178_imag'].value,
    'InP_real': result_params['N_sub_real'].value,
    'InP_imag': result_params['N_sub_imag'].value,
    'InP_real_0': result_params['N_sub_real_0'].value,
    'InP_imag_0': result_params['N_sub_imag_0'].value,
    'raw_max': raw_max_original,
    'raw_min': raw_min_original,
    'raw_first': raw_first_original,
    'raw_mean': raw_mean_original,
    'fit_max': optimized_reflectance.max(),
    'fit_min': optimized_reflectance.min(),
    'fit_first': optimized_reflectance[first_occurrences.index[0]],
    'fit_mean': optimized_reflectance.mean(),
    'params_at_boundary': len(params_at_boundary),
    'boundary_warning': ','.join(params_at_boundary) if params_at_boundary else 'None',
    'interface_weight': INTERFACE_WEIGHT,
    'n_interface_points': n_interface,
    'n_regular_points': n_regular
}

results_df = pd.DataFrame([results_dict])
output_filename = 'fitting_results_12param_with_norm.csv'
results_df.to_csv(output_filename, index=False)
print(f"\n[OK] Results exported to: {output_filename}")
print(f"[OK] Total columns exported: {len(results_dict)}")
print(f"\nKey information:")
print(f"  - Method: BFGS with standardized parameters + normalization")
print(f"  - Parameters: 12 (added norm_factor)")
print(f"  - Rsquare: {R2_optimized:.6f}")
print(f"  - NRMSE: {NRMSE_optimized:.4f}%")
print(f"  - Function evaluations: {optimization_nfev}")
print(f"  - Normalization factor: {result_params['norm_factor'].value:.6f} ({norm_change:+.2f}%)")
