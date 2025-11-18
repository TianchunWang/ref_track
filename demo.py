# 9-parameter fitting using lmfit
# Interface points excluded from fitting except the first one (AsH3 to 1050nm transition)
# ADDED: Boundary limit checking

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lmfit import Parameters, minimize

# Column name
REFLECTANCE_COLUMN = 'EpiReflect1_8.Current Value'  # Change this to match your data column name

# Constants
WAVELENGTH = 633.0e-9 # WG
C = 2.998e8
PI = np.pi

# Read layer as initial (trial) parameters
df_layer = pd.read_csv('layers.csv', index_col='index')

df_log = pd.read_csv('2025.2025-07.C3300_HL13B5-WG_L1_C3300_HL13B5-WG_L1_3523.csv')


# Fix "1050nm InGaAsP" to "1050 nm InGaAsP" when it appears after "end 1170nm InGaAsP"
print("\n" + "="*70)
print("MESSAGE CORRECTION")
print("="*70)

# Find the trigger message that indicates the second 1050nm period
trigger_msg = 'end 1170nm InGaAsP, set flow for 1050nm InGaAsP'
trigger_indices = df_log[df_log['Message'] == trigger_msg].index

if len(trigger_indices) > 0:
    trigger_idx = trigger_indices[0]
    print(f"Found trigger message at index {trigger_idx}: '{trigger_msg}'")

    # Find all "1050nm InGaAsP" messages after the trigger
    mask = (df_log['Message'] == '1050 nm InGaAsP') & (df_log.index > trigger_idx)
    corrected_count = mask.sum()

    if corrected_count > 0:
        df_log.loc[mask, 'Message'] = '1050nm InGaAsP'
        print(f"[OK] Corrected {corrected_count} instances of '1050 nm InGaAsP' -> '1050nm InGaAsP'")
        print(f"  (after index {trigger_idx})")
    else:
        print("No '1050 nm InGaAsP' messages found after trigger.")
else:
    print(f"[WARNING] Trigger message not found: '{trigger_msg}'")
    print("No correction applied. Please check your data.")

# Verify the correction
print("\nMessage distribution after correction:")
for msg in ['introduce AsH3', '1050 nm InGaAsP', '1178nm InGaAsP', '1050nm InGaAsP', 'InP cap']:
    count = (df_log['Message'] == msg).sum()
    if count > 0:
        print(f"  '{msg}': {count} occurrences")

# Filter log data by message
msg_list = ['introduce AsH3',
            '1050 nm InGaAsP',
            '1178nm InGaAsP',
            '1050nm InGaAsP',
            'InP cap']
col_list = ['Time (rel)', REFLECTANCE_COLUMN, 'Message']

df_log = df_log[df_log['Message'].isin(msg_list)][col_list].reset_index(drop=True)

print(f"\nFiltered data: {len(df_log)} rows retained")
print("="*70)

# --- Simulation Function ---
eta_0 = 1 / C

def simulate_reflectance(GR_1050=0.5, GR_1178=0.52, GR_InP=0.42,
                        N_1050_real=3.84, N_1050_imag=0.53,
                        N_1178_real=3.95, N_1178_imag=0.55,
                        N_sub_real=3.679, N_sub_imag=0.52):
    
    M = np.eye(2, dtype=complex)
    reflectance_values = []

    for idx in range(len(df_log)):

        msg = df_log.loc[idx, 'Message']

        if '1050' in msg:
            N_j, GR_j = N_1050_real - 1j * N_1050_imag, GR_1050
        elif '1178' in msg:
            N_j, GR_j = N_1178_real - 1j * N_1178_imag, GR_1178
        elif 'InP' in msg:
            N_j, GR_j = N_sub_real - 1j * N_sub_imag, GR_InP
        else:
            N_j, GR_j = 1.0, 0.0

        d = GR_j * 1.0e-9
        delta_j = 2 * PI * N_j * d / WAVELENGTH
        eta_j = N_j / C
        eta_m = (N_sub_real - 1j * N_sub_imag) / C

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
first_raw = df_log[REFLECTANCE_COLUMN].iloc[0]
norm_ratio = first_raw / initial_simulation[0]

print(f"Normalization ratio (first point): {norm_ratio:.4f}")

# Store original raw data statistics BEFORE normalization
raw_data_original = df_log[REFLECTANCE_COLUMN].copy()
raw_max_original = raw_data_original.max()
raw_min_original = raw_data_original.min()
raw_first_original = raw_data_original.iloc[0]
raw_mean_original = raw_data_original.mean()

# Normalization

df_log[REFLECTANCE_COLUMN] = df_log[REFLECTANCE_COLUMN] / norm_ratio

# Identify First Occurrences and Interface Points
df_log['is_first_occurrence'] = df_log[REFLECTANCE_COLUMN] != df_log[REFLECTANCE_COLUMN].shift(1)
first_occurrences = df_log[df_log['is_first_occurrence']].copy()

# Mark interface points (message changes)
first_occurrences['message_changed'] = first_occurrences['Message'] != first_occurrences['Message'].shift(1)
first_occurrences.iloc[0, first_occurrences.columns.get_loc('message_changed')] = False

# Exclude all interface points EXCEPT the first one
first_occurrences['exclude_from_fitting'] = False
first_interface_idx = first_occurrences[first_occurrences['message_changed']].index[0]
first_occurrences.loc[first_interface_idx, 'exclude_from_fitting'] = False

# Get points for visualization
interface_excluded = first_occurrences[first_occurrences['exclude_from_fitting']]
interface_points = first_occurrences[first_occurrences['message_changed']].copy()

# --- Objective Function ---
def lmfit_objective(params):
    simulated = simulate_reflectance(
        params['GR_1050'].value, params['GR_1178'].value, params['GR_InP'].value,
        params['N_1050_real'].value, params['N_1050_imag'].value,
        params['N_1178_real'].value, params['N_1178_imag'].value,
        params['N_sub_real'].value, params['N_sub_imag'].value
    )

    fitting_points = first_occurrences[~first_occurrences['exclude_from_fitting']]
    measured = fitting_points[REFLECTANCE_COLUMN].values
    simulated_fit = simulated[fitting_points.index.values]

    return simulated_fit - measured

# --- Setup Parameters --- (5sigma or 3sigma)
params = Parameters()
params.add('GR_1050', value=0.4664, min=0.4350, max=0.4976)
params.add('GR_1178', value=0.5135, min=0.5026, max=0.5244)
params.add('GR_InP', value=0.4246, min=0.4185, max=0.4307) # tight bounds for InP
params.add('N_1050_real', value=3.8642, min=3.8501, max=3.8783)
params.add('N_1050_imag', value=0.4279, min=0.3961, max=0.4598)
params.add('N_1178_real', value=3.9774, min=3.9641, max=3.9908)
params.add('N_1178_imag', value=0.4498, min=0.4098, max=0.4898)
params.add('N_sub_real', value=3.7076, min=3.6994, max=3.7158) # tight bounds for InP
params.add('N_sub_imag', value=0.4138, min=0.3854, max=0.4422) # tight bounds for InP


# --- Calculate Initial Simulation (BEFORE fitting) ---
print("\nCalculating initial simulation with default parameters...")
initial_simulation = simulate_reflectance()  # Uses default parameters

# --- Perform Fitting ---
print("\nStarting optimization...")
result = minimize(lmfit_objective, params, method='leastsq')
print("\n" + "="*50)
print("FITTING RESULTS")
print("="*50)
print(result.params.pretty_print())

# --- NEW: Check Boundary Limits ---
print("\n" + "="*50)
print("BOUNDARY CHECK")
print("="*50)

boundary_tolerance = 1e-6  # Tolerance for checking if at boundary
params_at_boundary = []

for name, param in result.params.items():
    if param.vary:
        at_min = abs(param.value - param.min) < boundary_tolerance
        at_max = abs(param.value - param.max) < boundary_tolerance
        
        # Calculate how close to boundaries (as percentage of range)
        param_range = param.max - param.min
        distance_from_min = (param.value - param.min) / param_range * 100
        distance_from_max = (param.max - param.value) / param_range * 100
        
        if at_min or at_max:
            params_at_boundary.append(name)
            if at_min:
                print(f"[!] {name:20s} HIT LOWER BOUND: {param.value:.6f} (min={param.min:.6f})")
            if at_max:
                print(f"[!] {name:20s} HIT UPPER BOUND: {param.value:.6f} (max={param.max:.6f})")
        elif distance_from_min < 5 or distance_from_max < 5:
            # Warn if within 5% of boundary
            if distance_from_min < 5:
                print(f"[~] {name:20s} CLOSE TO LOWER BOUND: {param.value:.6f} ({distance_from_min:.1f}% from min)")
            if distance_from_max < 5:
                print(f"[~] {name:20s} CLOSE TO UPPER BOUND: {param.value:.6f} ({distance_from_max:.1f}% from max)")

if not params_at_boundary:
    print("[OK] All parameters are within bounds (not at limits)")
else:
    print(f"\n[WARNING] {len(params_at_boundary)} parameter(s) hit boundary limits!")
    print("   Consider widening the bounds for these parameters:")
    for name in params_at_boundary:
        print(f"   - {name}")

# Detailed boundary analysis table
print("\n" + "-"*80)
print("DETAILED BOUNDARY ANALYSIS")
print("-"*80)
print(f"{'Parameter':<20s} {'Value':<12s} {'Min':<12s} {'Max':<12s} {'%Min':<8s} {'%Max':<8s}")
print("-"*80)

for name, param in result.params.items():
    if param.vary:
        param_range = param.max - param.min
        pct_from_min = (param.value - param.min) / param_range * 100
        pct_from_max = (param.max - param.value) / param_range * 100
        
        # Add warning symbols
        warning = ""
        if abs(param.value - param.min) < boundary_tolerance:
            warning = "[!] MIN"
        elif abs(param.value - param.max) < boundary_tolerance:
            warning = "[!] MAX"
        elif pct_from_min < 5:
            warning = "[~] ~MIN"
        elif pct_from_max < 5:
            warning = "[~] ~MAX"
        
        print(f"{name:<20s} {param.value:<12.6f} {param.min:<12.6f} {param.max:<12.6f} "
              f"{pct_from_min:<8.1f} {pct_from_max:<8.1f} {warning}")

print("-"*80)
print("Legend: [!] = At boundary limit, [~] = Within 5% of boundary")
print("        %Min = Distance from minimum (%), %Max = Distance from maximum (%)")
print("="*50)

# --- Generate Optimized Reflectance ---
optimized_reflectance = simulate_reflectance(
    result.params['GR_1050'].value, result.params['GR_1178'].value, result.params['GR_InP'].value,
    result.params['N_1050_real'].value, result.params['N_1050_imag'].value,
    result.params['N_1178_real'].value, result.params['N_1178_imag'].value,
    result.params['N_sub_real'].value, result.params['N_sub_imag'].value
)

# Print max and min values
print(f"\n" + "="*50)
print("SIMULATION STATISTICS")
print("="*50)
print(f"Initial Simulation:")
print(f"  Min: {initial_simulation.min():.4f}%")
print(f"  Max: {initial_simulation.max():.4f}%")
print(f"  Range: {initial_simulation.max() - initial_simulation.min():.4f}%")
print(f"  Mean: {initial_simulation.mean():.4f}%")

print(f"\nOptimized Simulation:")
print(f"  Min: {optimized_reflectance.min():.4f}%")
print(f"  Max: {optimized_reflectance.max():.4f}%")
print(f"  Range: {optimized_reflectance.max() - optimized_reflectance.min():.4f}%")
print(f"  Mean: {optimized_reflectance.mean():.4f}%")

print(f"\nMeasured Data:")
print(f"  Min: {df_log[REFLECTANCE_COLUMN].min():.4f}%")
print(f"  Max: {df_log[REFLECTANCE_COLUMN].max():.4f}%")
print(f"  Range: {df_log[REFLECTANCE_COLUMN].max() - df_log[REFLECTANCE_COLUMN].min():.4f}%")
print(f"  Mean: {df_log[REFLECTANCE_COLUMN].mean():.4f}%")

# --- Visualization ---
fitting_points_plot = first_occurrences[~first_occurrences['exclude_from_fitting']]

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_log.index, y=df_log[REFLECTANCE_COLUMN],
                         mode='markers', name='Measured', marker=dict(color='lightgrey', size=4),
                         hovertext=df_log['Message'], hovertemplate='%{hovertext}<br>Reflectance: %{y:.4f}%'))
fig.add_trace(go.Scatter(x=df_log.index, y=optimized_reflectance,
                         mode='lines', name='Optimized Fit', line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=fitting_points_plot.index,
                         y=fitting_points_plot[REFLECTANCE_COLUMN],
                         mode='markers', marker=dict(size=8, color='green', symbol='circle'),
                         name='Fitting Data Points', hovertext=fitting_points_plot['Message']))
fig.add_trace(go.Scatter(x=interface_excluded.index,
                         y=interface_excluded[REFLECTANCE_COLUMN],
                         mode='markers', marker=dict(size=10, color='red', symbol='x'),
                         name='Interface Points (Excluded)', hovertext=interface_excluded['Message']))
fig.add_trace(go.Scatter(x=df_log.index, y=initial_simulation,
                         mode='lines', name='Initial Simulation',
                         line=dict(color='blue', width=2, dash='dash')))

# Add transition lines
colors = ["orange", "cyan", "magenta", "purple", "brown"]
for i, (idx, row) in enumerate(interface_points.iterrows()):
    color = colors[i % len(colors)]
    prev_msg = first_occurrences.loc[first_occurrences.index < idx, 'Message'].iloc[-1] if len(first_occurrences.loc[first_occurrences.index < idx]) > 0 else "Start"
    curr_msg = row['Message']

    # Create transition label
    annotation_text = f"{prev_msg} → {curr_msg}"
    annotation_position = "top left" if i % 2 == 0 else "top right"

    fig.add_vline(x=idx, line_dash="dash", line_color=color,
                  annotation_text=annotation_text,
                  annotation_position=annotation_position)

fig.update_layout(width=1400, height=600, xaxis_title='Time Index', yaxis_title='Reflectance (%)',
                  title='9-Parameter Fitting (with Boundary Check)',
                  legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top'))
fig.show()

# --- Performance Metrics ---
print(f"\n" + "="*50)
print("FITTING PERFORMANCE METRICS")
print("="*50)

fitting_points = first_occurrences[~first_occurrences['exclude_from_fitting']]
measured = fitting_points[REFLECTANCE_COLUMN].values
predicted_initial = simulate_reflectance()[fitting_points.index.values]
predicted_optimized = optimized_reflectance[fitting_points.index.values]

n = len(measured)
p = len([param for param in result.params.values() if param.vary])

# Calculate metrics
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
print(f"{'RMSE (%)':<30} {RMSE_initial:<15.4f} {RMSE_optimized:<15.4f} {RMSE_improvement:+.2f}%")
print(f"{'MAE (%)':<30} {'-':<15} {MAE_optimized:<15.4f} {'-'}")
print(f"{'MAPE (%)':<30} {'-':<15} {MAPE_optimized:<15.4f} {'-'}")
print(f"{'NRMSE (%)':<30} {'-':<15} {NRMSE_optimized:<15.4f} {'-'}")
print(f"{'SSE':<30} {SS_res_initial:<15.4f} {SS_res_optimized:<15.4f} {SSE_improvement:+.2f}%")

print(f"\nR² = {R2_optimized:.4f} (explains {R2_optimized*100:.2f}% of variance)")
print(f"NRMSE = {NRMSE_optimized:.2f}% ({'excellent' if NRMSE_optimized < 10 else 'good' if NRMSE_optimized < 20 else 'fair' if NRMSE_optimized < 30 else 'poor'})")

# Parameter changes
print(f"\n" + "="*50)
print("PARAMETER CHANGES")
print("="*50)
initial_vals = {'GR_1050': 0.5, 'GR_1178': 0.52, 'GR_InP': 0.42,
                'N_1050_real': 3.84, 'N_1050_imag': 0.53,
                'N_1178_real': 3.95, 'N_1178_imag': 0.55,
                'N_sub_real': 3.679, 'N_sub_imag': 0.52}

for name, initial_val in initial_vals.items():
    if result.params[name].vary:
        final_val = result.params[name].value
        change = ((final_val - initial_val) / initial_val) * 100
        print(f"  {name}: {initial_val:.4f} → {final_val:.4f} ({change:+.1f}%)")

# --- Export Results to CSV ---
print(f"\n" + "="*50)
print("EXPORTING RESULTS TO CSV")
print("="*50)

# Calculate period lengths for the 4 growth periods (excluding 'introduce AsH3')
growth_periods = ['1050 nm InGaAsP', '1178nm InGaAsP', '1050nm InGaAsP', 'InP cap']
period_lengths = {}
for i, msg in enumerate(growth_periods):
    period_data = df_log[df_log['Message'] == msg]
    period_lengths[f'period_{i+1}'] = len(period_data)
    print(f"Period {i+1} ({msg}): {len(period_data)} data points")

# Create a dictionary with all results
results_dict = {
    # Normalization and basic metrics
    'normalization_fact': norm_ratio,
    'Rsquare': R2_optimized,
    'NRMSE': NRMSE_optimized,

    # Fitted growth rates (final values)
    '1050_growth_rate_final': result.params['GR_1050'].value,
    '1178_growth_rate_final': result.params['GR_1178'].value,
    'InP_growth_rate_final': result.params['GR_InP'].value,

    # Period lengths (only the 4 growth periods)
    'period_1': period_lengths['period_1'],
    'period_2': period_lengths['period_2'],
    'period_3': period_lengths['period_3'],
    'period_4': period_lengths['period_4'],

    # Refractive indices - 1050nm
    '1050_real': result.params['N_1050_real'].value,
    '1050_imag': result.params['N_1050_imag'].value,

    # Refractive indices - 1178nm
    '1178_real': result.params['N_1178_real'].value,
    '1178_imag': result.params['N_1178_imag'].value,

    # Refractive indices - InP
    'InP_real': result.params['N_sub_real'].value,
    'InP_imag': result.params['N_sub_imag'].value,

    # Raw data statistics (ORIGINAL, before normalization)
    'raw_max': raw_max_original,
    'raw_min': raw_min_original,
    'raw_first': raw_first_original,
    'raw_mean': raw_mean_original,

    # Fitted data statistics
    'fit_max': optimized_reflectance.max(),
    'fit_min': optimized_reflectance.min(),
    'fit_first': optimized_reflectance[0],
    'fit_mean': optimized_reflectance.mean(),
    
    # NEW: Boundary check flags
    'params_at_boundary': len(params_at_boundary),
    'boundary_warning': ','.join(params_at_boundary) if params_at_boundary else 'None'
}

# Convert to DataFrame and save
results_df = pd.DataFrame([results_dict])
output_filename = 'fitting_results.csv'
results_df.to_csv(output_filename, index=False)
print(f"\n[OK] Results exported to: {output_filename}")
print(f"[OK] Total columns exported: {len(results_dict)}")
print(f"\nColumn names in CSV:")
for col in results_df.columns:
    print(f"  - {col}")