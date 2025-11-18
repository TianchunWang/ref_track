# Two-Stage 9-parameter fitting using lmfit
# Stage 1: Fit only Period 1 (1050 nm InGaAsP) with 1050nm parameters
# Stage 2: Fix 1050nm parameters, fit remaining periods with other parameters

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lmfit import Parameters, minimize

# --- Column---
REFLECTANCE_COLUMN = 'EpiReflect1_9.Current Value'  # Change this to match your data column name

# --- Constants ---
WAVELENGTH = 633.0e-9 # WG
C = 2.998e8
PI = np.pi

# Read layer as initial (trial) parameters
df_layer = pd.read_csv('layers.csv', index_col='index')

df_log = pd.read_csv('2025.2025-05.C2772_HL13B5-WG_L0_C2772_HL13B5-WG_L0_2968.csv')

# --- Message Correction Logic ---
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
    
    # Find all "1050 nm InGaAsP" messages after the trigger
    mask = (df_log['Message'] == '1050 nm InGaAsP') & (df_log.index > trigger_idx)
    corrected_count = mask.sum()
    
    if corrected_count > 0:
        df_log.loc[mask, 'Message'] = '1050nm InGaAsP'
        print(f"✓ Corrected {corrected_count} instances of '1050 nm InGaAsP' → '1050nm InGaAsP'")
        print(f"  (after index {trigger_idx})")
    else:
        print("No '1050 nm InGaAsP' messages found after trigger.")
else:
    print(f"⚠ Trigger message not found: '{trigger_msg}'")
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

def initial_reflectance(GR_1050=0.5, GR_1178=0.52, GR_InP=0.42, 
                        N_1050_real=3.84, N_1050_imag=0.53,
                        N_1178_real=3.95, N_1178_imag=0.55,
                        N_sub_real=3.679, N_sub_imag=0.52):
    M = np.eye(2, dtype=complex)
    reflectance_values = []
    
    for idx, (_, row) in enumerate(df_log.iterrows()):
        msg = row['Message']
        
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
initial_simulation = initial_reflectance()

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
from scipy.signal import savgol_filter
df_log[REFLECTANCE_COLUMN] = df_log[REFLECTANCE_COLUMN] / norm_ratio

# Identify First Occurrences and Interface Points
df_log['is_first_occurrence'] = df_log[REFLECTANCE_COLUMN] != df_log[REFLECTANCE_COLUMN].shift(1)
first_occurrences = df_log[df_log['is_first_occurrence']].copy()

# Mark interface points (message changes)
first_occurrences['message_changed'] = first_occurrences['Message'] != first_occurrences['Message'].shift(1)
first_occurrences.iloc[0, first_occurrences.columns.get_loc('message_changed')] = False

# Include all points in fitting
first_occurrences['exclude_from_fitting'] = False

# Get points for visualization
interface_points = first_occurrences[first_occurrences['message_changed']].copy()


def simulate_reflectance(GR_1050=0.5, GR_1178=0.52, GR_InP=0.42, 
                        N_1050_real=3.84, N_1050_imag=0.53, 
                        N_1178_real=3.95, N_1178_imag=0.55,
                        N_sub_real=3.679, N_sub_imag=0.52):
    M = np.eye(2, dtype=complex)
    reflectance_values = []
    
    for idx, (_, row) in enumerate(df_log.iterrows()):
        msg = row['Message']
        
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

# ============================================================================
# STAGE 1: FIT AsH3 + PERIOD 1 (1050 nm InGaAsP) with 1050nm + substrate parameters
# ============================================================================
print("\n" + "="*70)
print("STAGE 1: Fitting AsH3 + Period 1 (1050 nm InGaAsP)")
print("         Parameters: GR_1050, N_1050, N_substrate")
print("="*70)

# Select AsH3 introduction point + Period 1 data points
stage1_mask = (first_occurrences['Message'] == 'introduce AsH3') | (first_occurrences['Message'] == '1050 nm InGaAsP')
stage1_points = first_occurrences[stage1_mask]

print(f"Stage 1 has {len(stage1_points)} fitting points:")
print(f"  - AsH3 points: {(stage1_points['Message'] == 'introduce AsH3').sum()}")
print(f"  - 1050 nm points: {(stage1_points['Message'] == '1050 nm InGaAsP').sum()}")

# Objective function for Stage 1 (AsH3 + Period 1)
def lmfit_objective_stage1(params):
    simulated = simulate_reflectance(
        params['GR_1050'].value, params['GR_1178'].value, params['GR_InP'].value,
        params['N_1050_real'].value, params['N_1050_imag'].value,
        params['N_1178_real'].value, params['N_1178_imag'].value,
        params['N_sub_real'].value, params['N_sub_imag'].value
    )
    
    # Use AsH3 + Period 1 points
    measured = stage1_points[REFLECTANCE_COLUMN].values
    simulated_fit = simulated[stage1_points.index.values]
    
    return simulated_fit - measured

# Setup parameters for Stage 1 - 1050nm parameters + substrate parameters vary
params_stage1 = Parameters()
params_stage1.add('GR_1050', value=0.5, min=0.3, max=0.7, vary=True)
params_stage1.add('N_1050_real', value=3.84, min=3.5, max=4.0, vary=True)
params_stage1.add('N_1050_imag', value=0.53, min=0.3, max=0.8, vary=True)
params_stage1.add('N_sub_real', value=3.679, min=3.4, max=3.9, vary=True)
params_stage1.add('N_sub_imag', value=0.52, min=0.3, max=0.8, vary=True)

# Other parameters fixed at initial values
params_stage1.add('GR_1178', value=0.52, vary=False)
params_stage1.add('GR_InP', value=0.42, vary=False)
params_stage1.add('N_1178_real', value=3.95, vary=False)
params_stage1.add('N_1178_imag', value=0.55, vary=False)

print("Starting Stage 1 optimization (1050nm + substrate parameters)...")
result_stage1 = minimize(lmfit_objective_stage1, params_stage1, method='leastsq')

print("\nStage 1 Results:")
print(f"  GR_1050:      {result_stage1.params['GR_1050'].value:.4f}")
print(f"  N_1050_real:  {result_stage1.params['N_1050_real'].value:.4f}")
print(f"  N_1050_imag:  {result_stage1.params['N_1050_imag'].value:.4f}")
print(f"  N_sub_real:   {result_stage1.params['N_sub_real'].value:.4f}")
print(f"  N_sub_imag:   {result_stage1.params['N_sub_imag'].value:.4f}")

# ============================================================================
# STAGE 2: FIT REMAINING PERIODS with 1050nm + substrate parameters FIXED
# ============================================================================
print("\n" + "="*70)
print("STAGE 2: Fitting Remaining Periods (1178nm, InP)")
print("         FIXED: 1050nm parameters + substrate parameters from Stage 1")
print("="*70)

# Objective function for Stage 2 (all points, but 1050nm + substrate params fixed)
def lmfit_objective_stage2(params):
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

# Setup parameters for Stage 2
params_stage2 = Parameters()
# 1050nm parameters FIXED from Stage 1
params_stage2.add('GR_1050', value=result_stage1.params['GR_1050'].value, vary=False)
params_stage2.add('N_1050_real', value=result_stage1.params['N_1050_real'].value, vary=False)
params_stage2.add('N_1050_imag', value=result_stage1.params['N_1050_imag'].value, vary=False)

# Substrate parameters FIXED from Stage 1
params_stage2.add('N_sub_real', value=result_stage1.params['N_sub_real'].value, vary=False)
params_stage2.add('N_sub_imag', value=result_stage1.params['N_sub_imag'].value, vary=False)

# 1178nm and InP growth rate parameters VARY
params_stage2.add('GR_1178', value=0.52, min=0.3, max=0.7, vary=True)
params_stage2.add('GR_InP', value=0.42, min=0.2, max=0.6, vary=True)
params_stage2.add('N_1178_real', value=3.95, min=3.5, max=4.2, vary=True)
params_stage2.add('N_1178_imag', value=0.55, min=0.3, max=0.8, vary=True)

print("Starting Stage 2 optimization (1178nm and InP growth parameters)...")
result = minimize(lmfit_objective_stage2, params_stage2, method='leastsq')

print("\n" + "="*70)
print("FINAL FITTING RESULTS (After Stage 2)")
print("="*70)
print(result.params.pretty_print())

# --- Generate Optimized Reflectance ---
optimized_reflectance = simulate_reflectance(
    result.params['GR_1050'].value, result.params['GR_1178'].value, result.params['GR_InP'].value,
    result.params['N_1050_real'].value, result.params['N_1050_imag'].value,
    result.params['N_1178_real'].value, result.params['N_1178_imag'].value,
    result.params['N_sub_real'].value, result.params['N_sub_imag'].value
)

# Generate Stage 1 reflectance for visualization
stage1_reflectance = simulate_reflectance(
    result_stage1.params['GR_1050'].value, 0.52, 0.42,
    result_stage1.params['N_1050_real'].value, result_stage1.params['N_1050_imag'].value,
    3.95, 0.55, 3.679, 0.52
)

# Print max and min values
print(f"\n" + "="*70)
print("SIMULATION STATISTICS")
print("="*70)
print(f"Initial Simulation:")
print(f"  Min: {initial_simulation.min():.4f}%")
print(f"  Max: {initial_simulation.max():.4f}%")
print(f"  Mean: {initial_simulation.mean():.4f}%")

print(f"\nStage 1 (1050nm only):")
print(f"  Min: {stage1_reflectance.min():.4f}%")
print(f"  Max: {stage1_reflectance.max():.4f}%")
print(f"  Mean: {stage1_reflectance.mean():.4f}%")

print(f"\nFinal Optimized Simulation:")
print(f"  Min: {optimized_reflectance.min():.4f}%")
print(f"  Max: {optimized_reflectance.max():.4f}%")
print(f"  Mean: {optimized_reflectance.mean():.4f}%")

print(f"\nMeasured Data:")
print(f"  Min: {df_log[REFLECTANCE_COLUMN].min():.4f}%")
print(f"  Max: {df_log[REFLECTANCE_COLUMN].max():.4f}%")
print(f"  Mean: {df_log[REFLECTANCE_COLUMN].mean():.4f}%")

# --- Visualization ---
fitting_points_plot = first_occurrences[~first_occurrences['exclude_from_fitting']]

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_log.index, y=df_log[REFLECTANCE_COLUMN],
                         mode='markers', name='Measured', marker=dict(color='lightgrey', size=4),
                         hovertext=df_log['Message'], hovertemplate='%{hovertext}<br>Reflectance: %{y:.4f}%'))
fig.add_trace(go.Scatter(x=df_log.index, y=initial_simulation,
                         mode='lines', name='Initial Simulation', 
                         line=dict(color='blue', width=2, dash='dash')))
fig.add_trace(go.Scatter(x=df_log.index, y=stage1_reflectance,
                         mode='lines', name='Stage 1 (1050nm fitted)', 
                         line=dict(color='orange', width=1.5, dash='dot')))
fig.add_trace(go.Scatter(x=df_log.index, y=optimized_reflectance,
                         mode='lines', name='Final Fit (Stage 2)', line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=stage1_points.index, 
                         y=stage1_points[REFLECTANCE_COLUMN],
                         mode='markers', marker=dict(size=10, color='cyan', symbol='diamond'),
                         name='Stage 1 Points (AsH3+1050nm)', hovertext=stage1_points['Message']))
fig.add_trace(go.Scatter(x=fitting_points_plot.index, 
                         y=fitting_points_plot[REFLECTANCE_COLUMN],
                         mode='markers', marker=dict(size=8, color='green', symbol='circle'),
                         name='All Fitting Points', hovertext=fitting_points_plot['Message']))

# Add transition lines
colors = ["orange", "cyan", "magenta", "purple", "brown"]
for i, (idx, row) in enumerate(interface_points.iterrows()):
    color = colors[i % len(colors)]
    prev_msg = first_occurrences.loc[first_occurrences.index < idx, 'Message'].iloc[-1] if len(first_occurrences.loc[first_occurrences.index < idx]) > 0 else "Start"
    curr_msg = row['Message']
    
    annotation_text = f"{prev_msg} → {curr_msg}"
    annotation_position = "top left" if i % 2 == 0 else "top right"
    
    fig.add_vline(x=idx, line_dash="dash", line_color=color,
                  annotation_text=annotation_text, 
                  annotation_position=annotation_position)

fig.update_layout(width=1400, height=600, xaxis_title='Time Index', yaxis_title='Reflectance (%)',
                  title='Two-Stage Fitting: AsH3 + Period 1 First, Then Remaining Periods',
                  legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top'))
fig.show()

# --- Performance Metrics ---
print(f"\n" + "="*70)
print("FITTING PERFORMANCE METRICS")
print("="*70)

fitting_points = first_occurrences[~first_occurrences['exclude_from_fitting']]
measured = fitting_points[REFLECTANCE_COLUMN].values
predicted_initial = simulate_reflectance()[fitting_points.index.values]
predicted_optimized = optimized_reflectance[fitting_points.index.values]

n = len(measured)
p = len([param for param in result.params.values() if param.vary]) + len([param for param in result_stage1.params.values() if param.vary])

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
print(f"\n" + "="*70)
print("PARAMETER CHANGES")
print("="*70)
initial_vals = {'GR_1050': 0.5, 'GR_1178': 0.52, 'GR_InP': 0.42, 
                'N_1050_real': 3.84, 'N_1050_imag': 0.53,
                'N_1178_real': 3.95, 'N_1178_imag': 0.55,
                'N_sub_real': 3.679, 'N_sub_imag': 0.52}

print("Stage 1 (1050nm + substrate parameters):")
for name in ['GR_1050', 'N_1050_real', 'N_1050_imag', 'N_sub_real', 'N_sub_imag']:
    initial_val = initial_vals[name]
    final_val = result.params[name].value
    change = ((final_val - initial_val) / initial_val) * 100
    print(f"  {name}: {initial_val:.4f} → {final_val:.4f} ({change:+.1f}%)")

print("\nStage 2 (1178nm and InP growth parameters):")
for name in ['GR_1178', 'GR_InP', 'N_1178_real', 'N_1178_imag']:
    initial_val = initial_vals[name]
    final_val = result.params[name].value
    change = ((final_val - initial_val) / initial_val) * 100
    print(f"  {name}: {initial_val:.4f} → {final_val:.4f} ({change:+.1f}%)")

# --- Export Results to CSV ---
print(f"\n" + "="*70)
print("EXPORTING RESULTS TO CSV")
print("="*70)

# Calculate period lengths
growth_periods = ['1050 nm InGaAsP', '1178nm InGaAsP', '1050nm InGaAsP', 'InP cap']
period_lengths = {}
for i, msg in enumerate(growth_periods):
    period_data = df_log[df_log['Message'] == msg]
    period_lengths[f'period_{i+1}'] = len(period_data)
    print(f"Period {i+1} ({msg}): {len(period_data)} data points")

# Create results dictionary
results_dict = {
    'normalization_fact': norm_ratio,
    'Rsquare': R2_optimized,
    'NRMSE': NRMSE_optimized,
    
    '1050_growth_rate_final': result.params['GR_1050'].value,
    '1178_growth_rate_final': result.params['GR_1178'].value,
    'InP_growth_rate_final': result.params['GR_InP'].value,
    
    'period_1': period_lengths['period_1'],
    'period_2': period_lengths['period_2'],
    'period_3': period_lengths['period_3'],
    'period_4': period_lengths['period_4'],
    
    '1050_real': result.params['N_1050_real'].value,
    '1050_imag': result.params['N_1050_imag'].value,
    '1178_real': result.params['N_1178_real'].value,
    '1178_imag': result.params['N_1178_imag'].value,
    'InP_real': result.params['N_sub_real'].value,
    'InP_imag': result.params['N_sub_imag'].value,
    
    'raw_max': raw_max_original,
    'raw_min': raw_min_original,
    'raw_first': raw_first_original,
    'raw_mean': raw_mean_original,
    
    'fit_max': optimized_reflectance.max(),
    'fit_min': optimized_reflectance.min(),
    'fit_first': optimized_reflectance[0],
    'fit_mean': optimized_reflectance.mean(),
}

results_df = pd.DataFrame([results_dict])
output_filename = 'fitting_results_two_stage.csv'
results_df.to_csv(output_filename, index=False)
print(f"\n✓ Results exported to: {output_filename}")
print(f"✓ Total columns exported: {len(results_dict)}")