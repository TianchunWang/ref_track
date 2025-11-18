# This version uses leastsq for parameters fitting.
# Interface points are excluded from fitting EXCEPT the first one (AsH3 to 1050nm transition).
# There are totally 9 fitting parameters: refractive indices and growth rates for three types of layers.
# Includes comprehensive fitting performance metrics: R², Adjusted R², RMSE, MAE, MAPE, etc.

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lmfit import Parameters, minimize, fit_report

# --- Constants ---
WAVELENGTH = 633.0e-9  # nm
C = 2.998e8            # m/sec
PI = np.pi

# --- Read Layer Data ---
df_layer = pd.read_csv('layers.csv', index_col='index')
print(df_layer)

# --- Read and Filter Log Data ---
df_log = pd.read_csv('2025.2025-05.C2721_HL13B5-WG_L2_C2721_HL13B5-WG_L2_2913.csv')

msg_list = [
    'introduce AsH3',
    '1050 nm InGaAsP',
    '1178nm InGaAsP',
    '1050nm InGaAsP',
    'InP cap'
]
col_list = ['Time (rel)', 'EpiReflect1_9.Current Value', 'Message']

df_log = df_log[df_log['Message'].isin(msg_list)][col_list].reset_index(drop=True)

# Create a column that marks True for the first occurrence in each continuous group of same EpiReflect values
df_log['is_first_occurrence'] = df_log['EpiReflect1_9.Current Value'] != df_log['EpiReflect1_9.Current Value'].shift(1)
first_occurrences = df_log[df_log['is_first_occurrence']].copy()

# --- Identify Interface Points ---
# Interface points: first occurrences where the message changed from the previous first occurrence
# Skip the first point (index 0) since it has no previous value to compare
first_occurrences.loc[:, 'message_changed'] = first_occurrences['Message'] != first_occurrences['Message'].shift(1)
# Set first point to False (it's not an interface, just the start)
first_occurrences.iloc[0, first_occurrences.columns.get_loc('message_changed')] = False

# Identify the first interface point (AsH3 → 1050nm transition)
first_interface_idx = first_occurrences[first_occurrences['message_changed']].index[0] if len(first_occurrences[first_occurrences['message_changed']]) > 0 else None

# Mark interface points to exclude (all EXCEPT the first interface point)
first_occurrences.loc[:, 'exclude_from_fitting'] = first_occurrences['message_changed'].copy()
if first_interface_idx is not None:
    first_occurrences.loc[first_interface_idx, 'exclude_from_fitting'] = False

interface_points_excluded = first_occurrences[first_occurrences['exclude_from_fitting']].copy()
interface_points_all = first_occurrences[first_occurrences['message_changed']].copy()

print("\n" + "="*50)
print("ALL INTERFACE POINTS (Message Changes):")
print("="*50)
print(interface_points_all[['EpiReflect1_9.Current Value', 'Message']])

print("\n" + "="*50)
print("INTERFACE POINTS EXCLUDED FROM FITTING:")
print("="*50)
print(interface_points_excluded[['EpiReflect1_9.Current Value', 'Message']])

print("\n" + "="*50)
print("FIRST INTERFACE POINT (INCLUDED in fitting):")
print("="*50)
if first_interface_idx is not None:
    print(first_occurrences.loc[first_interface_idx, ['EpiReflect1_9.Current Value', 'Message']])
else:
    print("None found")

print("\n" + "="*50)
print("ALL FIRST OCCURRENCES:")
print("="*50)
print(first_occurrences[['EpiReflect1_9.Current Value', 'Message']])

# Find the actual transition indices (for visualization only)
# Transition 1: 1050 nm InGaAsP -> 1178nm InGaAsP
transition1_indices = df_log[df_log['Message'] == '1178nm InGaAsP'].index
actual_transition1_index = transition1_indices[0] if len(transition1_indices) > 0 else 0

# Transition 2: 1178nm InGaAsP -> 1050nm InGaAsP (second phase)
transition2_indices = df_log[df_log['Message'] == '1050nm InGaAsP'].index
actual_transition2_index = transition2_indices[0] if len(transition2_indices) > 0 else 0

# Transition 3: 1050nm InGaAsP (second phase) -> InP cap
transition3_indices = df_log[df_log['Message'] == 'InP cap'].index
actual_transition3_index = transition3_indices[0] if len(transition3_indices) > 0 else 0

print(f"\n" + "="*50)
print("TRANSITION INDICES (for visualization only):")
print("="*50)
print(f"Transition 1 (1050nm -> 1178nm): index {actual_transition1_index}")
print(f"Transition 2 (1178nm -> 1050nm second): index {actual_transition2_index}")
print(f"Transition 3 (1050nm second -> InP cap): index {actual_transition3_index}")

# --- Simulation Parameters ---
eta_0 = 1 / C

# --- Define Simulation Function ---
def simulate_reflectance(GR_1050=0.5, GR_1178=0.52, GR_InP=0.42, 
                        N_1050_real=3.84, N_1050_imag=0.53, 
                        N_1178_real=3.95, N_1178_imag=0.55,
                        N_sub_real=3.679, N_sub_imag=0.52):
    """
    Simulate reflectance with given parameters
    Returns array of reflectance values
    Uses message-based layer selection only
    """
    M = np.eye(2, dtype=complex)
    reflectance_values = []
    
    for idx, row in df_log.iterrows():
        msg = row['Message']
        
        # Use message to determine layer type
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
            N_j = 1.0
            GR_j = 0.0

        d = GR_j * 1.0e-9  # Convert nm to meters
        delta_j = 2 * PI * N_j * d / WAVELENGTH
        eta_j = N_j / C

        eta_m = (N_sub_real - 1j * N_sub_imag) / C

        M_j = np.array([
            [np.cos(delta_j), 1j * np.sin(delta_j) / eta_j],
            [1j * eta_j * np.sin(delta_j), np.cos(delta_j)]
        ])
        M = M_j @ M

        BC = M @ np.array([1, eta_m])
        B_val, C_val = BC[0], BC[1]
        rho = (B_val * eta_0 - C_val) / (B_val * eta_0 + C_val)
        R = np.vdot(rho, rho).real * 100 * 0.865  # Scale factor
        
        reflectance_values.append(R)
    
    return np.array(reflectance_values)

# --- Define Objective Function for lmfit ---
def lmfit_objective(params):
    """
    Objective function for lmfit: returns residuals (not SSE)
    lmfit minimizes sum of squares of residuals automatically
    EXCLUDES interface points from fitting EXCEPT the first one (AsH3 → 1050nm)
    """
    # Extract parameter values
    GR_1050 = params['GR_1050'].value
    GR_1178 = params['GR_1178'].value 
    GR_InP = params['GR_InP'].value
    N_1050_real = params['N_1050_real'].value
    N_1050_imag = params['N_1050_imag'].value
    N_1178_real = params['N_1178_real'].value
    N_1178_imag = params['N_1178_imag'].value
    N_sub_real = params['N_sub_real'].value
    N_sub_imag = params['N_sub_imag'].value
    
    # Simulate reflectance with current parameters
    simulated = simulate_reflectance(GR_1050, GR_1178, GR_InP, 
                                   N_1050_real, N_1050_imag, 
                                   N_1178_real, N_1178_imag,
                                   N_sub_real, N_sub_imag)
    
    # Get first occurrence indices and measured values, EXCLUDING marked interface points
    # (all interface points EXCEPT the first one)
    fitting_points = first_occurrences[~first_occurrences['exclude_from_fitting']].copy()
    fitting_indices = fitting_points.index.values
    measured_fitting = fitting_points['EpiReflect1_9.Current Value'].values
    simulated_fitting = simulated[fitting_indices]
    
    # Return residuals (differences), not SSE
    residuals = simulated_fitting - measured_fitting
    return residuals

# --- Setup Fitting Parameters ---
params = Parameters()
# Growth rates
params.add('GR_1050', value=0.5, min=0.1, max=1.0)
params.add('GR_1178', value=0.52, min=0.1, max=1.0)
params.add('GR_InP', value=0.42, min=0.1, max=1.0)
# Refractive indices for 1050nm layer
params.add('N_1050_real', value=3.84, min=3.8, max=4.0)
params.add('N_1050_imag', value=0.53, min=0.4, max=0.6)
# Refractive indices for 1178nm layer
params.add('N_1178_real', value=3.95, min=3.9, max=4.1)
params.add('N_1178_imag', value=0.55, min=0.4, max=0.6)
# Refractive indices for substrate
params.add('N_sub_real', value=3.679, min=3.6, max=3.8)
params.add('N_sub_imag', value=0.52, min=0.3, max=0.6)

# --- Perform Optimization ---
print("\n" + "="*50)
print("STARTING LMFIT OPTIMIZATION (9 PARAMETERS)")
print("="*50)
fitting_count = len(first_occurrences[~first_occurrences['exclude_from_fitting']])
excluded_count = len(interface_points_excluded)
print(f"Fitting points: {fitting_count} (excluding {excluded_count} interface points, keeping first interface point)")
print("Initial parameters:")
for name, param in params.items():
    print(f"  {name}: {param.value}")

# Optimize using Levenberg-Marquardt algorithm
print(f"\nRunning optimization...")

result = minimize(lmfit_objective, params, method='leastsq')
#or bfgs: method='lbfgsb'

print(f"Optimization completed!")
print(f"Success: {result.success}")
print(f"Number of function evaluations: {result.nfev}")
print(f"Number of data points used for fitting: {fitting_count}")
print(f"Number of fitted parameters: {len([p for p in result.params.values() if p.vary])}")

# --- Display Results ---
print(f"\n" + "="*50)
print("OPTIMIZED PARAMETERS")
print("="*50)
for name, param in result.params.items():
    if param.vary:
        stderr_str = f" ± {param.stderr:.4f}" if param.stderr else " ± N/A"
        print(f"  {name}: {param.value:.4f}{stderr_str}")
    else:
        print(f"  {name}: {param.value:.4f} (fixed)")

# Print detailed fit report
print(f"\n" + "="*50)
print("DETAILED FIT REPORT")
print("="*50)
print(fit_report(result))

# --- Generate Simulation Data for Plotting ---
# Initial simulation (original values, no transition adjustment)
initial_reflectance = simulate_reflectance()  # Uses default values
df_sim_initial = pd.DataFrame({
    'Time (sec)': df_log.index,
    'Reflectance (%)': initial_reflectance
})

# Optimized simulation
optimized_reflectance = simulate_reflectance(
    result.params['GR_1050'].value,
    result.params['GR_1178'].value,
    result.params['GR_InP'].value,
    result.params['N_1050_real'].value,
    result.params['N_1050_imag'].value,
    result.params['N_1178_real'].value,
    result.params['N_1178_imag'].value,
    result.params['N_sub_real'].value,
    result.params['N_sub_imag'].value
)
df_sim_optimized = pd.DataFrame({
    'Time (sec)': df_log.index,
    'Reflectance (%)': optimized_reflectance
})

# --- Plot Results ---
fig = go.Figure()

# All measured data points in light grey (background)
fig.add_trace(go.Scatter(
    x=df_log.index,
    y=df_log['EpiReflect1_9.Current Value'],
    hovertext=df_log['Message'],
    mode='markers',
    marker=dict(size=4, color='lightgrey'),
    name='All Measured Points'
))

# Initial simulation (dashed blue line)
fig.add_trace(go.Scatter(
    x=df_sim_initial['Time (sec)'],
    y=df_sim_initial['Reflectance (%)'],
    mode='lines',
    line=dict(color='blue', dash='dash', width=2),
    name='Initial Simulation'
))

# Optimized simulation (solid green line)
fig.add_trace(go.Scatter(
    x=df_sim_optimized['Time (sec)'],
    y=df_sim_optimized['Reflectance (%)'],
    mode='lines',
    line=dict(color='green', width=3),
    name='Optimized Simulation'
))

# Non-interface first occurrences + first interface (data used for fitting) - green diamonds
fitting_points_for_plot = first_occurrences[~first_occurrences['exclude_from_fitting']]
fig.add_trace(go.Scatter(
    x=fitting_points_for_plot.index,
    y=fitting_points_for_plot['EpiReflect1_9.Current Value'],
    mode='markers',
    marker=dict(size=8, color='green', symbol='diamond'),
    name='Fitting Data Points',
    hovertext=fitting_points_for_plot['Message']
))

# Interface points EXCLUDED from fitting - red X markers (not including first interface)
fig.add_trace(go.Scatter(
    x=interface_points_excluded.index,
    y=interface_points_excluded['EpiReflect1_9.Current Value'],
    mode='markers',
    marker=dict(size=10, color='red', symbol='x'),
    name='Interface Points (Excluded)',
    hovertext=interface_points_excluded['Message']
))

# Add vertical lines to show transition points (for reference only)
fig.add_vline(x=actual_transition1_index, line_dash="dash", line_color="orange", 
              annotation_text="T1: 1050→1178", annotation_position="top left")
fig.add_vline(x=actual_transition2_index, line_dash="dash", line_color="cyan", 
              annotation_text="T2: 1178→1050", annotation_position="top")
fig.add_vline(x=actual_transition3_index, line_dash="dash", line_color="magenta", 
              annotation_text="T3: 1050→InP", annotation_position="top right")

fig.update_layout(
    width=1400, height=600,
    xaxis_title='Time Index',
    yaxis_title='Reflectance (%)',
    title=f'9-Parameter Fitting (First Interface Included) - SSE reduction: {((np.sum(lmfit_objective(params)**2) - result.chisqr) / np.sum(lmfit_objective(params)**2) * 100):.1f}%',
    legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top')
)
fig.show()

# --- Calculate Comprehensive Performance Metrics ---
print(f"\n" + "="*50)
print("COMPREHENSIVE FITTING PERFORMANCE METRICS")
print("="*50)

# Get fitting data
fitting_points = first_occurrences[~first_occurrences['exclude_from_fitting']].copy()
fitting_indices = fitting_points.index.values
measured_fitting = fitting_points['EpiReflect1_9.Current Value'].values

# Initial predictions
simulated_initial = simulate_reflectance()
predicted_initial = simulated_initial[fitting_indices]

# Optimized predictions
predicted_optimized = optimized_reflectance[fitting_indices]

# Calculate metrics for FITTED points
n = len(measured_fitting)  # number of observations
p = len([param for param in result.params.values() if param.vary])  # number of parameters

# Residuals
residuals_initial = predicted_initial - measured_fitting
residuals_optimized = predicted_optimized - measured_fitting

# Sum of Squares
SS_res_initial = np.sum(residuals_initial**2)  # Residual sum of squares (initial)
SS_res_optimized = np.sum(residuals_optimized**2)  # Residual sum of squares (optimized)
SS_tot = np.sum((measured_fitting - np.mean(measured_fitting))**2)  # Total sum of squares

# R-squared (Coefficient of Determination)
R2_initial = 1 - (SS_res_initial / SS_tot)
R2_optimized = 1 - (SS_res_optimized / SS_tot)

# Adjusted R-squared
R2_adj_initial = 1 - (1 - R2_initial) * (n - 1) / (n - p - 1)
R2_adj_optimized = 1 - (1 - R2_optimized) * (n - 1) / (n - p - 1)

# Root Mean Square Error (RMSE)
RMSE_initial = np.sqrt(SS_res_initial / n)
RMSE_optimized = np.sqrt(SS_res_optimized / n)

# Mean Absolute Error (MAE)
MAE_initial = np.mean(np.abs(residuals_initial))
MAE_optimized = np.mean(np.abs(residuals_optimized))

# Mean Absolute Percentage Error (MAPE)
MAPE_initial = np.mean(np.abs(residuals_initial / measured_fitting)) * 100
MAPE_optimized = np.mean(np.abs(residuals_optimized / measured_fitting)) * 100

# Maximum Absolute Error
Max_Error_initial = np.max(np.abs(residuals_initial))
Max_Error_optimized = np.max(np.abs(residuals_optimized))

# Normalized RMSE (NRMSE) - normalized by range
data_range = np.max(measured_fitting) - np.min(measured_fitting)
NRMSE_initial = (RMSE_initial / data_range) * 100
NRMSE_optimized = (RMSE_optimized / data_range) * 100

print(f"\n{'Metric':<30} {'Initial':<15} {'Optimized':<15} {'Improvement':<15}")
print("="*75)

# R-squared
print(f"{'R² (Coefficient of Det.)':<30} {R2_initial:<15.6f} {R2_optimized:<15.6f} {R2_optimized - R2_initial:+.6f}")

# Adjusted R-squared
print(f"{'Adjusted R²':<30} {R2_adj_initial:<15.6f} {R2_adj_optimized:<15.6f} {R2_adj_optimized - R2_adj_initial:+.6f}")

# RMSE
print(f"{'RMSE (%)':<30} {RMSE_initial:<15.4f} {RMSE_optimized:<15.4f} {RMSE_optimized - RMSE_initial:+.4f}")

# MAE
print(f"{'MAE (%)':<30} {MAE_initial:<15.4f} {MAE_optimized:<15.4f} {MAE_optimized - MAE_initial:+.4f}")

# MAPE
print(f"{'MAPE (%)':<30} {MAPE_initial:<15.4f} {MAPE_optimized:<15.4f} {MAPE_optimized - MAPE_initial:+.4f}")

# NRMSE
print(f"{'NRMSE (%)':<30} {NRMSE_initial:<15.4f} {NRMSE_optimized:<15.4f} {NRMSE_optimized - NRMSE_initial:+.4f}")

# Max Error
print(f"{'Max Absolute Error (%)':<30} {Max_Error_initial:<15.4f} {Max_Error_optimized:<15.4f} {Max_Error_optimized - Max_Error_initial:+.4f}")

# SSE
print(f"{'SSE (Sum of Sq. Errors)':<30} {SS_res_initial:<15.4f} {SS_res_optimized:<15.4f} {SS_res_optimized - SS_res_initial:+.4f}")

print("\n" + "="*75)
print("METRIC INTERPRETATIONS:")
print("="*75)
print(f"R²: {R2_optimized:.4f}")
print("  → Explains {:.2f}% of variance in the data".format(R2_optimized * 100))
print("  → 1.0 = perfect fit, 0.0 = no better than mean")
print(f"\nAdjusted R²: {R2_adj_optimized:.4f}")
print("  → Adjusted for number of parameters")
print("  → Penalizes overfitting with too many parameters")
print(f"\nRMSE: {RMSE_optimized:.4f}%")
print("  → Average prediction error in same units as data")
print("  → Lower is better")
print(f"\nMAPE: {MAPE_optimized:.4f}%")
print("  → Average percentage error")
print("  → Scale-independent metric")
print(f"\nNRMSE: {NRMSE_optimized:.4f}%")
print("  → RMSE normalized by data range")
print("  → <10% = excellent, 10-20% = good, 20-30% = fair, >30% = poor")

# --- Calculate SSE on ALL first occurrence points (for fair comparison) ---
print(f"\n" + "="*50)
print("SSE ON ALL POINTS (Fair Comparison)")
print("="*50)

all_indices = first_occurrences.index.values
measured_all = first_occurrences['EpiReflect1_9.Current Value'].values
simulated_all_initial = simulate_reflectance()
simulated_all_optimized = simulate_reflectance(
    result.params['GR_1050'].value,
    result.params['GR_1178'].value,
    result.params['GR_InP'].value,
    result.params['N_1050_real'].value,
    result.params['N_1050_imag'].value,
    result.params['N_1178_real'].value,
    result.params['N_1178_imag'].value,
    result.params['N_sub_real'].value,
    result.params['N_sub_imag'].value
)
residuals_all_initial = simulated_all_initial[all_indices] - measured_all
residuals_all_optimized = simulated_all_optimized[all_indices] - measured_all
sse_all_initial = np.sum(residuals_all_initial**2)
sse_all_optimized = np.sum(residuals_all_optimized**2)
improvement_all = (sse_all_initial - sse_all_optimized) / sse_all_initial * 100

# R² for all points
SS_tot_all = np.sum((measured_all - np.mean(measured_all))**2)
R2_all_initial = 1 - (sse_all_initial / SS_tot_all)
R2_all_optimized = 1 - (sse_all_optimized / SS_tot_all)

print(f"SSE on ALL points (including all interface points):")
print(f"  Initial SSE: {sse_all_initial:.6f}")
print(f"  Optimized SSE: {sse_all_optimized:.6f}")
print(f"  SSE improvement: {improvement_all:.2f}%")
print(f"\nR² on ALL points:")
print(f"  Initial R²: {R2_all_initial:.6f}")
print(f"  Optimized R²: {R2_all_optimized:.6f}")
print(f"  R² improvement: {R2_all_optimized - R2_all_initial:+.6f}")

if hasattr(result, 'aic'):
    print(f"\nAIC (Akaike Information Criterion): {result.aic:.2f}")
if hasattr(result, 'bic'):
    print(f"BIC (Bayesian Information Criterion): {result.bic:.2f}")

print(f"\n" + "="*50)
print("PARAMETER CHANGES:")
print("="*50)
param_names = ['GR_1050', 'GR_1178', 'GR_InP', 'N_1050_real', 'N_1050_imag', 'N_1178_real', 'N_1178_imag', 'N_sub_real', 'N_sub_imag']
initial_vals = [0.5, 0.52, 0.42, 3.84, 0.53, 3.95, 0.55, 3.679, 0.52]
for name, initial_val in zip(param_names, initial_vals):
    if result.params[name].vary:
        final_val = result.params[name].value
        change = ((final_val - initial_val) / initial_val) * 100
        print(f"  {name}: {initial_val:.4f} → {final_val:.4f} ({change:+.1f}%)")