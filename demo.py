# 9-parameter fitting using lmfit
# Interface points excluded from fitting except the first one (AsH3 to 1050nm transition)

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

df_log = pd.read_csv('2025.2025-05.C2721_HL13B5-WG_L2_C2721_HL13B5-WG_L2_2913.csv')

# Filter log data by message
msg_list = ['introduce AsH3', 
            '1050 nm InGaAsP', 
            '1178nm InGaAsP', 
            '1050nm InGaAsP', 
            'InP cap']
col_list = ['Time (rel)', REFLECTANCE_COLUMN, 'Message']

df_log = df_log[df_log['Message'].isin(msg_list)][col_list].reset_index(drop=True)


df_log[REFLECTANCE_COLUMN] = df_log[REFLECTANCE_COLUMN] 

# Identify First Occurrences and Interface Points
df_log['is_first_occurrence'] = df_log[REFLECTANCE_COLUMN] != df_log[REFLECTANCE_COLUMN].shift(1)
first_occurrences = df_log[df_log['is_first_occurrence']].copy()

# Mark interface points (message changes)
first_occurrences['message_changed'] = first_occurrences['Message'] != first_occurrences['Message'].shift(1)
first_occurrences.iloc[0, first_occurrences.columns.get_loc('message_changed')] = False

# Exclude all interface points EXCEPT the first one
first_occurrences['exclude_from_fitting'] = first_occurrences['message_changed']
first_interface_idx = first_occurrences[first_occurrences['message_changed']].index[0]
first_occurrences.loc[first_interface_idx, 'exclude_from_fitting'] = False

# Get points for visualization
interface_excluded = first_occurrences[first_occurrences['exclude_from_fitting']]
interface_points = first_occurrences[first_occurrences['message_changed']].copy()

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
        
        # Apply 0.865 scaling factor only to the first row

        R = np.vdot(rho, rho).real * 100 *0.865

        reflectance_values.append(R)
    
    return np.array(reflectance_values)

print("\nCalculating initial simulation with default parameters...")
initial_simulation = initial_reflectance()

first_raw = df_log[REFLECTANCE_COLUMN].iloc[0]
norm_ratio = first_raw / initial_simulation[0]
print(f"Normalization ratio (first point): {norm_ratio:.4f}")



'''# Extract statistics
raw_min = raw_data.min()
raw_max = raw_data.max()
raw_first = raw_data.iloc[0]
raw_mean = raw_data.mean()
raw_range = raw_max - raw_min

sim_min = initial_simulation.min()
sim_max = initial_simulation.max()
sim_first = initial_simulation[0]
sim_mean = initial_simulation.mean()
sim_range = sim_max - sim_min

# Calculate ratios
ratio_min = raw_min / sim_min if sim_min != 0 else np.nan
ratio_max = raw_max / sim_max if sim_max != 0 else np.nan
ratio_first = raw_first / sim_first if sim_first != 0 else np.nan
ratio_mean = raw_mean / sim_mean if sim_mean != 0 else np.nan
ratio_range = raw_range / sim_range if sim_range != 0 else np.nan

print(raw_min, raw_max, raw_first, raw_range)
print(sim_min, sim_max, sim_first, sim_range)
print(raw_min/sim_min)
print(raw_max/sim_max)
print(raw_first/sim_first)
print(raw_range/sim_range)

'''

# Normalization


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
        
        # Apply 0.865 scaling factor only to the first row

        R = np.vdot(rho, rho).real * 100  *0.865

        
        reflectance_values.append(R)
    
    return np.array(reflectance_values)

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

# --- Setup Parameters ---
params = Parameters()
params.add('GR_1050', value=0.5, min=0.3, max=0.7)
params.add('GR_1178', value=0.52, min=0.3, max=0.7)
params.add('GR_InP', value=0.42, min=0.2, max=0.6)
params.add('N_1050_real', value=3.84, min=3.5, max=4.0)
params.add('N_1050_imag', value=0.53, min=0.3, max=0.8)
params.add('N_1178_real', value=3.95, min=3.5, max=4.2)
params.add('N_1178_imag', value=0.55, min=0.3, max=0.8)
params.add('N_sub_real', value=3.679, min=3.4, max=3.9)
params.add('N_sub_imag', value=0.52, min=0.3, max=0.8)


# --- Calculate Initial Simulation (BEFORE fitting) ---
print("\nCalculating initial simulation with default parameters...")
initial_simulation = initial_reflectance()  # Uses default parameters


# --- Perform Fitting ---
print("\nStarting optimization...")
result = minimize(lmfit_objective, params, method='leastsq')
print("\n" + "="*50)
print("FITTING RESULTS")
print("="*50)
print(result.params.pretty_print())

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
                  title='9-Parameter Fitting (First Interface Included)',
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