# This version uses leastsq for parameters fitting.
# interface points are neglected for fitting when calculating SSE, which attmpts to improve fitting quality.
# There are totally 9 fitting parameters, which includes refractive indices and growth rates for three types of layers.

# The result of SSE reduction is about 88.1% (Although fewer fitting points will result in lower SSE))
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
first_occurrences = df_log[df_log['is_first_occurrence']]

#print("First occurrences of new EpiReflect values:")
#print(first_occurrences[['EpiReflect1_9.Current Value', 'Message']])


# --- Simulation Parameters ---
#N_sub = 3.679 - 0.52j
#eta_m = N_sub / C

# --- Identify Interface Points ---
# Interface points: first occurrences where the message changed from the previous first occurrence
# Skip the first point (index 0) since it has no previous value to compare
first_occurrences_copy = first_occurrences.copy()
first_occurrences_copy.loc[:, 'message_changed'] = first_occurrences_copy['Message'] != first_occurrences_copy['Message'].shift(1)
# Set first point to False (it's not an interface, just the start)
first_occurrences_copy.iloc[0, first_occurrences_copy.columns.get_loc('message_changed')] = False
interface_points = first_occurrences_copy[first_occurrences_copy['message_changed']].copy()
first_occurrences = first_occurrences_copy

print("\n" + "="*50)
print("INTERFACE POINTS (Message Changes):")
print("="*50)
print(interface_points[['EpiReflect1_9.Current Value', 'Message']])

print("\n" + "="*50)
print("ALL FIRST OCCURRENCES:")
print("="*50)
print(first_occurrences[['EpiReflect1_9.Current Value', 'Message']])

eta_0 = 1 / C

# --- Define Simulation Function ---
def simulate_reflectance(GR_1050=0.5, GR_1178=0.52, GR_InP=0.42, 
                        N_1050_real=3.84, N_1050_imag=0.53, 
                        N_1178_real=3.95, N_1178_imag=0.55,
                        N_sub_real=3.679, N_sub_imag=0.52):
    """
    Simulate reflectance with given parameters
    Returns array of reflectance values
    """
    M = np.eye(2, dtype=complex)
    reflectance_values = []
    
    for idx, row in df_log.iterrows():
        msg = row['Message']
        # Set refractive index and growth rate based on message
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
    
    # Get first occurrence indices and measured values, EXCLUDING interface points
    fitting_points = first_occurrences[~first_occurrences['message_changed']].copy()
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
params.add('N_sub_real', value=3.679, min=3.6, max=3.8)  # Fixed
params.add('N_sub_imag', value=0.52, min=0.3, max=0.6)   # Fixed

# --- Perform Optimization ---
print("\n" + "="*50)
print("STARTING LMFIT OPTIMIZATION")
print("="*50)
print("Initial parameters:")
for name, param in params.items():
    print(f"  {name}: {param.value}")

# Optimize using Levenberg-Marquardt algorithm
print(f"\nRunning optimization...")

result = minimize(lmfit_objective, params, method='lbfgsb')

print(f"Optimization completed!")
print(f"Success: {result.success}")
print(f"Number of function evaluations: {result.nfev}")
print(f"Number of data points: {len(first_occurrences)}")
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
# Initial simulation (original values)
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

# First occurrences with special markers (data used for fitting)
fig.add_trace(go.Scatter(
    x=first_occurrences.index,
    y=first_occurrences['EpiReflect1_9.Current Value'],
    mode='markers',
    marker=dict(size=8, color='red', symbol='diamond'),
    name='Fitting Data Points',
    hovertext=first_occurrences['Message']
))

fig.update_layout(
    width=1000, height=600,
    xaxis_title='Time (sec)',
    yaxis_title='Reflectance (%)',
    title=f'7-Parameter Fitting Results (SSE reduction: {((np.sum(lmfit_objective(params)**2) - result.chisqr) / np.sum(lmfit_objective(params)**2) * 100):.1f}%)',
    legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top')
)
fig.show()

# --- Print Summary Statistics ---
print(f"\n" + "="*50)
print("FITTING QUALITY SUMMARY")
print("="*50)

# Calculate initial and final SSE
initial_residuals = lmfit_objective(params)  # Initial parameters
final_residuals = result.residual
initial_sse = np.sum(initial_residuals**2)
final_sse = result.chisqr  # Same as np.sum(final_residuals**2)
improvement = (initial_sse - final_sse) / initial_sse * 100

print(f"Initial SSE: {initial_sse:.6f}")
print(f"Final SSE: {final_sse:.6f}")
print(f"SSE improvement: {improvement:.1f}%")
print(f"Reduced chi-square: {result.redchi:.6f}")
print(f"Root Mean Square Error: {np.sqrt(final_sse/len(final_residuals)):.4f}")

if hasattr(result, 'aic'):
    print(f"AIC (Akaike Information Criterion): {result.aic:.2f}")
if hasattr(result, 'bic'):
    print(f"BIC (Bayesian Information Criterion): {result.bic:.2f}")

print(f"\nParameter Changes:")
param_names = ['GR_1050', 'GR_1178', 'GR_InP', 'N_1050_real', 'N_1050_imag', 'N_1178_real', 'N_1178_imag']
initial_vals = [0.5, 0.52, 0.42, 3.84, 0.53, 3.95, 0.55]
for name, initial_val in zip(param_names, initial_vals):
    if result.params[name].vary:
        final_val = result.params[name].value
        change = ((final_val - initial_val) / initial_val) * 100
        print(f"  {name}: {initial_val:.4f} → {final_val:.4f} ({change:+.1f}%)")

