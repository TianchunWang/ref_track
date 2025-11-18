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
message_map = {msg: idx for idx, msg in enumerate(msg_list)}

col_list = ['Time (rel)', 'EpiReflect1_9.Current Value', 'Message']

df_log = df_log[df_log['Message'].isin(msg_list)][col_list].reset_index(drop=True)
df_log['is_first_occurrence'] = df_log['EpiReflect1_9.Current Value'] != df_log['EpiReflect1_9.Current Value'].shift(1)
df_log['time_index'] = df_log.index
#df_log = df_log[df_log['is_first_occurrence']].reset_index(drop=True)

df_log['message_numeric'] = df_log['Message'].map(message_map)
message_changed = df_log['message_numeric'] != df_log['message_numeric'].shift(1)

df_dup = df_log.copy()

df_1050_first = df_dup[df_dup['Message'] == '1050 nm InGaAsP'].copy()
df_1050_second = df_dup[df_dup['Message'] == '1050nm InGaAsP'].copy()
df_1178 = df_dup[df_dup['Message'] == '1178nm InGaAsP'].copy()
df_InP = df_dup[df_dup['Message'] == 'InP cap'].copy()

print("\n" + "="*50)
print("DATA SEPARATION BY LAYER TYPE")
print("="*50)
print(f"1050nm (first phase) data points: {len(df_1050_first)}")
print(f"1050nm (second phase) data points: {len(df_1050_second)}")
print(f"1178nm InGaAsP data points: {len(df_1178)}")
print(f"InP cap data points: {len(df_InP)}")
print(f"Total: {len(df_1050_first) + len(df_1050_second) + len(df_1178) + len(df_InP)}")

exit()
# Method 1: See all rows where message changed
#print("Rows where message changed:")
#print(df_log[message_changed])

first_occurrences = df_log[df_log['is_first_occurrence']].copy()
# Create truncated dataframes for simulation and fitting
#df_log_step1 = df_log[(df_log['message_numeric'] == 1) | (df_log['message_numeric'] == 0) | (df_log['message_numeric'] == 2)].copy()
df_log_step1 = df_log.copy()


#df_log_step1 = first_occurrences[first_occurrences['message_numeric'] == 1].copy().reset_index(drop=True)

#print(f"Fitting data indices: {df_log_step1['time_index'].values}")
# --- Simulation Parameters ---
N_sub = 3.679 - 0.52j
eta_m = N_sub / C
eta_0 = 1 / C

# --- Define Simulation Function (Uses Truncated Data Only) ---
def simulate_reflectance(GR_1050=0.5, GR_1178=0.52, GR_InP=0.42, 
                              N_1050_real=3.84, N_1050_imag=0.53, 
                              N_1178_real=3.95, N_1178_imag=0.55,
                              N_sub_real=3.679, N_sub_imag=0.52):
    """
    Simulate reflectance for step 1 data only
    """
    M = np.eye(2, dtype=complex)
    reflectance_values = []
    
    # Simulate all data up to each step 1 point
    for idx, row in df_log_step1.iterrows():
        msg = row['Message']
        
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
            N_j = 1.0 + 0j
            GR_j = 0.0

        d = GR_j * 1.0e-9  # Convert nm to meters
        delta_j = 2 * PI * N_j * d / WAVELENGTH
        eta_j = N_j / C

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
    Uses only first occurrences from truncated data for fitting
    """
    # Extract parameter values
    GR_1050 = params['GR_1050'].value
    GR_1178 = params['GR_1178'].value 
    GR_InP = params['GR_InP'].value
    N_1050_real = params['N_1050_real'].value
    N_1050_imag = params['N_1050_imag'].value
    N_1178_real = params['N_1178_real'].value
    N_1178_imag = params['N_1178_imag'].value
    N_InP_real = params['N_InP_real'].value
    N_InP_imag = params['N_InP_imag'].value
    
    # Get full simulation for truncated data
    simulated = simulate_reflectance(GR_1050, GR_1178, GR_InP,
                                         N_1050_real, N_1050_imag,
                                         N_1178_real, N_1178_imag,
                                         N_InP_real, N_InP_imag)
    
    # Build mapping from original indices to simulation indices
    first_occ_data = df_log_step1[df_log_step1['is_first_occurrence']]

    # Filter to only include indices between 128 and 158 (inclusive)
    index_mask = (first_occ_data.index <= 121) | (first_occ_data.index >= 128) & (first_occ_data.index <= 506)
    
    first_occ_data = first_occ_data[index_mask]

    first_indices = first_occ_data.index.values
    measured_first = first_occ_data['EpiReflect1_9.Current Value'].values


    #first_occ_data = df_log_step1[df_log_step1['is_first_occurrence']]
    #first_indices = first_occ_data.index.values
    #measured_first = first_occ_data['EpiReflect1_9.Current Value'].values

    simulated_first = simulated[first_indices]

    

    # Return residuals (differences), not SSE
    residuals = simulated_first - measured_first
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
#Refractive indices for InP layer
params.add('N_InP_real', value=3.679, min=3.5, max=3.8)
params.add('N_InP_imag', value=0.52, min=0.4, max=0.6)

# --- Perform Optimization ---
print("\n" + "="*50)
print("STARTING LMFIT OPTIMIZATION")
print("="*50)
print("Initial parameters:")
for name, param in params.items():
    print(f"  {name}: {param.value}")

# Optimize using Levenberg-Marquardt algorithm
print(f"\nRunning optimization...")
result = minimize(lmfit_objective, params, method='leastsq')

print(f"Optimization completed!")
print(f"Success: {result.success}")
print(f"Number of function evaluations: {result.nfev}")
print(f"Number of data points: {len(df_log_step1)}")
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
# Initial simulation (original values) - uses truncated data
if result is not None:
    # Initial simulation for step 1
    initial_reflectance_step1 = simulate_reflectance()
    df_sim_initial_step1 = pd.DataFrame({
        'Time (sec)': df_log_step1.index,
        'Reflectance (%)': initial_reflectance_step1
    })

    # Optimized simulation for step 1
    optimized_reflectance_step1 = simulate_reflectance(
        result.params['GR_1050'].value,
        result.params['GR_1178'].value,
        result.params['GR_InP'].value,
        result.params['N_1050_real'].value,
        result.params['N_1050_imag'].value,
        result.params['N_1178_real'].value,
        result.params['N_1178_imag'].value,
        result.params['N_InP_real'].value,
        result.params['N_InP_imag'].value
    )
    df_sim_optimized_step1 = pd.DataFrame({
        'Time (sec)': df_log_step1.index,
        'Reflectance (%)': optimized_reflectance_step1
    })
else:
    df_sim_initial = pd.DataFrame({'Time (sec)': [], 'Reflectance (%)': []})
    df_sim_optimized= pd.DataFrame({'Time (sec)': [], 'Reflectance (%)': []})

print("Step 1 measured values:")
print(df_log_step1['EpiReflect1_9.Current Value'].values)
print("Step 1 simulated values (initial):")
print(simulate_reflectance())


# Remove the exit() statement and replace the plotting section with this:

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

# First occurrences with special markers (data used for fitting) - like second code
first_occurrence_points = df_log_step1[df_log_step1['is_first_occurrence']]
fig.add_trace(go.Scatter(
    x=first_occurrence_points.index,
    y=first_occurrence_points['EpiReflect1_9.Current Value'],
    mode='markers',
    marker=dict(size=8, color='red', symbol='diamond'),
    name='Fitting Data Points',
    hovertext=first_occurrence_points['Message']
))

# Initial simulation (dashed blue line) - only for step 1 points
if result is not None:
    fig.add_trace(go.Scatter(
        x=df_sim_initial_step1['Time (sec)'],
        y=df_sim_initial_step1['Reflectance (%)'],
        mode='lines',
        line=dict(color='blue', dash='dash', width=2),
        marker=dict(size=6, color='blue'),
        name='Initial Simulation'
    ))

    # Optimized simulation (solid green line) - only for step 1 points
    fig.add_trace(go.Scatter(
        x=df_sim_optimized_step1['Time (sec)'],
        y=df_sim_optimized_step1['Reflectance (%)'],
        mode='lines',
        line=dict(color='green', width=3),
        marker=dict(size=6, color='green'),
        name='Optimized Simulation'
    ))

    # Calculate improvement for title
    initial_residuals = lmfit_objective(params)
    final_residuals = result.residual
    initial_sse = np.sum(initial_residuals**2)
    final_sse = np.sum(final_residuals**2)
    improvement = ((initial_sse - final_sse) / initial_sse * 100) if initial_sse > 0 else 0
    
    title_text = f'Step 1 Data Fitting Results (SSE reduction: {improvement:.1f}%)'
else:
    title_text = 'Step 1 Data Fitting Results'

# Add step labels as annotations for step 1 data
for idx, row in first_occurrence_points.iterrows():
    # Create short labels for each step
    if 'introduce AsH3' in row['Message']:
        label = 'AsH3'
    elif '1050' in row['Message']:
        label = '1050nm'
    elif '1178' in row['Message']:
        label = '1178nm'
    elif 'InP' in row['Message']:
        label = 'InP'
    else:
        label = 'Other'
    
    fig.add_annotation(
        x=idx,
        y=row['EpiReflect1_9.Current Value'],
        text=label,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor='red',
        ax=0,
        ay=-30,
        font=dict(size=10, color='red'),
        bgcolor='white',
        bordercolor='red',
        borderwidth=1
    )

fig.update_layout(
    width=1000, height=600,
    xaxis_title='Time Index',
    yaxis_title='Reflectance (%)',
    title=title_text,
    legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top')
)

fig.show()

# --- Print Summary Statistics ---
print(f"\n" + "="*50)
print("FITTING QUALITY SUMMARY")
print("="*50)

if result is not None:
    # Calculate initial and final SSE
    initial_residuals = lmfit_objective(params)
    final_residuals = result.residual
    initial_sse = np.sum(initial_residuals**2)
    final_sse = np.sum(final_residuals**2)
    improvement = ((initial_sse - final_sse) / initial_sse * 100) if initial_sse > 0 else 0

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
else:
    print("No optimization result available for summary statistics.")
