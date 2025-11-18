# 9-parameter fitting using lmfit
# Interface points excluded from fitting except the first one (AsH3 to 1050nm transition)
# ADDED: Boundary limit checking
# ADDED: Data smoothing before fitting

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lmfit import Parameters, minimize
from scipy.signal import savgol_filter  # <<<< CHANGE 1: Added import for smoothing

# Column name
REFLECTANCE_COLUMN = 'EpiReflect1_1.Current Value'  # Change this to match your data column name
#df_log = pd.read_csv('2025.2025-07.C3300_HL13B5-WG_L1_C3300_HL13B5-WG_L1_3523.csv')

'''
start_idx, end_idx = 1400, 2200
df_focus = df_log.loc[start_idx:end_idx]

col3 = 'EpiReflect1_7.Current Value'
col4 = 'EpiReflect1_8.Current Value'

# Calculate max, min, and difference for both columns
max3 = df_focus[col3].max()
min3 = df_focus[col3].min()
diff3 = max3 - min3

max4 = df_focus[col4].max()
min4 = df_focus[col4].min()
diff4 = max4 - min4

print(f"\nTime index {start_idx} to {end_idx}:")
print(f"{col3}: max={max3:.4f}, min={min3:.4f}, diff={diff3:.4f}")
print(f"{col4}: max={max4:.4f}, min={min4:.4f}, diff={diff4:.4f}")

# Plot with max/min lines
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_focus.index,
    y=df_focus[col3],
    mode='lines+markers',
    name=col3
))
fig.add_trace(go.Scatter(
    x=df_focus.index,
    y=df_focus[col4],
    mode='lines+markers',
    name=col4
))
# Add max/min lines for col3
fig.add_hline(y=max3, line_dash="dash", line_color="blue", annotation_text=f"{col3} max", annotation_position="top left")
fig.add_hline(y=min3, line_dash="dash", line_color="blue", annotation_text=f"{col3} min", annotation_position="bottom left")
# Add max/min lines for col4
fig.add_hline(y=max4, line_dash="dash", line_color="red", annotation_text=f"{col4} max", annotation_position="top right")
fig.add_hline(y=min4, line_dash="dash", line_color="red", annotation_text=f"{col4} min", annotation_position="bottom right")

fig.update_layout(
    title=f'Comparison of {col3} and {col4} (Index {start_idx}-{end_idx})',
    xaxis_title='Index',
    yaxis_title='Reflectance (%)',
    width=1000,
    height=500
)
fig.show()
'''
# <<<< CHANGE 2: Added smoothing configuration with both methods
# Smoothing parameters
ENABLE_SMOOTHING = False    # Set to False to disable smoothing
SMOOTHING_METHOD = 'savgol'      # Options: 'savgol' or 'ewma'

# Savitzky-olay filter parameter
SAVGOL_WINDOW = 11               # Must be odd number, larger = more smoothing
SAVGOL_POLYORDER = 3             # Polynomial order, typically 2-5

# EWMA (Exponential Weighted Moving Average) parameters
EWMA_SPAN = 10                   # Larger = more smoothing, typical range: 5-20

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

def simulate_reflectance(GR_1050=0.4664, GR_1178=0.5135, GR_InP=0.4246,  # <<<< CHANGE 3: Updated default values to match your means
                        N_1050_real=3.8642, N_1050_imag=0.4279,
                        N_1178_real=3.9774, N_1178_imag=0.4498,
                        N_sub_real=3.7076, N_sub_imag=0.4138):
    """Calculate reflectance simulation with given parameters."""  # <<<< CHANGE 4: Added docstring
    
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
#df_log[REFLECTANCE_COLUMN] = df_log[REFLECTANCE_COLUMN] / (0.8561)
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

# <<<< CHANGE 5: Added data smoothing section - SMOOTH ONLY FIRST OCCURRENCES
# ====================================================================
# SMOOTHING SECTION - Apply smoothing to first occurrence points only
# ====================================================================
print("\n" + "="*70)
print("DATA SMOOTHING (First Occurrences Only)")
print("="*70)

# Store unsmoothed first occurrences for comparison
first_occurrences['unsmoothed_reflectance'] = first_occurrences[REFLECTANCE_COLUMN].copy()

if ENABLE_SMOOTHING:
    print(f"Smoothing method: {SMOOTHING_METHOD.upper()}")
    print(f"Number of first occurrence points: {len(first_occurrences)}")
    
    if SMOOTHING_METHOD.lower() == 'savgol':
        # Savitzky-Golay filter
        print(f"Applying Savitzky-Golay filter:")
        print(f"  Window length: {SAVGOL_WINDOW}")
        print(f"  Polynomial order: {SAVGOL_POLYORDER}")
        
        # Check if we have enough points
        if len(first_occurrences) < SAVGOL_WINDOW:
            print(f"[WARNING] Not enough points ({len(first_occurrences)}) for window size ({SAVGOL_WINDOW})")
            print(f"          Reducing window to {len(first_occurrences) - (1 if len(first_occurrences) % 2 == 0 else 0)}")
            actual_window = len(first_occurrences) - (1 if len(first_occurrences) % 2 == 0 else 0)
            if actual_window < SAVGOL_POLYORDER + 2:
                print(f"[ERROR] Still not enough points. Disabling smoothing.")
                ENABLE_SMOOTHING = False
            else:
                smoothed_data = savgol_filter(
                    first_occurrences[REFLECTANCE_COLUMN].values, 
                    window_length=actual_window, 
                    polyorder=SAVGOL_POLYORDER
                )
                first_occurrences[REFLECTANCE_COLUMN] = smoothed_data
        else:
            smoothed_data = savgol_filter(
                first_occurrences[REFLECTANCE_COLUMN].values, 
                window_length=SAVGOL_WINDOW, 
                polyorder=SAVGOL_POLYORDER
            )
            first_occurrences[REFLECTANCE_COLUMN] = smoothed_data
        
    elif SMOOTHING_METHOD.lower() == 'ewma':
        # Exponential Weighted Moving Average
        print(f"Applying EWMA (Exponential Weighted Moving Average):")
        print(f"  Span: {EWMA_SPAN}")
        print(f"  Note: EWMA introduces forward lag and may shift peaks")
        
        smoothed_data = first_occurrences[REFLECTANCE_COLUMN].ewm(span=EWMA_SPAN, adjust=False).mean()
        first_occurrences[REFLECTANCE_COLUMN] = smoothed_data
        
    else:
        print(f"[ERROR] Unknown smoothing method: '{SMOOTHING_METHOD}'")
        print(f"        Valid options: 'savgol' or 'ewma'")
        print(f"        Proceeding without smoothing...")
        ENABLE_SMOOTHING = False
    
    if ENABLE_SMOOTHING:
        # Calculate smoothing statistics
        difference = first_occurrences['unsmoothed_reflectance'] - first_occurrences[REFLECTANCE_COLUMN]
        print(f"\n[OK] Smoothing applied to {len(first_occurrences)} first occurrence points")
        print(f"     Mean absolute difference: {np.abs(difference).mean():.4f}%")
        print(f"     Max absolute difference: {np.abs(difference).max():.4f}%")
        print(f"     RMS difference: {np.sqrt(np.mean(difference**2)):.4f}%")
        
        # Update the main dataframe with smoothed first occurrence values
        df_log.loc[first_occurrences.index, REFLECTANCE_COLUMN] = first_occurrences[REFLECTANCE_COLUMN].values
else:
    print("[INFO] Smoothing disabled (ENABLE_SMOOTHING = False)")

print("="*70)
# ====================================================================

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

print(f"\nMeasured Data (Smoothed):")  # <<<< CHANGE 7: Updated label
print(f"  Min: {df_log[REFLECTANCE_COLUMN].min():.4f}%")
print(f"  Max: {df_log[REFLECTANCE_COLUMN].max():.4f}%")
print(f"  Range: {df_log[REFLECTANCE_COLUMN].max() - df_log[REFLECTANCE_COLUMN].min():.4f}%")
print(f"  Mean: {df_log[REFLECTANCE_COLUMN].mean():.4f}%")

# <<<< CHANGE 8: Enhanced visualization with smoothed and unsmoothed data
# --- Visualization ---
fitting_points_plot = first_occurrences[~first_occurrences['exclude_from_fitting']]

fig = go.Figure()

# All raw data points (not first occurrences)
fig.add_trace(go.Scatter(
    x=df_log.index, 
    y=df_log[REFLECTANCE_COLUMN],
    mode='markers', 
    name='All measured data', 
    marker=dict(color='lightgrey', size=3, opacity=0.3),
    hovertext=df_log['Message'], 
    hovertemplate='%{hovertext}<br>Reflectance: %{y:.4f}%<extra></extra>'
))

# Unsmoothed first occurrences (if smoothing was applied)
if ENABLE_SMOOTHING and 'unsmoothed_reflectance' in first_occurrences.columns:
    fig.add_trace(go.Scatter(
        x=first_occurrences.index, 
        y=first_occurrences['unsmoothed_reflectance'],
        mode='markers', 
        name='First occurrence (unsmoothed)', 
        marker=dict(color='grey', size=5, opacity=0.6),
        hovertext=first_occurrences['Message'], 
        hovertemplate='%{hovertext}<br>Reflectance: %{y:.4f}%<extra></extra>'
    ))

# Smoothed first occurrences (or just first occurrences if no smoothing)
marker_name = 'First occurrence (smoothed)' if ENABLE_SMOOTHING else 'First occurrence'
fig.add_trace(go.Scatter(
    x=first_occurrences.index, 
    y=first_occurrences[REFLECTANCE_COLUMN],
    mode='markers',
    name=marker_name, 
    marker=dict(color='darkgrey', size=6),
    hovertext=first_occurrences['Message'], 
    hovertemplate='%{hovertext}<br>Reflectance: %{y:.4f}%<extra></extra>'
))

# Initial simulation
fig.add_trace(go.Scatter(
    x=df_log.index, 
    y=initial_simulation,
    mode='lines', 
    name='Initial Simulation',
    line=dict(color='blue', width=2, dash='dash')
))

# Optimized fit
fig.add_trace(go.Scatter(
    x=df_log.index, 
    y=optimized_reflectance,
    mode='lines', 
    name='Optimized Fit', 
    line=dict(color='red', width=2)
))

# Fitting data points
fig.add_trace(go.Scatter(
    x=fitting_points_plot.index,
    y=fitting_points_plot[REFLECTANCE_COLUMN],
    mode='markers', 
    marker=dict(size=8, color='green', symbol='circle'),
    name='Fitting Points', 
    hovertext=fitting_points_plot['Message'],
    hovertemplate='%{hovertext}<br>Reflectance: %{y:.4f}%<extra></extra>'
))

# Interface points (excluded)
if len(interface_excluded) > 0:
    fig.add_trace(go.Scatter(
        x=interface_excluded.index,
        y=interface_excluded[REFLECTANCE_COLUMN],
        mode='markers', 
        marker=dict(size=10, color='red', symbol='x'),
        name='Interface Points (Excluded)', 
        hovertext=interface_excluded['Message'],
        hovertemplate='%{hovertext}<br>Reflectance: %{y:.4f}%<extra></extra>'
    ))

# Add transition lines
colors = ["orange", "cyan", "magenta", "purple", "brown"]
for i, (idx, row) in enumerate(interface_points.iterrows()):
    color = colors[i % len(colors)]
    prev_msg = first_occurrences.loc[first_occurrences.index < idx, 'Message'].iloc[-1] if len(first_occurrences.loc[first_occurrences.index < idx]) > 0 else "Start"
    curr_msg = row['Message']

    # Create transition label
    annotation_text = f"{prev_msg} -> {curr_msg}"
    annotation_position = "top left" if i % 2 == 0 else "top right"

    fig.add_vline(x=idx, line_dash="dash", line_color=color,
                  annotation_text=annotation_text,
                  annotation_position=annotation_position)

# <<<< CHANGE 9: Updated title to indicate smoothing method
if ENABLE_SMOOTHING:
    if SMOOTHING_METHOD.lower() == 'savgol':
        smooth_status = f" (Savgol: window={SAVGOL_WINDOW}, poly={SAVGOL_POLYORDER})"
    elif SMOOTHING_METHOD.lower() == 'ewma':
        smooth_status = f" (EWMA: span={EWMA_SPAN})"
    else:
        smooth_status = " (Smoothing error)"
else:
    smooth_status = " (No smoothing)"
    
fig.update_layout(
    width=1400, height=600, 
    xaxis_title='Time Index', 
    yaxis_title='Reflectance (%)',
    title='9-Parameter Fitting with Boundary Check' + smooth_status,
    legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top')
)
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
# <<<< CHANGE 10: Updated initial values to match fitted defaults
initial_vals = {'GR_1050': 0.4664, 'GR_1178': 0.5135, 'GR_InP': 0.4246,
                'N_1050_real': 3.8642, 'N_1050_imag': 0.4279,
                'N_1178_real': 3.9774, 'N_1178_imag': 0.4498,
                'N_sub_real': 3.7076, 'N_sub_imag': 0.4138}

for name, initial_val in initial_vals.items():
    if result.params[name].vary:
        final_val = result.params[name].value
        change = ((final_val - initial_val) / initial_val) * 100
        print(f"  {name}: {initial_val:.4f} -> {final_val:.4f} ({change:+.1f}%)")

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

# <<<< CHANGE 11: Added smoothing info to results dictionary
# Create a dictionary with all results
results_dict = {
    # Normalization and basic metrics
    'normalization_fact': norm_ratio,
    'Rsquare': R2_optimized,
    'NRMSE': NRMSE_optimized,
    
    # Smoothing information
    'smoothing_enabled': ENABLE_SMOOTHING,
    'smoothing_method': SMOOTHING_METHOD if ENABLE_SMOOTHING else None,
    'savgol_window': SAVGOL_WINDOW if (ENABLE_SMOOTHING and SMOOTHING_METHOD.lower() == 'savgol') else None,
    'savgol_polyorder': SAVGOL_POLYORDER if (ENABLE_SMOOTHING and SMOOTHING_METHOD.lower() == 'savgol') else None,
    'ewma_span': EWMA_SPAN if (ENABLE_SMOOTHING and SMOOTHING_METHOD.lower() == 'ewma') else None,

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
    
    # Boundary check flags
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