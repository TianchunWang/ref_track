# 11-parameter fitting using lmfit
# ADDED: N_sub_real_0 and N_sub_imag_0 as separate parameters for scaling factor
# ADDED: Curvature-based oscillation weighting
# ADDED: INITIAL POINT PENALTY - Heavy weight on first point to force exact match
# ADDED: Comprehensive boundary checking and analysis

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lmfit import Parameters, minimize

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Column name
REFLECTANCE_COLUMN = 'EpiReflect1_9.Current Value'

# Weighting configuration
ENABLE_OSCILLATION_WEIGHTING = True
OSCILLATION_WEIGHT = 1.0      # Weight multiplier for high-curvature regions (peaks/troughs)
INTERFACE_WEIGHT = 1.0      # Weight for interface points
REGULAR_WEIGHT = 1.0           # Base weight for regular points
INITIAL_POINT_WEIGHT = 1.0  # Heavy weight for the very first point (NEW!)
APPLY_INTERFACE_TO_NEIGHBORS = False

# Constants
WAVELENGTH = 633.0e-9  # WG (633 nm laser)
C = 2.998e8            # Speed of light (m/s)
PI = np.pi
eta_0 = 1 / C          

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

# Read layer and read log data
df_layer = pd.read_csv('layers.csv', index_col='index')
df_log = pd.read_csv('2025.2025-05.C2720_HL13B5-WG_L2_C2720_HL13B5-WG_L2_2912.csv')

def preprocess_log_data(df_log, reflectance_column):
    """
    Correct message naming inconsistencies and filter log data.
    """
    print("\n" + "="*70)
    print("MESSAGE CORRECTION")
    print("="*70)
    
    # Find the trigger message that indicates the second 1050nm period
    trigger_msg = 'end 1170nm InGaAsP, set flow for 1050nm InGaAsP'
    trigger_indices = df_log[df_log['Message'] == trigger_msg].index
    
    if len(trigger_indices) > 0:
        trigger_idx = trigger_indices[0]
        print(f"Found trigger message at index {trigger_idx}: '{trigger_msg}'")
        
        # Find all "1050 nm InGaAsP" messages after the trigger and correct them
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
    for msg in ['introduce AsH3', '1050 nm InGaAsP', '1178nm InGaAsP', 
                '1050nm InGaAsP', 'InP cap']:
        count = (df_log['Message'] == msg).sum()
        if count > 0:
            print(f"  '{msg}': {count} occurrences")
    
    # Filter log data by message
    msg_list = ['introduce AsH3', '1050 nm InGaAsP', '1178nm InGaAsP', 
                '1050nm InGaAsP', 'InP cap']
    col_list = ['Time (rel)', reflectance_column, 'Message']
    
    df_filtered = df_log[df_log['Message'].isin(msg_list)][col_list].reset_index(drop=True)
    
    print(f"\nFiltered data: {len(df_filtered)} rows retained")
    print("="*70)
    
    return df_filtered

df_log = preprocess_log_data(df_log, REFLECTANCE_COLUMN)

# Store original raw data statistics BEFORE any normalization
raw_data_original = df_log[REFLECTANCE_COLUMN].copy()
raw_max_original = raw_data_original.max()
raw_min_original = raw_data_original.min()
raw_first_original = raw_data_original.iloc[0]
raw_mean_original = raw_data_original.mean()

# ============================================================================
# SIMULATION FUNCTION
# ============================================================================

def simulate_reflectance(GR_1050=0.4664, GR_1178=0.5135, GR_InP=0.4246,
                        N_1050_real=3.8642, N_1050_imag=0.4279,
                        N_1178_real=3.9774, N_1178_imag=0.4498,
                        N_sub_real=3.7076, N_sub_imag=0.4138,
                        N_sub_real_0=3.7076, N_sub_imag_0=0.4138):
    """
    Simulate optical reflectance during epitaxial growth using transfer matrix method.
    
    Parameters:
    -----------
    GR_1050, GR_1178, GR_InP : float
        Growth rates in nm/s for different materials
    N_1050_real, N_1050_imag : float
        Complex refractive index (n - iκ) for 1050nm InGaAsP
    N_1178_real, N_1178_imag : float
        Complex refractive index (n - iκ) for 1178nm InGaAsP
    N_sub_real, N_sub_imag : float
        Complex refractive index (n - iκ) for InP substrate during growth
    N_sub_real_0, N_sub_imag_0 : float
        Complex refractive index (n - iκ) for InP substrate at t=0 (sets scaling)
    
    Returns:
    --------
    reflectance_values : array
        Simulated reflectance (%) at each time point
    
    Notes:
    ------
    The N_sub_real_0 and N_sub_imag_0 parameters control the absolute scaling factor
    by determining the initial reflectance value. This separates the scaling factor
    from the InP properties during growth, reducing parameter correlation.
    """
    
    M = np.eye(2, dtype=complex)
    reflectance_values = []
    
    for idx in range(len(df_log)):
        msg = df_log.loc[idx, 'Message']

        # Select material properties based on current growth phase
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
            # N = 1.0 represents air (refractive index ≈ 1.0)
            N_j = 1.0 + 0.0j
            GR_j = 0.0

        # Calculate layer thickness (1 second time step)
        d = GR_j * 1.0e-9  # Convert nm/s to m
        
        # Calculate optical phase change
        delta_j = 2 * PI * N_j * d / WAVELENGTH
        
        # Optical admittance
        eta_j = N_j / C
        
        # Substrate admittance (uses N_sub_real_0, N_sub_imag_0 to set scaling)
        eta_m = (N_sub_real_0 - 1j * N_sub_imag_0) / C

        # Build transfer matrix for this layer
        M_j = np.array([
            [np.cos(delta_j), 1j * np.sin(delta_j) / eta_j],
            [1j * eta_j * np.sin(delta_j), np.cos(delta_j)]
        ])
        
        # Accumulate layers
        M = M_j @ M

        BC = M @ np.array([1, eta_m])
        rho = (BC[0] * eta_0 - BC[1]) / (BC[0] * eta_0 + BC[1])

        # Calculate reflectance (intensity)
        R = np.vdot(rho, rho).real * 100  # Convert to percentage

        reflectance_values.append(R)

    return np.array(reflectance_values)

print("\nCalculating initial simulation with default parameters...")
initial_simulation = simulate_reflectance()

# Calculate initial normalization estimate for reference !!!
scaling_estimate = raw_first_original / initial_simulation[0]
print(f"Initial normalization estimate: {scaling_estimate:.6f}")

# Apply normalization to measured data (for consistent scaling during fitting)
df_log[REFLECTANCE_COLUMN] = df_log[REFLECTANCE_COLUMN] / scaling_estimate

# ============================================================================
# IDENTIFY FIRST OCCURRENCES AND INTERFACE POINTS
# ============================================================================

df_log['is_first_occurrence'] = df_log[REFLECTANCE_COLUMN] != df_log[REFLECTANCE_COLUMN].shift(1)
first_occurrences = df_log[df_log['is_first_occurrence']].copy()

# Mark interface points (message changes)
first_occurrences['message_changed'] = first_occurrences['Message'] != first_occurrences['Message'].shift(1)
first_occurrences.iloc[0, first_occurrences.columns.get_loc('message_changed')] = False

# Get points for visualization
interface_points = first_occurrences[first_occurrences['message_changed']].copy()

# ============================================================================
# CURVATURE-BASED OSCILLATION WEIGHTING FUNCTION
# ============================================================================

def calculate_curvature_weights(measured_values, curvature_weight=100.0):
    """
    Calculate weights based on curvature (second derivative).
    High curvature regions (peaks and troughs) get higher weights.
    
    Parameters:
    -----------
    measured_values : array
        Measured reflectance values
    curvature_weight : float
        Maximum weight for highest curvature regions
    
    Returns:
    --------
    weights : array
        Weight for each point (1.0 to curvature_weight)
    abs_curvature : array
        Absolute curvature values for analysis
    """
    
    # Calculate first derivative (gradient)
    first_deriv = np.gradient(measured_values)
    
    # Calculate second derivative (curvature)
    second_deriv = np.gradient(first_deriv)
    
    # Take absolute value
    abs_curvature = np.abs(second_deriv)
    
    # Normalize to [0, 1]
    if abs_curvature.max() > 0:
        normalized_curvature = abs_curvature / abs_curvature.max()
    else:
        normalized_curvature = abs_curvature
    
    # Convert to weights: 1.0 (flat regions) to curvature_weight (high curvature)
    weights = 1.0 + normalized_curvature * (curvature_weight - 1.0)
    
    return weights, abs_curvature

# ============================================================================
# 🆕 NEW: ENHANCED INTERFACE WEIGHTING FUNCTION
# ============================================================================

def apply_interface_weights_with_neighbors(weights, fitting_points, interface_weight):
    """
    🆕 NEW FUNCTION
    Apply interface weighting to interface points and their immediate neighbors.
    
    Parameters:
    -----------
    weights : array
        Current weight array
    fitting_points : DataFrame
        Fitting points data with 'message_changed' column
    interface_weight : float
        Weight to apply to interface points and neighbors
    
    Returns:
    --------
    weights : array
        Modified weight array
    interface_indices : list
        List of indices that received interface weighting
    """
    
    # Find interface points
    interface_mask = fitting_points['message_changed'].values
    interface_positions = np.where(interface_mask)[0]
    
    # Track which indices get interface weight
    interface_indices = []
    
    for pos in interface_positions:
        # Apply weight to interface point itself
        weights[pos] = interface_weight
        interface_indices.append(pos)
        
        if APPLY_INTERFACE_TO_NEIGHBORS:  # 🆕 NEW: Conditional neighbor weighting
            # Apply to previous neighbor if it exists
            if pos > 0:
                weights[pos - 1] = interface_weight
                interface_indices.append(pos - 1)
            
            # Apply to next neighbor if it exists
            if pos < len(weights) - 1:
                weights[pos + 1] = interface_weight
                interface_indices.append(pos + 1)
    
    # Remove duplicates and sort
    interface_indices = sorted(list(set(interface_indices)))
    
    return weights, interface_indices

# ============================================================================
# OBJECTIVE FUNCTION WITH INITIAL POINT PENALTY
# ============================================================================

# ============================================================================
# ✏️ MODIFIED: OBJECTIVE FUNCTION WITH ENHANCED INTERFACE WEIGHTING
# ============================================================================

def lmfit_objective(params):
    """
    Objective function for optimization with weighted residuals.
    
    Applies weighting in priority order:
    1. INITIAL POINT PENALTY: Very heavy weight on first point (index 0) - HIGHEST
    2. Curvature-based weighting: Higher weight for peaks/troughs
    3. ✏️ MODIFIED: ENHANCED Interface weighting: Interface points AND their neighbors - LOWEST
    """
    simulated = simulate_reflectance(
        params['GR_1050'].value, 
        params['GR_1178'].value, 
        params['GR_InP'].value,
        params['N_1050_real'].value, 
        params['N_1050_imag'].value,
        params['N_1178_real'].value, 
        params['N_1178_imag'].value,
        params['N_sub_real'].value, 
        params['N_sub_imag'].value,
        params['N_sub_real_0'].value,
        params['N_sub_imag_0'].value,
    )

    fitting_points = first_occurrences
    measured = fitting_points[REFLECTANCE_COLUMN].values
    simulated_fit = simulated[fitting_points.index.values]
    
    residuals = simulated_fit - measured
    
    # Initialize weights with regular weight
    weights = np.ones(len(fitting_points)) * REGULAR_WEIGHT
    
    # Apply curvature-based oscillation weighting (first pass)
    if ENABLE_OSCILLATION_WEIGHTING:
        curvature_weights, _ = calculate_curvature_weights(measured, OSCILLATION_WEIGHT)
        weights *= curvature_weights
    
    # 🆕 NEW: Apply ENHANCED interface weighting (second pass - can override curvature)
    # ✏️ MODIFIED: Changed from simple mask to function call with neighbor support
    weights, interface_indices = apply_interface_weights_with_neighbors(
        weights, fitting_points, INTERFACE_WEIGHT
    )
    
    # Apply INITIAL POINT PENALTY (final pass - highest priority, overrides everything)
    weights[0] = max(weights[0], INITIAL_POINT_WEIGHT)
    
    # Apply weights to residuals (sqrt because leastsq squares the residuals)
    weighted_residuals = residuals * np.sqrt(weights)
    
    return weighted_residuals

def scale_to_unit(value, bounds):
    """Scale parameter from original range to [0, 1]"""
    min_val, max_val = bounds
    return (value - min_val) / (max_val - min_val)

def unscale_from_unit(scaled_value, bounds):
    """Unscale parameter from [0, 1] back to original range"""
    min_val, max_val = bounds
    return scaled_value * (max_val - min_val) + min_val

# ============================================================================
# SETUP PARAMETERS (11 PARAMETERS)
# ============================================================================
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
}

def lmfit_objective_standardized(params_scaled):
    """
    Objective function that works with standardized parameters.
    
    Parameters are in [0, 1] range, need to unscale before simulation.
    """
    
    # Unscale all parameters back to original values
    params_original = {}
    for name in param_bounds.keys():
        scaled_val = params_scaled[name].value
        original_val = unscale_from_unit(scaled_val, param_bounds[name])
        params_original[name] = original_val
    
    # Run simulation with original-scale parameters
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
    
    # Calculate residuals (same as before)
    fitting_points = first_occurrences
    measured = fitting_points[REFLECTANCE_COLUMN].values
    simulated_fit = simulated[fitting_points.index.values]
    residuals = simulated_fit - measured
    
    # Apply weighting (same as before)
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

params = Parameters()

# Growth rates
params.add('GR_1050', value=0.4664, min=0.42, max=0.8)
params.add('GR_1178', value=0.5135, min=0.4, max=0.8)
params.add('GR_InP', value=0.4246, min=0.4, max=0.8)

# Refractive indices - 1050nm InGaAsP
params.add('N_1050_real', value=3.8642, min=3.5, max=4.2)
params.add('N_1050_imag', value=0.4279, min=0.2, max=0.9)

# Refractive indices - 1178nm InGaAsP
params.add('N_1178_real', value=3.9774, min=3.5, max=4.2)
params.add('N_1178_imag', value=0.4498, min=0.2, max=0.9)

# Refractive indices - InP substrate (during growth)
params.add('N_sub_real', value=3.7076, min=3.5, max=4.2)
params.add('N_sub_imag', value=0.4138, min=0.2, max=0.9)

# Refractive indices - InP substrate at t=0 (controls scaling factor)
params.add('N_sub_real_0', value=3.7076, min=3.5, max=4.2)
params.add('N_sub_imag_0', value=0.4138, min=0.2, max=0.9)

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

print("\n" + "="*70)
print("FITTING CONFIGURATION")
print("="*70)
print(f"Total parameters: {len([p for p in params.values() if p.vary])}")
print(f"  Growth rates: 3 (GR_1050, GR_1178, GR_InP)")
print(f"  Refractive indices (layers): 4 (N_1050, N_1178)")
print(f"  Refractive indices (InP substrate): 2 (N_sub)")
print(f"  Refractive indices (InP at t=0): 2 (N_sub_0) - controls scaling")
print("="*70)

print("\n" + "="*70)
print("WEIGHTING CONFIGURATION")
print("="*70)
print(f"🔴 INITIAL POINT PENALTY: {INITIAL_POINT_WEIGHT:.1f}x (NEW!)")
print(f"   → Forces first point to match exactly")
print(f"\nOscillation weighting: {'ENABLED' if ENABLE_OSCILLATION_WEIGHTING else 'DISABLED'}")
if ENABLE_OSCILLATION_WEIGHTING:
    print(f"  Method: Curvature-based (second derivative)")
    print(f"  Oscillation weight: {OSCILLATION_WEIGHT:.1f}x")
print(f"\nInterface weight: {INTERFACE_WEIGHT:.2f}x")
print(f"Regular weight: {REGULAR_WEIGHT:.1f}x")

fitting_points_config = first_occurrences
n_interface = fitting_points_config['message_changed'].sum()
n_regular = len(fitting_points_config) - n_interface

print(f"\nPoint distribution:")
print(f"  Initial point: 1 (weight: {INITIAL_POINT_WEIGHT:.0f}x)")
print(f"  Interface points: {n_interface} (weight: {INTERFACE_WEIGHT:.1f}x)")
print(f"  Regular points: {n_regular} (weight: {REGULAR_WEIGHT:.1f}x)")
print(f"  Total fitting points: {len(fitting_points_config)}")
print("="*70)

# ============================================================================
# PERFORM FITTING
# ============================================================================

print("\nStarting optimization with 11 parameters...")
print("Method: BFGS (gradient-based quasi-Newton)")
result = minimize(lmfit_objective, params, method='bfgs')

print("\n" + "="*50)
print("FITTING RESULTS")
print("="*50)
print(result.params.pretty_print())

print(f"\nOptimization summary:")
print(f"  Success: {result.success}")
print(f"  Function evaluations: {result.nfev}")
print(f"  Message: {result.message}")

# ============================================================================
# INITIAL POINT MATCHING ANALYSIS (NEW!)
# ============================================================================

print("\n" + "="*70)
print("INITIAL POINT MATCHING ANALYSIS")
print("="*70)

# Generate optimized reflectance
optimized_reflectance = simulate_reflectance(
    result.params['GR_1050'].value, 
    result.params['GR_1178'].value, 
    result.params['GR_InP'].value,
    result.params['N_1050_real'].value, 
    result.params['N_1050_imag'].value,
    result.params['N_1178_real'].value, 
    result.params['N_1178_imag'].value,
    result.params['N_sub_real'].value, 
    result.params['N_sub_imag'].value,
    result.params['N_sub_real_0'].value,
    result.params['N_sub_imag_0'].value
)

# Check initial point match
measured_first = first_occurrences[REFLECTANCE_COLUMN].iloc[0]
simulated_first = optimized_reflectance[first_occurrences.index[0]]
initial_error = simulated_first - measured_first
initial_error_pct = (initial_error / measured_first) * 100

print(f"First point comparison:")
print(f"  Measured:  {measured_first:.6f}%")
print(f"  Simulated: {simulated_first:.6f}%")
print(f"  Error:     {initial_error:+.6f}% ({initial_error_pct:+.4f}% relative)")
print(f"  Weight applied: {INITIAL_POINT_WEIGHT:.0f}x")

if abs(initial_error_pct) < 0.01:
    print(f"  ✓ Excellent match! (< 0.01% relative error)")
elif abs(initial_error_pct) < 0.1:
    print(f"  ✓ Very good match (< 0.1% relative error)")
elif abs(initial_error_pct) < 1.0:
    print(f"  ✓ Good match (< 1% relative error)")
else:
    print(f"  ⚠ Warning: Significant mismatch at initial point")
    print(f"     Consider increasing INITIAL_POINT_WEIGHT")

print("="*70)

# ============================================================================
# SCALING FACTOR ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("SCALING FACTOR ANALYSIS")
print("="*70)

# Show changes
n_real_0_change = ((result.params['N_sub_real_0'].value - params['N_sub_real_0'].value) / 
                   params['N_sub_real_0'].value) * 100
n_imag_0_change = ((result.params['N_sub_imag_0'].value - params['N_sub_imag_0'].value) / 
                   params['N_sub_imag_0'].value) * 100

print(f"Optimized substrate properties at t=0:")
print(f"  N_sub_real_0 = {result.params['N_sub_real_0'].value:.6f} ({n_real_0_change:+.2f}%)")
print(f"  N_sub_imag_0 = {result.params['N_sub_imag_0'].value:.6f} ({n_imag_0_change:+.2f}%)")

print(f"\nSubstrate properties during growth:")
print(f"  N_sub_real = {result.params['N_sub_real'].value:.6f}")
print(f"  N_sub_imag = {result.params['N_sub_imag'].value:.6f}")

print(f"\nDifference (growth - t=0):")
diff_real = result.params['N_sub_real'].value - result.params['N_sub_real_0'].value
diff_imag = result.params['N_sub_imag'].value - result.params['N_sub_imag_0'].value
print(f"  Δn (real) = {diff_real:+.6f}")
print(f"  Δκ (imag) = {diff_imag:+.6f}")
print("="*70)

# ============================================================================
# BOUNDARY CHECK
# ============================================================================

def check_parameter_boundaries(result_params, boundary_tolerance=1e-6, warning_threshold=5.0):
    """Check if fitted parameters are at or near their boundary limits."""
    print("\n" + "="*50)
    print("BOUNDARY CHECK")
    print("="*50)

    params_at_boundary = []
    
    for name, param in result_params.items():
        if param.vary:
            at_min = abs(param.value - param.min) < boundary_tolerance
            at_max = abs(param.value - param.max) < boundary_tolerance
            
            param_range = param.max - param.min
            distance_from_min = (param.value - param.min) / param_range * 100
            distance_from_max = (param.max - param.value) / param_range * 100
            
            if at_min or at_max:
                params_at_boundary.append(name)
                if at_min:
                    print(f"[!] {name:20s} HIT LOWER BOUND: {param.value:.6f} (min={param.min:.6f})")
                if at_max:
                    print(f"[!] {name:20s} HIT UPPER BOUND: {param.value:.6f} (max={param.max:.6f})")
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
            param_range = param.max - param.min
            pct_from_min = (param.value - param.min) / param_range * 100
            pct_from_max = (param.max - param.value) / param_range * 100
            
            warning = ""
            if abs(param.value - param.min) < boundary_tolerance:
                warning = "[!] MIN"
            elif abs(param.value - param.max) < boundary_tolerance:
                warning = "[!] MAX"
            elif pct_from_min < warning_threshold:
                warning = "[~] ~MIN"
            elif pct_from_max < warning_threshold:
                warning = "[~] ~MAX"
            
            print(f"{name:<20s} {param.value:<12.6f} {param.min:<12.6f} {param.max:<12.6f} "
                  f"{pct_from_min:<8.1f} {pct_from_max:<8.1f} {warning}")
    
    print("-"*80)
    print("Legend: [!] = At boundary limit, [~] = Within 5% of boundary")
    print("        %Min = Distance from minimum (%), %Max = Distance from maximum (%)")
    print("="*50)
    
    return params_at_boundary

params_at_boundary = check_parameter_boundaries(result.params)

# Print statistics
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
print(f"  First point: {optimized_reflectance[first_occurrences.index[0]]:.6f}%")

print(f"\nMeasured Data (normalized):")
print(f"  Min: {df_log[REFLECTANCE_COLUMN].min():.4f}%")
print(f"  Max: {df_log[REFLECTANCE_COLUMN].max():.4f}%")
print(f"  Range: {df_log[REFLECTANCE_COLUMN].max() - df_log[REFLECTANCE_COLUMN].min():.4f}%")
print(f"  Mean: {df_log[REFLECTANCE_COLUMN].mean():.4f}%")
print(f"  First point: {first_occurrences[REFLECTANCE_COLUMN].iloc[0]:.6f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================

fitting_points_plot = first_occurrences

# Calculate curvature weights for visualization
if ENABLE_OSCILLATION_WEIGHTING:
    vis_curv_weights, vis_abs_curvature = calculate_curvature_weights(
        fitting_points_plot[REFLECTANCE_COLUMN].values, OSCILLATION_WEIGHT
    )
    high_curv_threshold_vis = 1.0 + (OSCILLATION_WEIGHT - 1.0) * 0.5
    high_curv_points = fitting_points_plot[vis_curv_weights > high_curv_threshold_vis]

fig = go.Figure()

# All measured data
fig.add_trace(go.Scatter(
    x=df_log.index, 
    y=df_log[REFLECTANCE_COLUMN],
    mode='markers', 
    name='Measured (normalized)', 
    marker=dict(color='lightgrey', size=4),
    hovertext=df_log['Message'], 
    hovertemplate='%{hovertext}<br>Reflectance: %{y:.4f}%<extra></extra>'
))

# Optimized fit
fig.add_trace(go.Scatter(
    x=df_log.index, 
    y=optimized_reflectance,
    mode='lines', 
    name='Optimized Fit (11-param)', 
    line=dict(color='red', width=2)
))

# Regular fitting points
fig.add_trace(go.Scatter(
    x=fitting_points_plot.index,
    y=fitting_points_plot[REFLECTANCE_COLUMN],
    mode='markers', 
    marker=dict(size=8, color='green', symbol='circle'),
    name='Fitting Data Points', 
    hovertext=fitting_points_plot['Message'],
    hovertemplate='%{hovertext}<br>Reflectance: %{y:.4f}%<extra></extra>'
))

# NEW: Highlight initial point with special marker
initial_point_idx = first_occurrences.index[0]
initial_point_val = first_occurrences[REFLECTANCE_COLUMN].iloc[0]
fig.add_trace(go.Scatter(
    x=[initial_point_idx],
    y=[initial_point_val],
    mode='markers',
    marker=dict(size=16, color='red', symbol='star', 
               line=dict(width=2, color='darkred')),
    name=f'Initial Point (weight={INITIAL_POINT_WEIGHT:.0f}x)',
    hovertemplate=f'Initial Point<br>Reflectance: {initial_point_val:.4f}%<extra></extra>'
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
        hovertemplate='%{hovertext}<br>Reflectance: %{y:.4f}%<extra></extra>'
    ))

# Add transition lines
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

# Update layout
title_text = f'11-Parameter Fitting with Initial Point Penalty ({INITIAL_POINT_WEIGHT:.0f}x)'
fig.update_layout(
    width=1400, height=600, 
    xaxis_title='Time Index', 
    yaxis_title='Reflectance (%)',
    title=title_text,
    legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top')
)
fig.show()

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

print(f"\n" + "="*50)
print("FITTING PERFORMANCE METRICS")
print("="*50)

fitting_points = first_occurrences
measured = fitting_points[REFLECTANCE_COLUMN].values
predicted_initial = initial_simulation[fitting_points.index.values]
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
print("PARAMETER CHANGES (from initial values)")
print("="*50)
initial_vals = {
    'GR_1050': 0.4664, 'GR_1178': 0.5135, 'GR_InP': 0.4246,
    'N_1050_real': 3.8642, 'N_1050_imag': 0.4279,
    'N_1178_real': 3.9774, 'N_1178_imag': 0.4498,
    'N_sub_real': 3.7076, 'N_sub_imag': 0.4138,
    'N_sub_real_0': 3.7076, 'N_sub_imag_0': 0.4138
}

for name, initial_val in initial_vals.items():
    if result.params[name].vary:
        final_val = result.params[name].value
        change = ((final_val - initial_val) / initial_val) * 100
        print(f"  {name}: {initial_val:.4f} → {final_val:.4f} ({change:+.1f}%)")

# ============================================================================
# EXPORT RESULTS TO CSV
# ============================================================================

print(f"\n" + "="*50)
print("EXPORTING RESULTS TO CSV")
print("="*50)

# Calculate period lengths
growth_periods = ['1050 nm InGaAsP', '1178nm InGaAsP', '1050nm InGaAsP', 'InP cap']
period_lengths = {}
for i, msg in enumerate(growth_periods):
    period_data = df_log[df_log['Message'] == msg]
    period_lengths[f'period_{i+1}'] = len(period_data)
    print(f"Period {i+1} ({msg}): {len(period_data)} data points")

# Create results dictionary
results_dict = {
    # Model configuration
    'n_parameters': 11,
    'optimization_method': 'bfgs',
    
    # Weighting configuration
    'initial_point_weight': INITIAL_POINT_WEIGHT,
    'initial_point_error': initial_error,
    'initial_point_error_pct': initial_error_pct,
    'oscillation_weighting_enabled': ENABLE_OSCILLATION_WEIGHTING,
    'oscillation_weight': OSCILLATION_WEIGHT if ENABLE_OSCILLATION_WEIGHTING else 0,
    'oscillation_method': 'curvature' if ENABLE_OSCILLATION_WEIGHTING else 'none',
    
    # Fitting quality metrics
    'Rsquare': R2_optimized,
    'Rsquare_adj': R2_adj_optimized,
    'RMSE': RMSE_optimized,
    'MAE': MAE_optimized,
    'NRMSE': NRMSE_optimized,
    
    # Optimization info
    'optimization_success': result.success,
    'n_function_evals': result.nfev,

    # Fitted growth rates
    '1050_growth_rate_final': result.params['GR_1050'].value,
    '1178_growth_rate_final': result.params['GR_1178'].value,
    'InP_growth_rate_final': result.params['GR_InP'].value,

    # Period lengths
    'period_1': period_lengths['period_1'],
    'period_2': period_lengths['period_2'],
    'period_3': period_lengths['period_3'],
    'period_4': period_lengths['period_4'],

    # Refractive indices - 1050nm InGaAsP
    '1050_real': result.params['N_1050_real'].value,
    '1050_imag': result.params['N_1050_imag'].value,
    
    # Refractive indices - 1178nm InGaAsP
    '1178_real': result.params['N_1178_real'].value,
    '1178_imag': result.params['N_1178_imag'].value,
    
    # Refractive indices - InP substrate (during growth)
    'InP_real': result.params['N_sub_real'].value,
    'InP_imag': result.params['N_sub_imag'].value,
    
    # Refractive indices - InP substrate at t=0 (scaling control)
    'InP_real_0': result.params['N_sub_real_0'].value,
    'InP_imag_0': result.params['N_sub_imag_0'].value,

    # Raw data statistics (original, before normalization)
    'raw_max': raw_max_original,
    'raw_min': raw_min_original,
    'raw_first': raw_first_original,
    'raw_mean': raw_mean_original,

    # Fitted data statistics
    'fit_max': optimized_reflectance.max(),
    'fit_min': optimized_reflectance.min(),
    'fit_first': optimized_reflectance[first_occurrences.index[0]],
    'fit_mean': optimized_reflectance.mean(),
    
    # Boundary check flags
    'params_at_boundary': len(params_at_boundary),
    'boundary_warning': ','.join(params_at_boundary) if params_at_boundary else 'None',
    
    # Weighting information
    'interface_weight': INTERFACE_WEIGHT,
    'n_interface_points': n_interface,
    'n_regular_points': n_regular
}

# Convert to DataFrame and save
results_df = pd.DataFrame([results_dict])
output_filename = 'fitting_results_11param_with_initial_penalty.csv'
results_df.to_csv(output_filename, index=False)
print(f"\n[OK] Results exported to: {output_filename}")
print(f"[OK] Total columns exported: {len(results_dict)}")
print(f"\nKey new columns:")
print(f"  - initial_point_weight: {INITIAL_POINT_WEIGHT:.0f}x")
print(f"  - initial_point_error: {initial_error:+.6f}%")
print(f"  - initial_point_error_pct: {initial_error_pct:+.4f}%")
print(f"  - Rsquare: {R2_optimized:.6f}")
print(f"  - NRMSE: {NRMSE_optimized:.4f}%")