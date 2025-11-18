import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def transfer_matrix(n, k, thickness, wavelength):
    """Calculate transfer matrix for a single layer"""
    n_complex = n + 1j*k
    beta = 2 * np.pi * n_complex * thickness / wavelength
    
    M11 = np.cos(beta)
    M12 = -1j * np.sin(beta) / n_complex
    M21 = -1j * n_complex * np.sin(beta)
    M22 = np.cos(beta)
    
    return np.array([[M11, M12], [M21, M22]])

def reflectance_simulation(n_layer, k_layer, n_substrate, growth_rate, 
                          wavelength, time_points):
    """Simulate reflectance vs time for growing layer"""
    reflectances = []
    
    for t in time_points:
        thickness = growth_rate * t
        
        # Transfer matrix for the growing layer
        M = transfer_matrix(n_layer, k_layer, thickness, wavelength)
        
        # Fresnel coefficients
        r01 = (1 - n_layer) / (1 + n_layer)  # air-layer interface
        r12 = (n_layer - n_substrate) / (n_layer + n_substrate)  # layer-substrate
        
        # Total reflection coefficient
        numerator = M[0,0] + M[0,1]*n_substrate - M[1,0] - M[1,1]*n_substrate
        denominator = M[0,0] + M[0,1]*n_substrate + M[1,0] + M[1,1]*n_substrate
        
        r_total = numerator / denominator
        R = abs(r_total)**2
        reflectances.append(R)
    
    return np.array(reflectances)

# Example simulation parameters
wavelength = 550e-9  # 550 nm
growth_rate = 1e-9   # 1 nm/s
n_layer = 2.0        # refractive index
k_layer = 0.01       # extinction coefficient
n_substrate = 1.5    

# Time points
t_max = 1000  # seconds
time = np.linspace(0, t_max, 1000)
thickness = growth_rate * time

# Calculate reflectance evolution
R = reflectance_simulation(n_layer, k_layer, n_substrate, growth_rate, wavelength, time)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(thickness*1e9, R)
plt.xlabel('Thickness (nm)')
plt.ylabel('Reflectance')
plt.title('Reflectance vs Layer Thickness')
plt.grid(True)
plt.savefig('reflectance_simulation.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'reflectance_simulation.png'")