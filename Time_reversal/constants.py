import numpy as np

eps_0 = 8.8541878128e-12
mu_0 = 4*np.pi*10**-7
c_0 = 1/np.sqrt(eps_0*mu_0)

lambda_0 = 690e-9
sensor_radius = 10*lambda_0

freq = c_0/lambda_0
k_0 = 2*np.pi*freq*np.sqrt(eps_0*mu_0)
omega = 2*np.pi*freq

# FoV = 4*lambda_0
# N_reconstruction = 100
# N_sensors = 300
