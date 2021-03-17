import numpy as np
from imaging import Scalar_Scattering_Imaging_Sysytem as SSIS
import matplotlib.pyplot as plt

N_sensors = 50
N_emitters = 100
n_0 = 1
n_1 = 1.33
wl = 690e-9
microscope_radius = 100*wl

dipoles = np.array([[1*wl,-1*wl,0*wl],[-1*wl,1*wl,0*wl]])

microscope = SSIS(N_sensors,N_emitters,microscope_radius,n_0,n_1,dipoles,wl)
microscope.make_sensors()
microscope.make_emitters()
microscope.field_calculation()

print(microscope.measurements.shape)
