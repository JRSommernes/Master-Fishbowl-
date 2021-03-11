import numpy as np
from imaging import Imaging_microscope
import matplotlib.pyplot as plt

N_sensors = 50
N_emitters = 100
n_0 = 1
wl = 690e-9
microscope_radius = 100*wl

dipoles = np.array([[1*wl,-1*wl,0*wl],[-1*wl,1*wl,0*wl]])

microscope = Imaging_microscope(N_sensors,N_emitters,microscope_radius,n_0,dipoles)
microscope.make_sensors()
microscope.make_emitters()
print(microscope.dipoles.shape)
