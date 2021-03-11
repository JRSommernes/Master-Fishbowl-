import numpy as np
from microscope import Microscope

class Imaging_microscope(Microscope):
    def __init__(self,N_sensors,N_emitters,microscope_radius,n_0,dipoles):
        Microscope.__init__(self,N_sensors,N_emitters,microscope_radius,n_0)
        self.dipoles = dipoles

    def scalar_incident_wave(self):
        r_p = self.dipoles-emitter_pos.reshape(3,1)
        R = np.sqrt(np.sum((r_p)**2,axis=0))

        g = np.exp(1j*k_0*R)
        return g

    def scalar_scattering(self):
        pass
