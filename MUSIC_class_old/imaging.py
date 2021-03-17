import numpy as np
from microscope import Microscope,Scattering_Microscope
from scipy.constants import mu_0,epsilon_0

class Scalar_Scattering_Imaging_Sysytem(Scattering_Microscope):
    def __init__(self,N_sensors,N_emitters,microscope_radius,n_0,n_1,dipoles,wl):
        Scattering_Microscope.__init__(self,N_sensors,N_emitters,microscope_radius,n_0)
        self.dipoles = dipoles
        self.c_0 = 1/np.sqrt(epsilon_0*mu_0)
        self.freq = self.c_0/wl
        self.k_0 = 2*np.pi*self.freq*np.sqrt(epsilon_0*mu_0)
        self.n_1 = n_1

    def scalar_incident(self):
        dist = self.dipoles.reshape(*self.dipoles.shape,1)-self.emitters.reshape(1,*self.emitters.shape)
        dist = np.sqrt(np.sum(dist**2,axis=1))

        self.incident_wave = np.exp(1j*self.k_0*dist)

    def scalar_scattering(self):
        dist = self.sensors.reshape(1,*self.sensors.shape)-self.dipoles.reshape(*self.dipoles.shape,1)
        dist = np.sqrt(np.sum(dist**2,axis=1))

        self.scattering_wave = np.exp(1j*self.k_0*dist)/(4*np.pi*dist)

    def field_calculation(self):
        self.scalar_incident()
        self.scalar_scattering()

        E_s_m = self.scattering_wave.T@self.incident_wave*(self.n_1-self.n_0)
        E_s_1_m = E_s_m[:,0].reshape(-1,1)+E_s_m[:,1:]
        E_s_1_im = E_s_m[:,0].reshape(-1,1)+1j*E_s_m[:,1:]

        self.imaged_wave = np.append(E_s_m,np.append(E_s_1_m,E_s_1_im,axis=1),axis=1)

        self.measurements = np.abs(self.imaged_wave)**2
