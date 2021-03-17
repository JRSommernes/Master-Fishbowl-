import numpy as np
from numba import jit

Mag = 60
N_sensors = 61**2

eps_0 = 8.8541878128e-12
mu_0 = 4*np.pi*10**-7
c_0 = 1/np.sqrt(eps_0*mu_0)

wl = 561e-9
freq = c_0/wl
k_0 = 2*np.pi*freq*np.sqrt(eps_0*mu_0)
omega = 2*np.pi*freq

N_cam = int(np.sqrt(N_sensors))
z_cam = 0
x_cam = y_cam = np.linspace(-3,3,N_cam)*wl*Mag

n_obj = 1.33
mur_obj = 1
epsr_obj = n_obj**2/mur_obj
k_obj = k_0*n_obj

mur_sub = 1
n_sub = 4.3
epsr_sub = n_sub**2/mur_sub
k_sub = k_0*n_sub
z_Interface_sub = -30e-9

mur_cam = 1
n_cam = 1
epsr_cam = n_cam**2/mur_cam
k_cam = k_0*n_cam

f_cam = 16e-2
f_obj = f_cam/(Mag*n_cam/n_obj)
NA = 1.2
theta_max = np.arcsin(NA/n_obj)

opt_ax_Mag = (n_cam/n_obj)*Mag**2


def kz_obj(theta_obj):
     return k_obj*np.cos(theta_obj)

def kz_sub(theta_obj):
     return np.sqrt(k_sub**2-(k_obj*np.sin(theta_obj))**2)

def RTE(theta_obj):
    return (kz_obj(theta_obj)/mur_obj-kz_sub(theta_obj)/mur_sub)/(kz_obj(theta_obj)/mur_obj+kz_sub(theta_obj)/mur_sub)*np.exp(-2j*kz_obj(theta_obj)*z_Interface_sub)

def Q(theta_obj):
    return (kz_obj(theta_obj)*k_sub**2*mur_obj)/(kz_sub(theta_obj)*k_obj**2*mur_sub)

def RTM(theta_obj):
    return (Q(theta_obj)-1)*np.exp(-2j*kz_obj(theta_obj)*z_Interface_sub)/(Q(theta_obj)+1)

def cosTheta_cam(theta_obj):
    return np.sqrt(1-((f_obj/f_cam)**2)*(np.sin(theta_obj)**2))

def sinTheta_cam(theta_obj):
    return f_obj*np.sin(theta_obj)/f_cam

def rho_x(theta_obj):
    return k_cam*sinTheta_cam(theta_obj)*x_cam \
                          - k_obj*np.sin(theta_obj)*x_dip

def rho_y(theta_obj):
    return k_cam*sinTheta_cam(theta_obj)*y_cam \
                          - k_obj*np.sin(theta_obj)*y_dip

def rho(theta_obj):
    return np.sqrt(rho_y(theta_obj)**2+rho_x(theta_obj)**2)

def psi(theta_obj):
    return np.arctan2(rho_y(theta_obj),rho_x(theta_obj))

def Zz(theta_obj):
    return k_cam*cosTheta_cam(theta_obj)*z_cam-k_obj*np.cos(theta_obj)*z_dip

def Z(theta_obj):
    return k_cam*cosTheta_cam(theta_obj)*z_cam+k_obj*np.cos(theta_obj)*z_dip
