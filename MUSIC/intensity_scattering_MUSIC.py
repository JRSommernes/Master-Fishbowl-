import numpy as np
try:
    import cupy as cp
except:
    pass
from time import time
from misc_functions import dyadic_green, high_inner, loadbar
import matplotlib.pyplot as plt

def free_space_green(sensors,emitter_pos,k_0):
    r_p = sensors-emitter_pos.reshape(3,1)
    R = np.sqrt(np.sum((r_p)**2,axis=0))
    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)

    E = np.array((g_R,np.zeros_like(g_R),np.zeros_like(g_R))).T
    return E

def scatter_MUSIC(I,sensors,emitters,N_recon,FoV,k_0,E_sensors):
    N_sensors = sensors.shape[1]

    x = np.linspace(FoV[0,0],FoV[0,1],N_recon)
    y = np.linspace(FoV[1,0],FoV[1,1],N_recon)
    z = np.linspace(FoV[2,0],FoV[2,1],N_recon)

    xx,yy = np.meshgrid(x,y)

    plane = np.array((xx.flatten(),yy.flatten(),np.zeros_like(xx.flatten())))

    K = xx.size
    M = np.shape(emitters)[0]

    A = np.zeros((M,K,3),dtype=np.complex128)
    for m in range(M):
        emitter_pos = emitters[m]
        tmp = free_space_green(plane,emitter_pos,k_0)
        A[m] = tmp

    LAM_n = np.zeros((N_sensors,K,3))
    for k in range(K):
        scatter_pos = plane[:,k]
        tmp = 
        exit()
