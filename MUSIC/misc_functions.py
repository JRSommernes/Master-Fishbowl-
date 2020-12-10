import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from numba import njit
import sys

#Slower if njit
def dyadic_green(sensors,dipole_pos,k_0):
    r_p = sensors-dipole_pos.reshape(3,1)

    R = np.sqrt(np.sum((r_p)**2,axis=0))
    R_hat = ((r_p)/R)

    RR_hat = np.einsum('ik,jk->ijk',R_hat,R_hat)

    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)
    expr_1 = (3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R
    expr_2 = (1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R

    I = np.identity(3)
    G = (expr_1*RR_hat + expr_2*I.reshape(3,3,1)).T

    return G

def plot_sensors(sensors,wl):
    coordinates = sensors/wl
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = coordinates[0],coordinates[1],coordinates[2]
    scat = ax.scatter(X, Y, Z)

    ax.set_box_aspect([X.max(),Y.max(),Z.max()])
    ax.set_xlabel('x-position [wavelengths]')
    ax.set_ylabel('y-position [wavelengths]')
    ax.set_zlabel('z-position [wavelengths]')
    plt.savefig('Detector_locations')

@njit(parallel=True)
def high_inner(A,B):
    a,b,c,d = A.shape
    C = np.zeros((a,b,c),dtype=np.complex64)
    for i in range(a):
        for j in range(b):
            for k in range(c):
                C[i,j,k] = A[i,j,k].dot(B[i,j,k])

    return C

def save_stack(I,dir,data):
    for i in range(I.shape[2]):
        im = Image.fromarray(np.abs(I[:,:,i]).astype(np.float64))
        im.save(dir+'/{}.tiff'.format(i))

def loadbar(counter,len):
    counter +=1
    done = (counter*100)//len
    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" % ('='*done, done))
    sys.stdout.flush()
    if counter == len:
        print('\n')
