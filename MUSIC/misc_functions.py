import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from numba import njit

def plot_sensors(sensors,wl):
    coordinates = sensors/wl
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = coordinates[0],coordinates[1],coordinates[2]
    scat = ax.scatter(X, Y, Z)
    # ax.plot([-15, 15], [0,0],zs=[0,0])
    # ax.plot([0,0], [-15, 15],zs=[0,0])
    # ax.plot([0,0], [0,0],zs=[-15, 15])

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

@njit(parallel=True)
def high_outer(A,B):
    a,b,c,d,e = A.shape
    C = np.zeros((a,b,c,d,e,e),dtype=np.complex64)
    for i in range(a):
        for j in range(b):
            for k in range(c):
                for l in range(d):
                    C[i,j,k,l] = np.outer(A[i,j,k,l],B[i,j,k,l])

    return C

def save_stack(I,dir):
    for i in range(I.shape[2]):
        im = Image.fromarray(np.abs(I[:,:,i]).astype(np.float64))
        im.save(dir+'/{}.tiff'.format(i))
