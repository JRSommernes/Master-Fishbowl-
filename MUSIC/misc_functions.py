import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
