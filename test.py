import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test1():
    NA = 1.2
    n = 1.33
    N = 50

    theta_0 = np.arcsin(NA/n)
    phi = np.linspace(0,2*np.pi,N)
    theta = np.linspace(-theta_0,theta_0,2*N)

    phi, theta = np.meshgrid(phi,theta)

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)

    p = ax.scatter(x,y,z)
    ax.set_box_aspect((5,5,2))
    plt.show()

def test2():
    NA = 1.2
    n = 1.33
    N = 50

    theta_0 = np.arcsin(NA/n)

    x = np.linspace(-1,1,N)
    y = np.linspace(-1,1,N)
    z = np.linspace(-1,1,N)

    xx,yy,zz = np.meshgrid(x,y,z)
    theta = np.arctan2(np.sqrt(xx**2+yy**2),zz)
    r = np.sqrt(xx**2+yy**2+zz**2)

    pupil = 1*(r<1)*(theta<theta_0)*(z>np.cos(theta_0))
    pupil_T = np.flip(pupil,axis=2)

    pupil_fft = np.fft.fftn(pupil)
    pupil_T_fft = np.fft.fftn(pupil_T)
    PSF = pupil_fft*pupil_fft

    OTF = np.fft.fftshift(np.fft.fftn(PSF))

    min = np.amin(OTF.real)
    max = np.amax(OTF.real)

    for i in range(N):
        plt.imshow(OTF[:,:,i].real,vmin=min,vmax=max)
        plt.show()

    # pupil_func = np.array([np.where(pupil==1)[0],np.where(pupil==1)[1],np.where(pupil==1)[2]])
    #
    # x_p = xx[pupil_func[0],pupil_func[1],pupil_func[2]]
    # y_p = yy[pupil_func[0],pupil_func[1],pupil_func[2]]
    # z_p = zz[pupil_func[0],pupil_func[1],pupil_func[2]]
    #
    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    #
    # p = ax.scatter(x_p,y_p,z_p)
    # ax.set_box_aspect((1,1,1))
    # fig.colorbar(p)
    # plt.show()


if __name__ == '__main__':
    test2()
