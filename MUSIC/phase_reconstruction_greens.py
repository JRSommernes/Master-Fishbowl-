import numpy as np
import matplotlib.pyplot as plt
from misc_functions import dyadic_green
from quadraticInv import *

def plot_sensor_field(sensors,E):
    x = sensors[0]
    y = sensors[1]

    plt.scatter(x,y,c=np.abs(E))
    plt.colorbar()
    plt.show()

def make_2d_sensor_grid(M0,N,wl):
    phi = np.linspace(0,2*np.pi,M0)
    rho = np.arange(5,5+2*N,2)*wl

    phi_mat, rho_mat = np.meshgrid(phi,rho)

    xx = rho_mat*np.cos(phi_mat)
    x = xx.flatten()
    yy = rho_mat*np.sin(phi_mat)
    y = yy.flatten()
    z = np.zeros_like(y)

    return np.array([x,y,z])

def make_dipoles():
    dipole_pos = np.array([[-0.8*wl,0,0],[0.8*wl,0,0]])
    # dipole_pos = np.array([[0,0,0]])


    phi = np.random.uniform(0,2*np.pi,(len(dipole_pos)))
    theta = np.random.uniform(-np.pi/2,np.pi/2,(len(dipole_pos)))

    # phi = np.ones_like(phi)*0
    # theta = np.ones_like(phi)*np.pi


    polarizations = np.array([np.cos(phi)*np.sin(theta),
                              np.sin(phi)*np.sin(theta),
                              np.cos(theta)]).swapaxes(0,1)

    return dipole_pos, polarizations

def find_xTrue(FoV,M,pos,pol):
    xTrue = np.zeros((M,M,3))

    x = np.linspace(-FoV/2,FoV/2,M)
    y = np.linspace(-FoV/2,FoV/2,M)
    for i,p in enumerate(pos):
        x_idx = np.argmin(np.abs(x-p[0]))
        y_idx = np.argmin(np.abs(y-p[1]))
        xTrue[x_idx,y_idx] = pol[i]

    return xTrue

def make_Am(sensors,wl,M,FoV):
    N = sensors.shape[1]

    x = np.linspace(-FoV/2,FoV/2,M)
    y = np.linspace(-FoV/2,FoV/2,M)

    G = np.zeros((M,M,N,3,3),dtype=np.complex128)
    for i,xx in enumerate(x):
        for j,yy in enumerate(y):
            pos = np.array([xx,yy,0])
            G[i,j] += dyadic_green(sensors,pos,k_0)

    G_th = G[:,:,:,2].transpose(3,0,1,2)

    phi = np.angle(sensors[0]+sensors[1]*1j)
    phi = phi*(phi>0)+(2*np.pi+phi)*(phi<=0)
    phi = (phi*180/np.pi).astype(int)
    phi = phi%90
    alpha = (90-phi)*np.pi/180
    G_phi_1 = G[:,:,:,0].transpose(3,0,1,2)*np.cos(alpha)
    G_phi_2 = G[:,:,:,1].transpose(3,0,1,2)*np.cos(phi)
    G_ph = G_phi_1 + G_phi_2

    return G_th,G_ph

def make_E_field(sensors,wl,k_0):
    N = sensors.shape[1]
    pos, pol = make_dipoles()
    FoV = 4*wl
    NN = 50

    G = np.zeros((N,3),dtype=np.complex128)
    for i in range(len(pos)):
        G += dyadic_green(sensors,pos[i],k_0)@pol[i]

    E_theta = G[:,2]
    # E_phi = G[:,0]+G[:,1]

    phi = np.angle(sensors[0]+sensors[1]*1j)
    phi = phi*(phi>0)+(2*np.pi+phi)*(phi<=0)
    phi = (phi*180/np.pi).astype(int)
    phi = phi%90
    alpha = (90-phi)*np.pi/180
    E_phi_1 = G[:,0]*np.cos(alpha)
    E_phi_2 = G[:,1]*np.cos(phi)
    E_phi = E_phi_1 + E_phi_2

    I_th = np.abs(E_theta)**2
    I_ph = np.abs(E_phi)**2


    Am_th, Am_ph = make_Am(sensors,wl,NN,FoV)

    Am = np.append(Am_th, Am_ph, axis=-1)
    b = np.append(I_th, I_ph)
    E_true = np.append(E_theta, E_phi)

    xTrue = find_xTrue(FoV,NN,pos,pol).transpose(2,0,1)


    # test1 = Am.reshape(-1, Am.shape[-1])
    # test2 = xTrue.flatten().reshape(-1,1)
    #
    # test3 = test1.transpose()@test2
    # test4 = np.abs(test3)**2
    #
    # print(test1.shape,test2.shape)
    # print(np.unique(np.equal(test3,b)))
    #
    # plt.plot(test3.real,c='b')
    # plt.plot(E_true.real,c='r')
    # plt.show()
    # plt.plot(test3.imag,c='b')
    # plt.plot(E_true.imag,c='r')
    # plt.show()
    # exit()

    Am = Am.reshape(-1, Am.shape[-1]).T
    b = b.reshape(-1, 1)

    return Am, b, E_true, xTrue


if __name__ == '__main__':
    eps_0 = 8.8541878128e-12
    mu_0 = 4*np.pi*10**-7
    c_0 = 1/np.sqrt(eps_0*mu_0)

    wl = 690e-9
    freq = c_0/wl
    k_0 = 2*np.pi*freq*np.sqrt(eps_0*mu_0)

    M0 = 49
    N = 4

    sensors = make_2d_sensor_grid(M0,N,wl)

    Am, b, E_true, xTrue = make_E_field(sensors,wl,k_0)

    xEst = algQuadraticInv(Am,b)

    normalized_xTrue = xTrue*np.conj(xTrue[0])/np.abs(np.conj(xTrue[0]))
    normalized_xEst = xEst*np.conj(xEst[0])/np.abs(np.conj(xEst[0]))
    N0 = (len(xEst)-1)/2
    vecn = np.arange(-N0,N0+1)

    plt.plot(vecn,np.angle(normalized_xTrue))
    plt.plot(vecn,np.angle(normalized_xEst),'*')
    plt.xlim([-12, 12])
    plt.show()

    plt.plot(vecn,np.abs(normalized_xTrue))
    plt.plot(vecn,np.abs(normalized_xEst),'*')
    plt.xlim([-12, 12])
    plt.show()
