import numpy as np
from misc_functions import *

def dyadic_green_FoV_2D(sensors,xx,yy,zz,N_sensors,grid_size,k_0):
    I = np.identity(3)

    shape_1 = np.append(N_sensors,np.ones(len(xx.shape),dtype=int))
    shape_2 = np.append(1,xx.shape)

    r_x = sensors[0].reshape(shape_1)-xx.reshape(shape_2)
    r_y = sensors[1].reshape(shape_1)-yy.reshape(shape_2)
    r_z = sensors[2].reshape(shape_1)-zz*np.ones(shape_2)
    r_p = np.array((r_x,r_y,r_z))

    R = np.sqrt(np.sum((r_p)**2,axis=0))
    R_hat = ((r_p)/R)
    RR_hat = np.einsum('iklm,jklm->ijklm',R_hat,R_hat)

    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)
    expr_1 = (3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R
    expr_1 = np.broadcast_to(expr_1,RR_hat.shape)

    expr_2 = (1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R
    expr_2 = np.broadcast_to(expr_2,RR_hat.shape)

    I = np.broadcast_to(I,(N_sensors,grid_size,grid_size,3,3))
    I = I.transpose(3,4,0,1,2)

    G = (expr_1*RR_hat + expr_2*I).transpose((2,0,1,3,4))

    return G

def correlation_space(E_field):
    S = E_field@np.conjugate(E_field).T

    eigvals,eigvecs = np.linalg.eig(S)

    dist = np.sqrt(eigvals.real**2 + eigvals.imag**2)

    noice_idx = np.where(dist<1)[0]
    N = len(noice_idx)
    D = len(E_field)-N
    # print(D//3)
    #
    # E_N = eigvecs[:,noice_idx]

    return np.ascontiguousarray(eigvecs)

def fishbowl(E_sensors,sensors,wl,k_0,N_recon,FoV,dipoles):
    wl = 690e-9
    N_sensors = sensors.shape[1]

    E_x = E_sensors[0::3]
    E_y = E_sensors[1::3]
    E_z = E_sensors[2::3]

    E_sensors = np.append(E_x,np.append(E_y,E_z,axis=0),axis=0)

    theta = np.arctan2(np.sqrt(sensors[0]**2+sensors[1]**2),sensors[2])
    phi = np.arctan2(sensors[1],sensors[0])

    I_x = np.abs(E_x)**2
    I_y = np.abs(E_y)**2
    I_z = np.abs(E_z)**2

    theta = np.pi/2-theta

    I_theta = I_z
    I_phi = I_x+I_y

    N_theta = correlation_space(I_theta)
    N_phi = correlation_space(I_phi)


    x = np.linspace(FoV[0,0],FoV[0,1],N_recon)
    y = np.linspace(FoV[1,0],FoV[1,1],N_recon)
    z = np.linspace(FoV[2,0],FoV[2,1],N_recon)

    labels = np.zeros((N_recon,N_recon,N_recon),dtype=np.uint8)

    for k,tmp in enumerate(dipoles):
        xyz = np.array([x,y,z])
        dist = np.abs(xyz-tmp.reshape(3,1))
        x_dist = dist[0]
        y_dist = dist[1]
        z_dist = dist[2]

        x_pos = np.where(x_dist==np.amin(x_dist))[0][0]
        y_pos = np.where(y_dist==np.amin(y_dist))[0][0]
        z_pos = np.where(z_dist==np.amin(z_dist))[0][0]

        labels[x_pos,y_pos,z_pos] = 1

    xx,yy = np.meshgrid(x,y)
    for i,zz in enumerate(z):
        labels_plane = labels[:,:,i]
        labels_plane = labels_plane.reshape(1,N_recon**2)

        N_samples = np.random.randint(0,N_recon**2//100)
        sample_points = np.random.randint(0,N_recon**2,N_samples)

        if len(np.unique(labels_plane)) == 1:
            continue

        else:
            dipole_pos = np.nonzero(labels_plane)[1]
            sample_points = np.sort(np.append(sample_points,dipole_pos))



        #
        #     G = dyadic_green_FoV_2D(sensors,xx,yy,zz,N_sensors,N_recon,k_0)
        #     G = G.transpose((3,4,1,2,0))
        #
        #     T_1 = (G@N_theta).reshape(N_recon**2,3**2*N_sensors)
        #     T_2 = (G@N_phi).reshape(N_recon**2,3**2*N_sensors)
        #     T = np.append(T_1,T_2,axis=1)
        #
        #     batch_size = 256
        #     N_batch = int(np.ceil(T.shape[0]/batch_size))
        #
        #     for j in range(N_batch):
        #         batch = T[j*batch_size:(j+1)*batch_size]


    exit()



    # print(T.shape)
    #
    # exit()

    # print(A.shape,B.shape)
    # print(T_1.shape)
    # exit()

    return T_1,T_2

    # return T


    # Simple P calulation
    # pos = np.array([[0,0,0]])
    # G = dyadic_green(sensors,pos,k_0)
    #
    # A = G.reshape(-1, G.shape[-1])[0::3]
    # B =  N_theta.reshape(-1, N_theta.shape[-1])
    #
    # P_1 = np.conjugate(A.T)@B
    # P_2 = A.T@np.conjugate(B)
    #
    # P = (1/np.einsum('ij,ij->i',P_1,P_2)).T
    # P = np.sum(P,axis=0)
    #
    # print(P)
    # exit()
