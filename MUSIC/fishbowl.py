import numpy as np
from misc_functions import *
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self,in_size):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_size, 512),
            nn.Linear(512, 128),
            nn.Linear(128,16),
            nn.Linear(16,2)
        )
    def forward(self, x):
        out = self.mlp(x)
        return out

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
    # print(dist)
    #
    # E_N = eigvecs[:,noice_idx]

    return np.ascontiguousarray(eigvecs)

def fishbowl(E_sensors,sensors,wl,k_0,N_recon,FoV,dipoles):
    wl = 690e-9
    N_sensors = sensors.shape[1]

    model = MLP(N_sensors*3*3*2*2).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    E_x = E_sensors[0::3]
    E_y = E_sensors[1::3]
    E_z = E_sensors[2::3]

    E_sensors = np.append(E_x,np.append(E_y,E_z,axis=0),axis=0)

    theta = np.arctan2(np.sqrt(sensors[0]**2+sensors[1]**2),sensors[2])
    phi = np.arctan2(sensors[1],sensors[0])

    I_x = np.abs(E_x)**2
    I_y = np.abs(E_y)**2
    I_z = np.abs(E_z)**2
    I = I_x+I_y+I_z

    theta = np.pi/2-theta

    I_theta = I_z
    I_phi = I_x+I_y

    # plot_sensor_field(sensors,I_x[:,0])
    # plot_sensor_field(sensors,I_y[:,0])
    # plot_sensor_field(sensors,I_z[:,0])
    # plot_sensor_field(sensors,I_x[:,0]+I_y[:,0]+I_z[:,0])

    pos = np.array([[0,0,0]])
    A = dyadic_green(sensors,pos,k_0)

    T = (A.T@I).reshape(3*3,1000)
    print(np.linalg.matrix_rank(T.T@T))

    N_theta = correlation_space(I_theta)
    N_phi = correlation_space(I_phi)
    exit()


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

    model.train()
    losses = []
    xx,yy = np.meshgrid(x,y)
    for i,zz in enumerate(z):
        loadbar(i,len(z))
        labels_plane = labels[:,:,i]
        labels_plane = labels_plane.reshape(1,N_recon**2)

        N_samples = np.random.randint(1,N_recon**2//100)
        sample_points = np.random.randint(0,N_recon**2,N_samples)

        if len(np.unique(labels_plane)) == 1:
            sample_points = np.sort(sample_points)

        else:
            dipole_pos = np.nonzero(labels_plane)[1]
            sample_points = np.unique(np.append(sample_points,dipole_pos))

        sample_labels = labels_plane[:,sample_points]
        sample_size = len(sample_points)

        idx = np.where(sample_labels==1)[1]

        tmp_1 = np.ones(sample_size)
        tmp_2 = np.zeros(sample_size)
        tmp_1[idx] = 0
        tmp_2[idx] = 1
        tmp = np.array([tmp_1,tmp_2]).T

        G = dyadic_green_FoV_2D(sensors,xx,yy,zz,N_sensors,N_recon,k_0)
        G = G.transpose((3,4,1,2,0)).reshape(N_recon**2,3,3,N_sensors)
        G = G[sample_points]

        T_1 = (G@N_theta).reshape(sample_size,3**2*N_sensors)
        T_2 = (G@N_phi).reshape(sample_size,3**2*N_sensors)
        T = np.append(T_1,T_2,axis=1)

        T_r = T.real
        T_i = T.imag
        T = np.append(T_r,T_i,axis=1).astype(np.float32)

        output = model(torch.from_numpy(T))
        y = torch.from_numpy(sample_labels.flatten()).long()



        loss = criterion(output, y)
        loss.backward()
        losses.append(loss.item())

        optimizer.step()


    tmp = [1000, 3080, 4000, 5120, 7120, 10000]
    G = dyadic_green_FoV_2D(sensors,xx,yy,30,N_sensors,N_recon,k_0)
    G = G.transpose((3,4,1,2,0)).reshape(N_recon**2,3,3,N_sensors)
    G = G[tmp]

    T_1 = (G@N_theta).reshape(len(tmp),3**2*N_sensors)
    T_2 = (G@N_phi).reshape(len(tmp),3**2*N_sensors)
    T = np.append(T_1,T_2,axis=1)

    T_r = T.real
    T_i = T.imag
    T = np.append(T_r,T_i,axis=1).astype(np.float32)
    test = model(torch.from_numpy(T))

    print(test)

    exit()

        # if len(np.unique(labels_plane)) == 1:
        #     if np.random.randint(0,10) < 7:
        #         continue
        #
        # G = dyadic_green_FoV_2D(sensors,xx,yy,zz,N_sensors,N_recon,k_0)
        # G = G.transpose((3,4,1,2,0))
        #
        # T_1 = (G@N_theta).reshape(N_recon**2,3**2*N_sensors)
        # T_2 = (G@N_phi).reshape(N_recon**2,3**2*N_sensors)
        # T = np.append(T_1,T_2,axis=1)
        #
        # batch_size = 256
        # N_batch = int(np.ceil(T.shape[0]/batch_size))
        #
        # print(T.shape,labels_plane.shape)
        # exit()
        #
        # for j in range(N_batch):
        #     batch = T[j*batch_size:(j+1)*batch_size]


    exit()



    # print(T.shape)
    #
    # exit()

    # print(A.shape,B.shape)
    # print(T_1.shape)
    # exit()

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
