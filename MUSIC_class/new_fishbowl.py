import numpy as np

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

class Fishbowl:
    def __init__(self,N_sensors,radius,wl,n,mur,epsr,k_0,dipoles,M_timepoints):
        self.N_sensors = N_sensors
        self.radius = radius
        self.wl = wl
        self.n_obj = n
        self.mur_obj = mur
        self.epsr_obj = epsr
        self.k_0 = k_0
        self.dipoles = dipoles
        self.M_timepoints = M_timepoints

        self.k_obj = self.k_0*self.n_obj

    def make_sensors(self):
        sensors = np.zeros((self.N_sensors,3))
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

        for i in range(self.N_sensors):
            y = (1 - (i / float(self.N_sensors - 1)) * 2)  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            x,y,z = x*self.radius, y*self.radius, z*self.radius

            sensors[i] = [x,y,z]
        self.sensors = np.ascontiguousarray(sensors.T)

    def limited_aperture_sensors(self,NA):
        self.NA = NA
        self.theta_max = np.arcsin(self.NA/self.n_obj)

        sphere_area = 4*np.pi*self.radius**2
        cap_area = 2*np.pi*self.radius**2*(1-np.cos(self.theta_max))
        mult = sphere_area/cap_area
        N_sensors = self.N_sensors
        self.N_sensors = int(self.N_sensors*mult)

        while 1:
            self.make_sensors()
            theta = np.arctan2(np.sqrt(self.sensors[0]**2+self.sensors[1]**2),self.sensors[2])
            if len(np.where(theta<=self.theta_max)[0]) > N_sensors:
                self.N_sensors -= 1
            elif len(np.where(theta<=self.theta_max)[0]) < N_sensors:
                self.N_sensors += 1
            else:
                break

        idx = np.where(theta<=self.theta_max)[0]
        self.sensors = self.sensors[:,idx]
        self.N_sensors = N_sensors

    #Slower if njit because of dyadic green dependecy
    def sensor_field(self,polarizations):
        E_tot = np.zeros((3*self.N_sensors,polarizations.shape[2]),dtype=np.complex128)

        for i in range(len(self.dipoles)):
            G = dyadic_green(self.sensors,self.dipoles[i],self.k_0).transpose(1,0,2)
            G = G.reshape(3*self.N_sensors,3)
            E_tot += G@polarizations[i]

        self.E_stack = E_tot

    #Slower if njit because dependencies
    def data_acquisition(self):
        N_dipoles = len(self.dipoles)

        phi = np.random.uniform(0,2*np.pi,(N_dipoles,self.M_timepoints))
        theta = np.random.uniform(-np.pi/2,np.pi/2,(N_dipoles,self.M_timepoints))

        polarizations = np.array([np.cos(phi)*np.sin(theta),
                                  np.sin(phi)*np.sin(theta),
                                  np.cos(theta)]).swapaxes(0,1)

        self.sensor_field(polarizations)

    def noise_space(self):
        self.E_N = []
        for i in range(3):
            E_tmp = self.E_stack[i*self.N_sensors:(i+1)*self.N_sensors]
            S = E_tmp@np.conjugate(E_tmp).T

            eigvals,eigvecs = np.linalg.eig(S)

            dist = np.sqrt(eigvals.real**2 + eigvals.imag**2)

            noice_idx = np.where(dist<1)[0]
            N = len(noice_idx)
            D = len(self.E_stack)-N

            self.E_N.append(eigvecs[:,noice_idx])

    def dyadic_green_2D(self,xx,yy,zz,grid_size):
        I = np.identity(3)
        shape_1 = np.append(self.N_sensors,np.ones(len(xx.shape),dtype=int))
        shape_2 = np.append(1,xx.shape)

        r_x = self.sensors[0].reshape(shape_1)-xx.reshape(shape_2)
        r_y = self.sensors[1].reshape(shape_1)-yy.reshape(shape_2)
        r_z = self.sensors[2].reshape(shape_1)-zz*np.ones(shape_2)
        r_p = np.array((r_x,r_y,r_z))

        R = np.sqrt(np.sum((r_p)**2,axis=0))
        R_hat = ((r_p)/R)
        RR_hat = np.einsum('iklm,jklm->ijklm',R_hat,R_hat)

        g_R = np.exp(1j*self.k_0*R)/(4*np.pi*R)
        expr_1 = (3/(self.k_0**2*R**2)-3j/(self.k_0*R)-1)*g_R
        expr_1 = np.broadcast_to(expr_1,RR_hat.shape)

        expr_2 = (1+1j/(self.k_0*R)-1/(self.k_0**2*R**2))*g_R
        expr_2 = np.broadcast_to(expr_2,RR_hat.shape)

        I = np.broadcast_to(I,(self.N_sensors,grid_size,grid_size,3,3))
        I = I.transpose(3,4,0,1,2)

        G = (expr_1*RR_hat + expr_2*I).transpose((2,0,1,3,4))

        return G

    def P_calc_2D(self,A_fov,N_recon):
        self.P = np.zeros((N_recon,N_recon),dtype = np.complex128)
        for i in range(3):
            a,b,c,d = A_fov[i].shape

            A = A_fov[i].reshape(-1, A_fov.shape[-1])
            B =  self.E_N[i].reshape(-1, self.E_N[i].shape[-1])

            P_fov_1 = np.conjugate(A)@B
            P_fov_2 = A@np.conjugate(B)

            P_fov_1 = P_fov_1.reshape(a,b,c,P_fov_1.shape[-1])
            P_fov_2 = P_fov_2.reshape(a,b,c,P_fov_2.shape[-1])

            P_fov_plane = (1/np.einsum('ijkl,ijkl->ijk',P_fov_1,P_fov_2)).T
            self.P += np.sum(P_fov_plane,axis=0)

    def P_estimation(self,N_recon,FoV):
        x = y = np.linspace(-FoV/2,FoV/2,N_recon)*self.wl
        xx,yy = np.meshgrid(x,y)
        z = [0]

        A_fov_plane = self.dyadic_green_2D(xx,yy,z,N_recon).transpose(0,2,3,4,1)
        # A_fov_plane = np.ascontiguousarray((A_fov_plane.reshape(3,self.N_sensors,3,N_recon,N_recon)).T)
        A_fov_plane = np.ascontiguousarray(A_fov_plane.T)

        self.P_calc_2D(A_fov_plane,N_recon)
