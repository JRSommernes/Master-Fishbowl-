import numpy as np

#Implement one scattering microscope and one fluorecent microscope


class Microscope:
    def __init__(self,N_sensors,microscope_radius,n_0):
        self.N_sensors = N_sensors
        self.n_0 = n_0
        self.radius = microscope_radius

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

class Scattering_Microscope(Microscope):
    def __init__(self,N_sensors,N_emitters,microscope_radius,n_0):
        Microscope.__init__(self,N_sensors,microscope_radius,n_0)
        self.N_emitters = N_emitters

    def make_emitters(self):
        emitters = np.zeros((self.N_emitters,3))
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

        for i in range(self.N_emitters):
            y = (1 - (i / float(self.N_emitters - 1)) * 2)  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            x,y,z = x*self.radius, y*self.radius, z*self.radius

            emitters[i] = [x,y,z]

        x = emitters[:,0]
        y = emitters[:,1]
        z = emitters[:,2]

        theta = np.arctan2(np.sqrt(x**2+y**2),z)
        phi = np.arctan2(y,x)+0.5

        x = self.radius*np.sin(theta)*np.cos(phi)
        y = self.radius*np.sin(theta)*np.sin(phi)
        z = self.radius*np.cos(theta)

        emitters = np.array((x,y,z))
        self.emitters = np.ascontiguousarray(emitters)
