import numpy as np
import matplotlib.pyplot as plt
import os, json

if __name__ == '__main__':
    dir = 'C:/Users/Jon-Richard Sommerne/github/Master-Fishbowl-/MUSIC_class/resolutions'

    microscope_data = np.zeros((3,4,3,3,4))
    """
    shape = (3,4,3,3,4)
    1: N_sensors
    2: M_timepoints
    3: Radius
    4: NA
    5: Camera FoV
    """

    wl = 690e-9
    N_sensors = np.array([25,100,225])
    M_timepoints = np.array([10,40,70,100])
    f_cam = np.array([5e-2,1e-1,2e-1])
    NA = np.array([0.8,1,1.2])
    voxel_size = np.array([60,80,100,120])*1e-9

    for i in range(432):
        with open(dir+'/{}_data_microscope.json'.format(i)) as f:
            data = json.load(f)

        sen = int(data['N_sensors'])
        tim = int(data['N_timepoints'])
        rad = float(data['f_cam'])
        na = float(data['NA'])

        fov = float(data['Camera FoV [wl]'])
        vox = fov/np.sqrt(sen)

        a = np.where(N_sensors==sen)[0][0]
        b = np.where(M_timepoints==tim)[0][0]
        c = np.where(f_cam==rad)[0][0]
        d = np.where(NA==na)[0][0]
        e = np.argmin(np.abs(voxel_size-vox*wl))

        res = float(data['Resolution limit [wl]'])

        microscope_data[a,b,c,d,e] = res

    dir = 'C:/Users/Jon-Richard Sommerne/github/Master-Fishbowl-/MUSIC_class/resolutions'
    fishbowl_data = np.zeros((8,4,3,3))

    # N_sensors = np.array([300,400,500,600,700,800,900,1000])
    for i in range(428):
        if i%4==0:
            with open(dir+'/{}_data_fishbowl.json'.format(i)) as f:
                data = json.load(f)

            sen = int(data['N_sensors'])
            tim = int(data['N_timepoints'])
            rad = float(data['Sensor radius'])
            na = float(data['NA'])

            # a = np.where(N_sensors==sen)[0][0]
            a = np.argmin(np.abs(N_sensors-sen))
            b = np.where(M_timepoints==tim)[0][0]
            c = np.where(f_cam==rad)[0][0]
            d = np.where(NA==na)[0][0]

            res = float(data['Resolution limit [wl]'])

            fishbowl_data[a,b,c,d] = res
            print(data['N_sensors'])


    # print(microscope_data[:,2,0,2,0])
    print(fishbowl_data[2,2,:,2])
