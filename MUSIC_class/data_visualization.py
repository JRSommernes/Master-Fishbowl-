import numpy as np
import matplotlib.pyplot as plt
import os, json

if __name__ == '__main__':
    dire = ['single_variable/fishbowl/N_sensors','single_variable/fishbowl/N_sensors_full_aperture']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for dir in dire:
        lst = os.listdir(dir)
        res_lim = np.zeros((len(lst),2),dtype=np.float64)

        for i, el in enumerate(lst):
            with open(dir+'/'+el) as f:
                data = json.load(f)

            sen = int(data['N_sensors'])
            res = float(data['Resolution limit [wl]'])
            res_lim[i] = (sen,res)

        sort = res_lim[res_lim[:,0].argsort()]

        if dir == 'single_variable/fishbowl/N_sensors_full_aperture':
            ax.plot(sort[:,0],sort[:,1],c='orange')
            plt.text(0.5,0.7,'Full Sphere Aperture',transform=ax.transAxes)
            plt.plot(0.47,0.712,'o',c='orange',transform=ax.transAxes)
        else:
            ax.plot(sort[:,0],sort[:,1],c='b')
            plt.text(0.5,0.8,'Semi Aperture, NA = {}'.format(data['NA']),transform=ax.transAxes)
            plt.plot(0.47,0.812,'o',c='b',transform=ax.transAxes)


    plt.ylabel('Resolution Limit [wl]')
    plt.xlabel('Number of sensors')
    plt.show()





    dir = 'single_variable/fishbowl/f_cam'
    lst = os.listdir(dir)
    res_lim = np.zeros((len(lst),2),dtype=np.float64)
    for i, el in enumerate(lst):
        with open(dir+'/'+el) as f:
            data = json.load(f)

        rad = float(data['Sensor radius'])
        res = float(data['Resolution limit [wl]'])
        res_lim[i] = (rad,res)

    sort = res_lim[res_lim[:,0].argsort()]
    plt.plot(sort[:,0],sort[:,1])
    plt.ylabel('Resolution limit [wl]')
    plt.xlabel('Detector radius [m]')
    plt.show()




    dir = 'single_variable/microscope/N_sensors'
    lst = os.listdir(dir)
    res_lim = np.zeros((len(lst),2),dtype=np.float64)
    for i, el in enumerate(lst):
        with open(dir+'/'+el) as f:
            data = json.load(f)

        rad = float(data['N_sensors'])
        res = float(data['Resolution limit [wl]'])
        res_lim[i] = (rad,res)

    sort = res_lim[res_lim[:,0].argsort()]
    plt.plot(sort[1:,0],sort[1:,1])
    plt.ylabel('Resolution limit [wl]')
    plt.xlabel('Number of sensors')
    plt.show()
