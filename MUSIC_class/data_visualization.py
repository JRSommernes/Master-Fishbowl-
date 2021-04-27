import numpy as np
import matplotlib.pyplot as plt
import os, json

def plot_N_sensors_fishbowl():
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

def plot_N_sensors_micro():
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

def plot_f_cam_fishbowl():
    dire = ['single_variable/fishbowl/f_cam','single_variable/fishbowl/f_cam_full_aperture']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for dir in dire:
        lst = os.listdir(dir)
        res_lim = np.zeros((len(lst),2),dtype=np.float64)
        for i, el in enumerate(lst):
            with open(dir+'/'+el) as f:
                data = json.load(f)

            rad = float(data['Sensor radius'])
            res = float(data['Resolution limit [wl]'])
            res_lim[i] = (rad,res)

        sort = res_lim[res_lim[:,0].argsort()]
        if dir == 'single_variable/fishbowl/f_cam_full_aperture':
            ax.plot(sort[:,0],sort[:,1],c='orange')
            plt.text(0.1,0.7,'Full Sphere Aperture',transform=ax.transAxes)
            plt.plot(0.07,0.712,'o',c='orange',transform=ax.transAxes)
        else:
            ax.plot(sort[:,0],sort[:,1],c='b')
            plt.text(0.1,0.8,'Semi Aperture, NA = {}'.format(data['NA']),transform=ax.transAxes)
            plt.plot(0.07,0.812,'o',c='b',transform=ax.transAxes)
        # plt.plot(sort[:,0],sort[:,1])
    plt.ylabel('Resolution limit [wl]')
    plt.xlabel('Detector radius [m]')
    plt.show()

def plot_offset_fishbowl():
    dire = ['single_variable/fishbowl/test_off_x']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for dir in dire:
        lst = os.listdir(dir)
        res_lim = np.zeros((len(lst),2),dtype=np.float64)
        for i, el in enumerate(lst):
            with open(dir+'/'+el) as f:
                data = json.load(f)

            off = float(el.split('_')[0])
            res = float(data['Resolution limit [wl]'])
            res_lim[i] = (off,res)

        sort = res_lim[res_lim[:,0].argsort()]
        if dir == 'single_variable/fishbowl/x_offset_full_aperture':
            ax.plot(sort[:,0],sort[:,1],c='orange')
            # ax.plot(sort[:,0],sort[:,1],c='orange')
            plt.text(0.6,0.7,'Full Sphere Aperture',transform=ax.transAxes)
            plt.plot(0.59,0.712,'o',c='orange',transform=ax.transAxes)
        elif dir == 'single_variable/fishbowl/test_off_x':
            ax.plot(sort[:,0],sort[:,1],c='b')
            # ax.plot(sort[:,0],sort[:,1],c='b')
            plt.text(0.6,0.8,'Semi Aperture, NA = {}'.format(data['NA']),transform=ax.transAxes)
            plt.plot(0.59,0.812,'o',c='b',transform=ax.transAxes)
        elif dir == 'single_variable/fishbowl/y_offset_full_aperture':
            ax.plot(sort[:,0],sort[:,1],'-.',c='orange')
            # ax.plot(sort[:,0],sort[:,1],c='orange')
            plt.text(0.6,0.7,'Full Sphere Aperture',transform=ax.transAxes)
            plt.plot(0.59,0.712,'o',c='orange',transform=ax.transAxes)
        elif dir == 'single_variable/fishbowl/y_offset':
            ax.plot(sort[:,0],sort[:,1],'-.',c='b')
            # ax.plot(sort[:,0],sort[:,1],c='b')
            plt.text(0.6,0.8,'Semi Aperture, NA = {}'.format(data['NA']),transform=ax.transAxes)
            plt.plot(0.59,0.812,'o',c='b',transform=ax.transAxes)
        # plt.plot(sort[:,0],sort[:,1])
    plt.ylabel('Resolution limit [wl]')
    plt.xlabel('Dipole offset [wl]')
    plt.show()

def plot_directs(dir,var,lab):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    for j,d in enumerate(dir):
        lst = os.listdir(d)
        res_lim = np.zeros((len(lst),2),dtype=np.float64)
        for i, el in enumerate(lst):
            with open(d+'/'+el) as f:
                data = json.load(f)

            x = float(data[var[j]])
            res = float(data['Resolution limit [wl]'])
            res_lim[i] = (x,res)

        sort = res_lim[res_lim[:,0].argsort()]

        ax.plot(sort[:,0],sort[:,1],label=lab[j])

    # plt.legend(loc='best')
    plt.ylabel('Resolution limit [wl]')
    plt.xlabel('Working distance')
    plt.savefig('images/plots/{}.png'.format(var[0]),dpi=300,format='png')
    plt.show()



if __name__ == '__main__':
    # plot_N_sensors_fishbowl()
    # plot_N_sensors_micro()
    # plot_f_cam_fishbowl()
    # plot_offset_fishbowl()

    # dir = ['single_variable/fishbowl/N_sensors','single_variable/fishbowl/N_sensors_full_aperture']
    # var = ['N_sensors','N_sensors']
    # lab = ['Semi Aperture, NA = 1.2', 'Full Sphere Aperture']
    #
    dir = ['single_variable/microscope/f_obj']
    var = ['f_obj']
    lab = ['Semi Aperture, NA = 1.2']
    plot_directs(dir,var,lab)
