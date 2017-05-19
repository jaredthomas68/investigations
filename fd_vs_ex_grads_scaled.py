from openmdao.api import pyOptSparseDriver, Problem, profile

from time import time
import matplotlib.pyplot as plt
import numpy as np

from wakeexchange.OptimizationGroups import OptAEP
from wakeexchange.GeneralWindFarmComponents import calculate_boundary
from wakeexchange.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from wakeexchange.floris import floris_wrapper, add_floris_params_IndepVarComps
from wakeexchange.larsen import larsen_wrapper, add_larsen_params_IndepVarComps
# from wakeexchange.jensen import jensen_wrapper, add_jensen_params_IndepVarComps


def plot_results(data, save=False, nTurbines=None, maxDirections=None):

    time_ex_f = data[0, :]
    time_ex_g = data[1, :]
    time_fd_f = data[2, :]
    time_fd_g = data[3, :]
    AEPf = data[4, :]
    AEPg = data[5, :]
    directions = data[6, :]


    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle('%i Turbines' % nTurbines)
    ax1.plot(directions, time_ex_f, linestyle='-', color='b', marker='o', markersize=4, label='ex. f')
    ax1.plot(directions, time_ex_g, linestyle='-', color='r', marker='o', markersize=4, label='ex. g')
    ax1.plot(directions, time_fd_f, linestyle=':', color='b', marker='o', markersize=4, label='fd. f')
    ax1.plot(directions, time_fd_g, linestyle=':', color='r', marker='o', markersize=4, label='fd. g')
    ax1.set_xlabel('N Directions')
    ax1.set_ylabel('Time to Calculate Gradients of X and Y (s)')
    ax1.legend(loc=2)
    ax2.plot(directions, AEPf, linestyle='-', color='b', marker='o', markersize=4, label='FLORIS')
    ax2.plot(directions, AEPg, linestyle='-', color='r', marker='o', markersize=4, label='Gauss')
    ax2.set_xlabel('N Directions')
    ax2.set_ylabel('AEP (kWh)')
    ax2.legend(loc=4)
    plt.tight_layout()
    if save:
        plt.savefig('scaling_by_directions_combined_%i_turbines_%i_directions.pdf' % (nTurbines, maxDirections))
    plt.show()


def calc_data(nRows, maxDirections, savetxt=False):

    # Scaling grid case
    rotor_diameter = 126.4
    spacing = 4  # turbine grid spacing in diameters
    nTurbines = nRows ** 2

    # Set up position arrays
    points = np.linspace(start=spacing * rotor_diameter, stop=nRows * spacing * rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)

    print turbineX, turbineY

    # initialize input variable arrays
    rotorDiameter = np.ones(nTurbines) * np.random.random() * rotor_diameter
    axialInduction = np.ones(nTurbines) * np.random.random() * (1. / 3.)
    Ct = np.ones(nTurbines) * np.random.random()
    Cp = np.ones(nTurbines) * np.random.random()
    generatorEfficiency = np.ones(nTurbines) * np.random.random()
    yaw = np.zeros(nTurbines)

    # Define flow properties
    wind_speed = 8.0  # m/s
    air_density = 1.1716  # kg/m^3

    spacing = 5  # turbine grid spacing in diameters

    directions = np.arange(1, maxDirections + 1, 20)
    time_ex_f = np.zeros_like(directions) * 0.0
    time_ex_g = np.zeros_like(directions) * 0.0
    time_fd_f = np.zeros_like(directions) * 0.0
    time_fd_g = np.zeros_like(directions) * 0.0
    AEPf = np.zeros_like(directions)
    AEPg = np.zeros_like(directions)
    index = 0

    for d in directions:
        # set up problem

        probg = Problem(root=OptAEP(nTurbines, nDirections=d, use_rotor_components=False,
                                    wake_model=gauss_wrapper,
                                    params_IdepVar_func=add_gauss_params_IndepVarComps,
                                    wake_model_options={'nSamples': 0},
                                    params_IndepVar_args=None))

        probf = Problem(root=OptAEP(nTurbines, nDirections=d, use_rotor_components=False,
                                    wake_model=floris_wrapper, differentiable=True,
                                    params_IdepVar_func=add_floris_params_IndepVarComps,
                                    wake_model_options=None,
                                    params_IndepVar_args=None))

        # initialize problem
        # profile.setup(probf)
        probf.setup()
        probg.setup()

        # assign values to constant inputs (not design variables)
        probf['turbineX'] = probg['turbineX'] = turbineX
        probf['turbineY'] = probg['turbineY'] = turbineY
        probf['hubHeight'] = probg['hubHeight'] = np.zeros_like(turbineX) + 90.
        probf['yaw0'] = probg['yaw0'] = yaw
        probf['rotorDiameter'] = probg['rotorDiameter'] = rotorDiameter
        probf['axialInduction'] = probg['axialInduction'] = axialInduction
        probf['Ct_in'] = probg['Ct_in'] = Ct
        probf['Cp_in'] = probg['Cp_in'] = Cp
        probf['generatorEfficiency'] = probg['generatorEfficiency'] = generatorEfficiency
        probf['windSpeeds'] = probg['windSpeeds'] = np.ones(d) * wind_speed
        probf['air_density'] = probg['air_density'] = air_density
        probf['windDirections'] = probg['windDirections'] = np.linspace(0, 360, d)
        probf['windFrequencies'] = probg['windFrequencies'] = np.ones(d) / d

        # run problem
        probf.run()
        probg.run()
        AEPf[index] = probf['AEP']
        AEPg[index] = probg['AEP']

        # pass results to self for use with unit test
        tic = time()
        # profile.start()
        J = probf.calc_gradient(['turbineX', 'turbineY'], ['AEP'], return_format='dict', mode='auto')
        # profile.stop()
        toc = time()
        time_ex_f[index] = toc - tic

        tic = time()
        # profile.start()
        J = probf.calc_gradient(['turbineX', 'turbineY'], ['AEP'], return_format='dict', mode='fd')
        # profile.stop()
        toc = time()
        time_fd_f[index] = toc - tic

        tic = time()
        # profile.start()
        J = probg.calc_gradient(['turbineX', 'turbineY'], ['AEP'], return_format='dict', mode='auto')
        # profile.stop()
        toc = time()
        time_ex_g[index] = toc - tic

        tic = time()
        # profile.start()
        J = probg.calc_gradient(['turbineX', 'turbineY'], ['AEP'], return_format='dict', mode='fd')
        # profile.stop()
        toc = time()
        time_fd_g[index] = toc - tic

        index += 1

    if savetxt:
        np.savetxt('scaling_by_directions_combined_%i_turbines_%i_directions.txt' % (nTurbines, maxDirections),
                   np.array([time_ex_f, time_ex_g, time_fd_f, time_fd_g, AEPf, AEPg, directions]))

    return np.array([time_ex_f, time_ex_g, time_fd_f, time_fd_g, AEPf, AEPg, directions])


def load_data(nRows, maxDirections):

    data = np.loadtxt('scaling_by_directions_%i_turbines_%i_directions.txt' %(nRows**2, maxDirections))

    return data


def N2_diagram(nRows, maxDirections):

    from openmdao.api import view_tree

    # Scaling grid case
    rotor_diameter = 126.4
    spacing = 4  # turbine grid spacing in diameters
    nTurbines = nRows ** 2

    # Set up position arrays
    points = np.linspace(start=spacing * rotor_diameter, stop=nRows * spacing * rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)

    # initialize input variable arrays
    rotorDiameter = np.ones(nTurbines) * np.random.random() * rotor_diameter
    axialInduction = np.ones(nTurbines) * np.random.random() * (1. / 3.)
    Ct = np.ones(nTurbines) * np.random.random()
    Cp = np.ones(nTurbines) * np.random.random()
    generatorEfficiency = np.ones(nTurbines) * np.random.random()
    yaw = np.zeros(nTurbines)

    # Define flow properties
    wind_speed = 8.0  # m/s
    air_density = 1.1716  # kg/m^3

    spacing = 5  # turbine grid spacing in diameters

    directions = np.arange(5, maxDirections + 1, 5)
    time_ex_f = np.zeros_like(directions)
    time_ex_g = np.zeros_like(directions)
    time_fd_f = np.zeros_like(directions)
    time_fd_g = np.zeros_like(directions)
    AEPf = np.zeros_like(directions)
    AEPg = np.zeros_like(directions)
    index = 0

    probg = Problem(root=OptAEP(nTurbines, nDirections=maxDirections, use_rotor_components=False,
                                wake_model=gauss_wrapper,
                                params_IdepVar_func=add_gauss_params_IndepVarComps,
                                wake_model_options={'nSamples': 0},
                                params_IndepVar_args=None))

    probf = Problem(root=OptAEP(nTurbines, nDirections=maxDirections, use_rotor_components=False,
                                wake_model=floris_wrapper, differentiable=True,
                                params_IdepVar_func=add_floris_params_IndepVarComps,
                                wake_model_options=None,
                                params_IndepVar_args=None))

    # initialize problem
    probf.setup()
    probg.setup()

    view_tree(probf, show_browser=True, outfile='tree_floris.html')
    view_tree(probg, show_browser=True, outfile='tree_gauss.html')


if __name__ == '__main__':

    # select how large of a wind farm and how many directions to run
    nRows = 2 # wind farm will be of size nTurbines=nRows**2
    maxDirections = 100 # will start with 1 wind direction and run up to this number

    data = calc_data(nRows, maxDirections, savetxt=False)

    # data = load_data(nRows, maxDirections)

    plot_results(data, save=False, nTurbines=nRows**2, maxDirections=maxDirections)

    # N2_diagram(nRows, maxDirections)