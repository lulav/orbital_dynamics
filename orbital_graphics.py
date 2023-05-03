import math
import numpy as np
from pyquaternion import Quaternion
from kepler_functions import KeplerFunctions  
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

kf = KeplerFunctions()

class OrbitalGraphics:
    def draw_orbit(self, kep_array, planet_radius):

        orbital_array = []
        for kep in kep_array:
            [orbit,E] = kf.get_orbit_in_ECI(kep)
            orbital_array.append(orbit)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        if (planet_radius != None) or (planet_radius != 0):

            # Create data for the sphere
            u, v = np.mgrid[0:2*np.pi:80j, 0:np.pi:40j]
            x = planet_radius*np.cos(u)*np.sin(v)
            y = planet_radius*np.sin(u)*np.sin(v)
            z = planet_radius*np.cos(v)

            # Plot the sphere
            ax.plot_surface(x, y, z, cmap='viridis' ,zorder=0)

        for orbit in orbital_array:
            line, = plt.plot(orbit[:,0],orbit[:,1],orbit[:,2], zorder=1)
            z_order = 2 if orbit[0,2] > 0 else 0
            line.set_zorder(z_order)
            ax.collections[0].set_zorder(z_order - 1)

        # Set the x, y, and z limits

        # ax.set_xlim(min(x), max(x))
        # ax.set_ylim(min(y), max(y))
        # ax.set_zlim(min(z), max(z))

        # ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])
        # ax.axis('auto')

        plt.show()

    ##############
