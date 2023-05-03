import math
import numpy as np
from pyquaternion import Quaternion
from meth_functions import MethFunctions 
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


lmf = MethFunctions()

class KeplerFunctions:

    def __init__(self):
        
        self.Sun = {}
        self.Mercury = {}
        self.Venus = {}
        self.Earth = {}
        self.Moon = {}
        self.Mars = {}
        self.Jupiter = {}
        self.Saturn = {}
        self.Uranus = {}
        self.Neptune = {}
        self.Pluto = {}
        self.Eris = {}

        self.Sun['mu'] = 1.32712440018*(10**11)
        self.Mercury['mu'] = 2.2032*(10**4)
        self.Venus['mu'] = 3.24859*(10**5)
        self.Earth['mu'] = 3.986004418*(10**5) 
        self.Moon['mu'] = 4.9048695*(10**1)
        self.Mars['mu'] = 4.282837*(10**4)
        self.Jupiter['mu'] = 1.26686534*(10**8)
        self.Saturn['mu'] = 3.7931187*(10**7)
        self.Uranus['mu'] = 5.793939*(10**6)
        self.Neptune['mu'] = 6.836529*(10**6)
        self.Pluto['mu'] = 8.71*(10**0)
        self.Eris['mu'] = 1.108*(10**1)

        self.Sun['r'] = 696340
        self.Mercury['r'] = 2440
        self.Venus['r'] = 6052
        self.Earth['r'] = 6378
        self.Moon['r'] = 1734
        self.Mars['r'] = 3390
        self.Jupiter['r'] = 69911
        self.Saturn['r'] = 58232
        self.Uranus['r'] = 25362
        self.Neptune['r'] = 24622
        self.Pluto['r'] = 1188.3
        self.Eris['r'] = 1163

    def kepler_params_init(self, eccent, semi_major_axis, eccent_anomaly, true_anomaly, OMEGA, inclination, omega):

        kep = {}
        kep['e'] = eccent
        kep['a'] = semi_major_axis
        kep['E'] = eccent_anomaly
        kep['theta'] = true_anomaly
        kep['OM'] = OMEGA
        kep['i'] = inclination
        kep['om'] = omega

        if (kep['E'] != None) and (kep['theta'] != None):
            r_norm_from_E = semi_major_axis*(1 - np.cos(eccent_anomaly)*eccent)
            r_norm_from_theta = semi_major_axis*(1 - eccent**2)/(1 + eccent*np.cos(true_anomaly))

            if abs(r_norm_from_E - r_norm_from_theta) > 10**-6:
                print('true anomaly and eccentric anomaly do not match, fixing eccentric anomaly according to true anomaly')
                if kep['theta'] < np.pi:
                    kep['E'] = np.arccos(eccent + r_norm_from_theta*np.cos(kep['theta'])/semi_major_axis)
                else:
                    kep['E'] = 2*np.pi - np.arccos(eccent + r_norm_from_theta*np.cos(kep['theta'])/semi_major_axis)


        return(kep)

    #################

    def get_orbit_in_peri(self, kep_params):

        e = kep_params['e']
        a = kep_params['a']
        theta_0 = kep_params['theta']
        E_0 = kep_params['E']

        theta_division = 1000
        theta = np.zeros(theta_division)
        dtheta = np.pi/theta_division

        if (theta_0 == None) and (E_0 == None): 
            theta = np.arange(0, 2*np.pi, dtheta)
        elif (theta_0 == None) and (E_0 != None): 

            r_norm = a*(1 - np.cos(E_0)*e)
            if E_0 < np.pi:
                theta_0 = np.arccos(a*(np.cos(E_0) - e)/r_norm)
            else:
                theta_0 = 2*np.pi - np.arccos(a*(np.cos(E_0) - e)/r_norm)
                    
            theta = np.arange(theta_0, 2*np.pi + theta_0, dtheta)
        else:

            theta = np.arange(theta_0, 2*np.pi + theta_0, dtheta)

        r = np.zeros(len(theta))
        E = np.zeros(len(theta))
        elipsis_rad = np.zeros([len(theta), 2])
        for dtheta,i in zip(theta, range(len(theta))):
            r[i] = (a*(1 - e**2)/(1 + np.cos(dtheta)*e))
            elipsis_radX = (r[i]*np.cos(dtheta))
            elipsis_radY = (r[i]*np.sin(dtheta))
            if dtheta < np.pi:
                E[i] = np.arccos(e + r[i]*np.cos(dtheta)/a)
            else:
                E[i] = 2*np.pi - np.arccos(e + r[i]*np.cos(dtheta)/a)

            elipsis_rad[i,:] = [elipsis_radX, elipsis_radY]
        
        return(elipsis_rad, E)

    #################

    def get_orbit_in_ECI(self, kep_params):

        OM = kep_params['OM']
        i = kep_params['i']
        om = kep_params['om']
    
        Peri_2_ECI = self.Peri_2_ECI_matrix(OM,i,om)

        [r,E] = self.get_orbit_in_peri(kep_params)
        r_z = np.zeros((len(r[:,0])))
        r_peri_3d = np.insert(r, r.shape[1], r_z, axis=1)
        orbit_in_ECI = np.zeros((len(r[:,0]),3))

        for i in range(len(r[:,0])):

            r_peri = r_peri_3d[i,:]   
            r_ECI = np.dot(Peri_2_ECI,r_peri)
            orbit_in_ECI[i,:] = r_ECI

        return(np.array(orbit_in_ECI),E)

    ###############

    def Peri_2_ECI_matrix(self, OMEGA, inclanation, argument_of_perigee):

        R_inclanation = np.array([[1 , 0, 0],
                                [0, math.cos(inclanation), math.sin(inclanation)],
                                [0, -math.sin(inclanation),  math.cos(inclanation)]]) # around the X axis

        R_Omega = np.array([[math.cos(OMEGA), math.sin(OMEGA),  0],
                            [-math.sin(OMEGA),  math.cos(OMEGA),  0],
                            [0,   0,  1]]) # around the Z axis

        R_argument_of_perigee = np.array([[math.cos(argument_of_perigee), math.sin(argument_of_perigee),  0],
                                        [-math.sin(argument_of_perigee),  math.cos(argument_of_perigee),  0],
                                        [0,   0,  1]]) # around the Z axis 

        Peri_2_ECI = np.transpose(np.dot(R_argument_of_perigee,(np.dot(R_inclanation,R_Omega)))) #
        return(Peri_2_ECI)

    ###############

    def ECI_2_Peri_matrix(self, OMEGA, inclnation, argument_of_perigee):

        Peri_2_ECI = self.Peri_2_ECI_matrix(OMEGA, inclnation, argument_of_perigee)
        ECI_2_Peri = np.transpose(Peri_2_ECI)
        return(ECI_2_Peri)

    ###############

    def celestial_orbit_position_in_ECI(self, kep_params):

        e = kep_params['e']
        a = kep_params['a']
        E = kep_params['E']
        theta = kep_params['theta']
        OM = kep_params['OM']
        i = kep_params['i']
        om = kep_params['om']

        if theta == None:
            r_norm = a*(1 - np.cos(E)*e)
            if E < np.pi:
                theta = np.arccos(a*(np.cos(E) - e)/r_norm)
            else:
                theta = 2*np.pi - np.arccos(a*(np.cos(E) - e)/r_norm)
        else:
            r_norm = a*(1 - e**2)/(1 + e*np.cos(theta))

        
        r_x = r_norm*np.cos(theta)
        r_y = r_norm*np.sin(theta)
        r_z = 0

        r = np.transpose(np.array([r_x, r_y, r_z]))
        Peri_2_ECI = self.Peri_2_ECI_matrix(OM,i,om)
        r_ECI = np.dot(Peri_2_ECI,r)
        return(r_ECI)

    ###############

    def celestial_velocity_in_ECI(self, kep_params, mu):

        e = kep_params['e']
        a = kep_params['a']
        E = kep_params['E']
        theta = kep_params['theta']
        OM = kep_params['OM']
        i = kep_params['i']
        om = kep_params['om']
        
        h = np.sqrt(a*mu*(1 - e**2))

        if theta == None:

            r_norm = a*(1 - np.cos(E)*e)
            if E < np.pi:
                theta = np.arccos(a*(np.cos(E) - e)/r_norm)
            else:
                theta = 2*np.pi - np.arccos(a*(np.cos(E) - e)/r_norm)
        
        v_x = mu/h*(-np.sin(theta))
        v_y = mu/h*(e + np.cos(theta))
        v_z = 0

        v = np.transpose(np.array([v_x, v_y, v_z]))
        Peri_2_ECI = self.Peri_2_ECI_matrix(OM,i,om)
        v_ECI = np.dot(Peri_2_ECI,v)

        return(v_ECI)

    #################

    def celestial_orbit_position_in_Peri(self, kep_params):

        e = kep_params['e']
        a = kep_params['a']
        E = kep_params['E']
        theta = kep_params['theta']

        if theta == None:
            r_norm = a*(1 - np.cos(E)*e)
            if E < np.pi:
                theta = np.arccos(a*(np.cos(E) - e)/r_norm)
            else:
                theta = 2*np.pi - np.arccos(a*(np.cos(E) - e)/r_norm)
        else:
            r_norm = a*(1 - e**2)/(1 + e*np.cos(theta))

        r_X = r_norm*np.cos(theta)
        r_Y = r_norm*np.sin(theta)

        r = np.array([r_X, r_Y])
        return(r)

    ###############

    def celestial_velocity_in_peri(self, kep_params, mu):
        
        e = kep_params['e']
        a = kep_params['a']
        E = kep_params['E']
        theta = kep_params['theta']

        h = np.sqrt(a*mu*(1 - e**2))

        if theta == None:

            r_norm = a*(1 - np.cos(E)*e)
            if E < np.pi:
                theta = np.arccos(a*(np.cos(E) - e)/r_norm)
            else:
                theta = 2*np.pi - np.arccos(a*(np.cos(E) - e)/r_norm)
        
        # tangent_vel = h/r_norm
        # radial_vel = (mu/h)*eccent*np.sin(theta)

        velocity_in_peri = mu/h*np.array([-np.sin(theta), e + np.cos(theta)])
        return(velocity_in_peri)

    ###############
        
    def time_from_eccent_anomaly(self, kep_params, mu):

        e = kep_params['e']
        a = kep_params['a']
        E = kep_params['E']
        time_period = 2*np.pi*np.sqrt(a**3/mu)
        M_e = (E - e*np.sin(E))
        time = time_period*M_e/(2*np.pi)

        return(time)

    ###############

    def kep_params_from_pos_and_vel(self, r, v, mu):

        v_norm = np.linalg.norm(v)
        r_norm = np.linalg.norm(r)
        
        h = np.cross(r,v) 
        h_norm = np.linalg.norm(h)  
        i = np.arccos(h[2]/h_norm)

        k = [0,0,1]
        N = np.cross(k,h)
        N_norm = np.linalg.norm(N)
        if N[1] >= 0:
            Omega = np.arccos(N[0]/N_norm)
        else:
            Omega = 2*np.pi - np.arccos(N[0]/N_norm)

        specific_tot_energy = v_norm**2/2 - mu/r_norm
        
        e_vec = np.cross(v,h)/mu - r/r_norm
        e_norm = np.linalg.norm(e_vec)
        p = h_norm**2/mu
        semi_major_axis = p/(1 - e_norm**2)

        if e_vec[2] >= 0:
            omega = np.arccos(np.dot(N,e_vec)/(N_norm*e_norm))
        else:
            omega = 2*np.pi - np.arccos(np.dot(N,e_vec)/(N_norm*e_norm))

        v_r = np.dot(r,v)/r_norm
        if v_r >= 0:
            theta = np.arccos(np.dot(e_vec, r)/(e_norm*r_norm))
        else:
            theta = 2*np.pi - np.arccos(np.dot(e_vec, r)/(e_norm*r_norm))

        E = np.arccos(r_norm*np.cos(theta)/semi_major_axis + e_norm)

        kep_params = self.kepler_params_init(e_norm, semi_major_axis, E, theta, Omega, i, omega)

        return(kep_params)

    ##############

