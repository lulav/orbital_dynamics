import math
import numpy as np
from pyquaternion import Quaternion
from LULAV_math_functions import LulavMethFunctions
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kepler_functions import KeplerFunctions

lfm = LulavMethFunctions()
kf = KeplerFunctions()

class OrbitalManeuvers:
    def apse_line_rotation_from_new_apse_angle(self, kep_params_initial, kep_params_final, eta, mu):

        e_1 = kep_params_initial['e']
        a_1 = kep_params_initial['a']

        e_2 = kep_params_final['e']
        a_2 = kep_params_final['a']

        r_aps_1 = e_1*a_1 + a_1
        r_peri_1 = 2*a_1 - r_aps_1

        h_1 = np.sqrt(r_peri_1*mu*(1 + e_1))
    
        r_aps_2 = e_2*a_2 + a_2
        r_peri_2 = 2*a_2 - r_aps_2

        h_2 = np.sqrt(r_peri_2*mu*(1 + e_2))
    
        a = e_1*(h_2**2) - e_2*(h_1**2)*np.cos(eta)
        b = e_2*(h_1**2)*np.sin(eta)
        c = h_1**2 - h_2**2

        phi = np.atan2(b,a)
        theta_1 = phi - np.arcos((c/a) * np.cos(phi))

        r_rendez_vous = (h_1**2)/mu * (1/(1 + e_1*np.cos(theta_1))) 

        v_tangent_1 = h_1/r_rendez_vous
        v_radial_1 = (mu/h_1)*e_1*np.sin(theta_1)
        v_tot_1 = np.sqrt(v_tangent_1**2 + v_radial_1**2)
        gama_1 = np.atan(v_radial_1/v_tangent_1)

        v_tangent_2 = h_2/r_rendez_vous
        v_radial_2 = (mu/h_2)*e_2*np.sin(theta_1 - eta)
        v_tot_2 = np.sqrt(v_tangent_2**2 + v_radial_2**2)
        gama_2 = np.atan(v_radial_2/v_tangent_2)

        delta_v = np.sqrt(v_tot_1**2 + v_tot_2**2 - 2*np.cos(gama_2 - gama_1))
        PHI = np.atan2((v_radial_2 - v_radial_1)/(v_tangent_2 - v_tangent_1))    
        
        return(delta_v, PHI)

    ###############

    def apse_line_rotation_from_new_vel_pulse(self, kep_params_initial, delta_v, phi, mu):

        # phi is the angle of the added speed

        e_1 = kep_params_initial['e']
        a_1 = kep_params_initial['a']
        theta_1 = kep_params_initial['theta']

        r_aps_1 = e_1*a_1 + a_1
        r_peri_1 = 2*a_1 - r_aps_1

        h_1 = np.sqrt(r_peri_1*mu*(1 + e_1))
        delta_v_tangent = delta_v*np.cos(phi)
        delta_v_radial = delta_v*np.sin(phi)

        r_rendez_vous = (h_1**2)/mu * (1/(1 + e_1*np.cos(theta_1))) 
        h_2 = h_1 + r_peri_1*delta_v_tangent

        v_tangent_1 = h_1/r_rendez_vous
        v_radial_1 = (mu/h_1)*e_1*np.sin(theta_1)
        v_tangent_tot = delta_v_tangent + v_tangent_1
        v_radial_tot = delta_v_radial + v_radial_1
        
        num = v_tangent_tot*v_radial_tot*v_tangent_1**2
        a = (v_tangent_tot**2)*e_1*np.cos(theta_1)
        b = (2*v_tangent_1 + delta_v_tangent)*delta_v_tangent*(mu/r_peri_1)
        denum = a + b

        theta_2 = np.atan2(num, denum)
        eta = theta_1 - theta_2

        a = ((h_1 + r_peri_1*delta_v_tangent)**2)*e_1*np.cos(theta_1)
        b = (2*h_1 + r_peri_1*delta_v_tangent)*r_peri_1*delta_v_tangent
        num = a + b
        denum = h_1**2 *np.cos(theta_2)
        e_2 = num/denum


        r_peri_2 = (h_2**2 /mu)*1/(1 + e_2)
        r_apse_2 = (h_2**2 /mu)*1/(1 - e_2)
        a_2 = (r_peri_2 + r_apse_2)/2


        return(eta ,e_2, a_2)

    ###############

    def req_speed_change_from_inc_angle(self, radial_vel_1, tangent_vel_1, radial_vel_2, tangent_vel_2, inc):

        # the assumption is that this is happening very fast and therefore there is no change in the radial direction

        delta_vel = np.sqrt((radial_vel_1 - radial_vel_2)**2 + tangent_vel_1**2 + tangent_vel_2**2 - 2*tangent_vel_1*tangent_vel_2*np.cos(inc))
        return(delta_vel)

    ###############

    def req_inc_angle_from_speed_change(self, radial_vel_1, tangent_vel_1, radial_vel_2, tangent_vel_2):

        # the assumption is that this is happening very fast and therefore there is no change in the radial direction

        delta_inc = np.arccos((tangent_vel_1**2 + tangent_vel_2**2 + (radial_vel_1 - radial_vel_2)**2)/2*tangent_vel_1*tangent_vel_2)
        return(delta_inc)


    ###############

    def chase_manouver_in_same_orbit_from_req_time(self, eccent, semi_major_axis, target_true_anomaly, chaser_true_anomaly, req_time, mu):

        h = np.sqrt(semi_major_axis*mu*(1 - eccent**2))
        T = 2*np.pi*np.sqrt(semi_major_axis**3/mu)
        r_chaser = kf.celestial_orbit_position_in_Peri(eccent, semi_major_axis, chaser_true_anomaly, None)
        v_chaser = kf.celestial_velocity_in_peri(eccent, semi_major_axis, chaser_true_anomaly, None, mu)
    
        E_target = 2*np.arctan(np.sqrt((1 - eccent)/(1 + eccent))*np.tan(target_true_anomaly/2))
        time_of_target = kf.time_from_eccent_anomaly(eccent, semi_major_axis, E_target, mu)
        time_of_rendesvouz = req_time + time_of_target
        mean_anomaly_of_rendesvouz = 2*np.pi*(time_of_rendesvouz/T)

        E_0 = mean_anomaly_of_rendesvouz
        E_rendesvouz = kf.newton_raphson_for_eccentric_anomaly(E_0, eccent, mean_anomaly_of_rendesvouz, 10**-8, 1000)
        theta_rendesvouz = 2*np.arctan(np.sqrt((1 + eccent)/(1 - eccent))*np.tan(E_rendesvouz/2))
        if theta_rendesvouz < 0:
            theta_rendesvouz = 2*np.pi + theta_rendesvouz
        r_rendesvouz = kf.celestial_orbit_position_in_Peri(eccent, semi_major_axis, theta_rendesvouz, None)
        v_rendesvouz = kf.celestial_velocity_in_peri(eccent, semi_major_axis, theta_rendesvouz, None, mu)

        [v_chaser_initial, v_chaser_final] = kf.lamberts_problem_solution(r_chaser, r_rendesvouz, None, req_time, mu)
        delta_v_initial = np.array(v_chaser_initial) - v_chaser
        delta_v_final = v_rendesvouz - np.array(v_chaser_final)

        return(delta_v_initial,delta_v_final)

    ###############

    def chase_manouver_from_req_time(self, kep_target_params, kep_chaser_params, req_time, mu):

        a_target = kep_target_params['a']
        e_target = kep_target_params['e']
        target_true_anomaly = kep_target_params['theta']

        T = 2*np.pi*np.sqrt(a_target**3/mu)
        
        r_chaser = kf.celestial_orbit_position_in_ECI(kep_chaser_params)

        v_chaser = kf.celestial_velocity_in_ECI(kep_chaser_params, mu)

        E_target = 2*np.arctan(np.sqrt((1 - e_target)/(1 + e_target))*np.tan(target_true_anomaly/2))
        time_of_target = kf.time_from_eccent_anomaly(e_target, a_target, E_target, mu)
        time_of_rendesvouz = req_time + time_of_target
        mean_anomaly_of_rendesvouz = 2*np.pi*(time_of_rendesvouz/T)

        E_0 = mean_anomaly_of_rendesvouz
        E_rendesvouz = kf.newton_raphson_for_eccentric_anomaly(E_0, e_target, mean_anomaly_of_rendesvouz, 10**-8, 1000)
        theta_rendesvouz = 2*np.arctan(np.sqrt((1 + e_target)/(1 - e_target))*np.tan(E_rendesvouz/2))
        if theta_rendesvouz < 0:
            theta_rendesvouz = 2*np.pi + theta_rendesvouz
        
        kep_target_params['theta'] = theta_rendesvouz
        r_rendesvouz = kf.celestial_orbit_position_in_ECI(kep_target_params)
        v_rendesvouz = kf.celestial_velocity_in_ECI(kep_target_params, mu)
        [v_chaser_initial, v_chaser_final] = kf.lamberts_problem_solution(r_chaser, r_rendesvouz, None, req_time, mu)
        
        delta_v_initial = np.array(v_chaser_initial) - v_chaser
        delta_v_final = v_rendesvouz - np.array(v_chaser_final)

        return(delta_v_initial,delta_v_final)


    ###############

    def lamberts_problem_solution(self, r_1, r_2, delta_theta, delta_t, mu):

    # to see the solution for lamberts problem check "orbital mechanics for enginering students edition 6" page 202


        if delta_theta == None:
            if len(r_1) !=  len(r_2):
                return('Error!')
            
            elif len(r_1) != 1:
                r_1_norm = np.linalg.norm(r_1)
                r_2_norm = np.linalg.norm(r_2)

                if np.cross(r_1,r_2)[2] > 0:
                    delta_theta = np.arccos(np.dot(r_1,r_2)/(r_1_norm * r_2_norm))
                elif np.cross(r_1,r_2)[2] < 0:
                    delta_theta = 2*np.pi - np.arccos(np.dot(r_1,r_2)/(r_1_norm * r_2_norm))
                else:
                    return("r_1 and r_2 are the same vector!")
        else:
            r_1_norm = np.linalg.norm(r_1)
            r_2_norm = np.linalg.norm(r_2)

        A = np.sin(delta_theta)*np.sqrt((r_1_norm*r_2_norm)/(1 - np.cos(delta_theta)))
        
        z = -50
        F_z = self._F(z, delta_t, r_1_norm, r_2_norm, A, mu)

        while math.isnan(F_z) or (F_z < 0):
            z = z + 0.1
            F_z = self._F(z, delta_t, r_1_norm, r_2_norm, A, mu)

        tollerance = 10**-7
        n_max = 10000
    
        ratio = 1
        n = 0
        while (abs(ratio) > tollerance) and (n <= n_max):
            n = n + 1
            F_z = self._F(z, delta_t, r_1_norm, r_2_norm, A, mu)
            dF_dz = self._dFdz(z, A, r_1_norm, r_2_norm)
            ratio = F_z/dF_dz
            z = z - ratio

        # S_z = stumpff_function(z,3,50)
        # C_z = stumpff_function(z,2,50)

        S_z = self._stumpff_S_z(z)
        C_z = self._stumpff_C_z(z)
        y_z = self._y(z,r_1_norm,r_2_norm,S_z,C_z,A)

        f = 1 - y_z/r_1_norm
        g = A*np.sqrt(y_z/mu)
        g_dot = 1 - y_z/r_2_norm
        v_1 = 1/g*(r_2 - f*r_1)
        v_2 = 1/g*(g_dot*r_2 - r_1)
        return(v_1,v_2)

    ##################

    def stumpff_function(self, x,k,n):

        C = 0
        for i in range(n + 1):
            C = C + ((-x)**i)/(math.factorial(k + 2*i))

        return(C)

    ##################

    def _stumpff_S_z(self, z):

        if z > 0:
            C = (np.sqrt(z) - np.sin(np.sqrt(z)))/(np.sqrt(z)**3)
        elif z < 0:
            C = (np.sinh(np.sqrt(-z)) - np.sqrt(-z))/(np.sqrt(-z)**3)
        else:
            C = 1/6
            
        return(C)

    ##################

    def _stumpff_C_z(self, z):

        if z > 0:
            C = (1 - np.cos(np.sqrt(z)))/z
        elif z < 0:
            C = (np.cosh(np.sqrt(-z)) - 1)/(-z)
        else:
            C = 1/2
            
        return(C)

    ##################

    def _y(self, z,r_1_norm,r_2_norm,S_z,C_z,A):

        y_z = r_1_norm + r_2_norm + A*(z*S_z - 1)/np.sqrt(C_z)
        return(y_z)

    ##################

    def _F(self, z,t, r_1_norm, r_2_norm, A, mu):

        # S_z = stumpff_function(z,3,50)
        # C_z = stumpff_function(z,2,50)
        S_z = self._stumpff_S_z(z)
        C_z = self._stumpff_C_z(z)
    
        y_z = self._y(z,r_1_norm, r_2_norm,S_z,C_z,A)
        F_z = ((y_z/C_z)**1.5)*S_z + A*np.sqrt(y_z) - np.sqrt(mu)*t

        return(F_z)

    ##################

    def _dFdz(self, z,A,r_1_norm, r_2_norm):

        # S_z = stumpff_function(z,3,50)
        # C_z = stumpff_function(z,2,50)
        S_z = self._stumpff_S_z(z)
        C_z = self._stumpff_C_z(z)
        if z == 0:
            y_0 = self._y(0,r_1_norm,r_2_norm,S_z,C_z,A)
            F_tag = (np.sqrt(2)/40)*y_0**1.5 + A/8*(np.sqrt(y_0) + A*np.sqrt(1/(2*y_0)))
        else:
            y_z = self._y(z,r_1_norm, r_2_norm,S_z,C_z,A)
            F_tag = (y_z/C_z)**1.5*(1/(2*z)*(C_z - 3*S_z/(2*C_z)) + 3*S_z**2/(4*C_z)) + A/8*(3*(S_z/C_z)*np.sqrt(y_z) + A*np.sqrt(C_z/y_z))
        
        return(F_tag)

    #############

    def newton_raphson_for_eccentric_anomaly(self, E_0, e, M,tol, n_max):

        n = 0
        ratio = 1
        E = E_0
        while (n <= n_max) and (abs(ratio) > tol):
            n = n + 1
            f_E = E - e*np.sin(E) - M
            df_E = 1 - e*np.cos(E) 

            ratio = f_E/df_E
            E = E - ratio

        return(E)

    ##############

    def clohessy_wiltshire_matrices(self, Omega, delta_t):

        cs = np.cos(Omega*delta_t)
        sn = np.sin(Omega*delta_t)

        Phi_rr = np.array([[4 - 3*cs, 0, 0],
                        [6*(sn - Omega*delta_t),  1, 0],
                        [0,          0,         cs]]) # 

        Phi_rv =   np.array([[(1/Omega)*sn, (2/Omega)*(1 - cs), 0],
                            [(2/Omega)*(cs - 1),  (1/Omega)*(4*sn - 3*Omega*delta_t), 0],
                            [0,          0,         (1/Omega)*sn]]) #
        
        Phi_vr = np.array([[3*Omega*sn,    0, 0],
                        [6*Omega*(cs - 1),  0, 0],
                        [0,  0,  -Omega*sn]]) # 

        Phi_vv =   np.array([[cs, 2*sn, 0],
                            [-2*sn,  4*cs - 3, 0],
                            [0, 0,  cs]]) #
        

        return(Phi_rr, Phi_rv, Phi_vr, Phi_vv)

    ##############

    def two_impulse_rendezvous_maneuvers(self, kep_target_params, kep_chaser_params, req_time, mu):

        r_0_target = kf.celestial_orbit_position_in_ECI(kep_target_params)
        v_0_target = kf.celestial_velocity_in_ECI(kep_target_params, mu)

        r_0_chaser = kf.celestial_orbit_position_in_ECI(kep_chaser_params)
        v_0_chaser = kf.celestial_velocity_in_ECI(kep_chaser_params, mu)

        r_0_target_norm = np.linalg.norm(r_0_target)
        h_target = np.cross(r_0_target, v_0_target)
        v_tangent_target = np.cross(r_0_target, h_target)/r_0_target_norm
        v_tangent_target_norm = np.linalg.norm(v_tangent_target)

        x = r_0_target/r_0_target_norm
        y = -v_tangent_target/v_tangent_target_norm
        z = np.cross(x,y)

        target_axes_mat = np.transpose(np.column_stack((x, y, z)))

        T_target = 2*np.pi*np.sqrt((kep_target_params['a']**3)/mu)
        average_angular_velocity = 2*np.pi/T_target
        average_angular_velocity_vec = average_angular_velocity*z
    
        delta_r_0 = r_0_chaser - r_0_target
        delta_r_0_norm = np.linalg.norm(delta_r_0)
        if (delta_r_0_norm < r_0_target_norm/100):
            delta_v_0 = v_0_chaser - v_0_target - np.cross(average_angular_velocity_vec,delta_r_0)
            
            delta_r_0_target_frame = np.dot(target_axes_mat, delta_r_0)
            delta_v_0_target_frame = np.dot(target_axes_mat, delta_v_0)

            [Phi_rr, Phi_rv, Phi_vr, Phi_vv] = self.clohessy_wiltshire_matrices(average_angular_velocity, req_time)

            first_req_vel = np.dot(-np.linalg.inv(Phi_rv),np.dot(Phi_rr, delta_r_0_target_frame))
            final_req_vel = np.dot(Phi_vr,delta_r_0_target_frame) + np.dot(Phi_vv, first_req_vel)

            first_vel_pulse = first_req_vel - delta_v_0_target_frame
            final_vel_pulse = 0 - final_req_vel

            return(first_vel_pulse, final_vel_pulse)
        else:
            return("The CW method is linearized, therefore, doesn't suit big distances between target and chaser")
        
        
    ##############