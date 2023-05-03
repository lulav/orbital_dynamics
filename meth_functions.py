import math
import numpy as np
from pyquaternion import Quaternion

class MethFunctions:
    def dcm_from_euler_LULAV(self, phi,theta,psy,rot_config): # Boaz Gavriel

        rot_config = rot_config.lower()
        if rot_config == "zyx":
            
            R_roll = np.array([[1 , 0, 0],
                    [0, math.cos(psy), -math.sin(psy)],
                    [0, math.sin(psy),  math.cos(psy)]]) # around the X axis

            R_pitch = np.array([[math.cos(theta), 0 ,math.sin(theta)],
                        [0,  1,    0],
                        [-math.sin(theta),  0, math.cos(theta)]]) # around the Y axis   

            R_yaw = np.array([[math.cos(phi), -math.sin(phi),  0],
                        [math.sin(phi),  math.cos(phi),  0],
                        [0,   0,  1]]) # around the Z axis
        
            Rotation_Matrix = np.transpose(self, R_yaw*R_pitch*R_roll) #round((R_yaw*R_pitch*R_roll), 10); # how many numbers after the decimal dot
        elif rot_config == "xyz":
            R_roll = np.array([[1 , 0, 0],
                    [0, math.cos(phi), -math.sin(phi)],
                    [0, math.sin(phi),  math.cos(phi)]]) # around the X axis
            R_roll = np.transpose(R_roll)

            R_pitch = np.array([[math.cos(theta), 0 ,math.sin(theta)],
                        [0,  1,    0],
                        [-math.sin(theta),  0, math.cos(theta)]]) # around the Y axis   
            R_pitch = np.transpose(R_pitch)

            R_yaw =     np.array([[math.cos(psy), -math.sin(psy),  0],
                            [math.sin(psy),  math.cos(psy),  0],
                            [0,   0,  1]]) # around the Z axis 
            R_yaw = np.transpose(R_yaw)

            Rotation_Matrix = np.dot(R_yaw,np.dot(R_pitch,R_roll)) #round((R_yaw*R_pitch*R_roll), 10); # how many numbers after the decimal dot
        else:

            Rotation_Matrix = math.nan
            print('choose a correct rotation type!')
        return(Rotation_Matrix)


    def euler_angles_from_dcm_LULAV(self, M):
        pi = np.pi
        if (abs(M[2][0]) != 1): # in the "regular" case 
            theta1 = -math.arcsin(M[2][0]) # M(3,1) is the only free variable, therefore, we start with it
            theta2 = pi - theta1
            cs1 = 1/math.cos(theta1)
            cs2 = 1/math.cos(theta2) 
            phi1 = math.arctan2(M[2][1]*cs1,M[2][2]*cs1)
            phi2 = math.arctan2(M[2][1]*cs2,M[2][2]*cs2)
            psi1 = math.arctan2(M[1][0]*cs1,M[0][0]*cs1)
            psi2 = math.arctan2(M[1][0]*cs2,M[0][0]*cs2)
            Euler_angles1 = [phi1, theta1, psi1]
            Euler_angles2 = [phi2, theta2, psi2]

            return(Euler_angles1,Euler_angles2)
        else: # in case of so called gimbal lock
            phi1 = 0
            if ((M[2][0]) == -1):
                theta1 = pi/2
                psi1 = phi1 + math.arctan2(-M[0][1],M[0][2])
            else:
                theta1 = -pi/2
                psi1 = -phi1 + math.arctan2(-M[0][1],-M[0][2])
            Euler_angles1 = [phi1, theta1, psi1]
            Euler_angles2 = math.nan

            return(Euler_angles1,Euler_angles2)
        


    def lclf2lla_LULAV(self, LCLF,R_Moon): # Boaz Gavriel
    #this function recieves a vector in LCLF and transforms it to LLA
    #  if there's a problem - write it on a paper and wipe your ass with it

        LLA = [0 ,0 ,0]
        pi = np.pi
        r = np.norm(LCLF)

        if r > 0:

            LLA[0] = math.arcsin(LCLF[2]/r)

            LLA[1] = math.arctan2(LCLF[1],LCLF[0])
            if (LLA[1] > pi ):
                LLA[1] = LLA[1] - 2*pi 
            LLA[2] = r - R_Moon 
        return(LLA)



    def lla2lclf_LULAV(self, LLA, R_Moon): # Boaz Gavriel
    #this function recieves oordinates in LLA and transforms them to LCLF
    # it also recieves the radius of our beautiful Moon in [Km] 
    #  if there's a problem - write it on a paper and wipe your ass with it
        pi = np.pi
        LCLF = [0,0,0]
        if (R_Moon < 2000): # check for units in Km or m
            R_Moon = R_Moon*10^3
        if (len(LLA) != 3) or (LLA(1) > abs(2*pi)) or (LLA(2) > abs(pi/2)):
            print('not in LLA coordinates - asshole!')
        else:
            LCLF[0] = (R_Moon + LLA[2])*math.cos(LLA[1])*math.cos(LLA[0])
            LCLF[1] = (R_Moon + LLA[2])*math.cos(LLA[1])*math.sin(LLA[0])
            LCLF[2] = (R_Moon + LLA[2])*math.sin(LLA[1])
        return(LCLF)


    def lla2ned_dcm_LULAV(self, lat, long):
    # the function recieves the latitude and longtitude angles in LLA frame and
    # outputs the DCM from LCLF to NED

        cs_lat = math.cos(lat)
        sn_lat = math.sin(lat)
        cs_long = math.cos(long)
        sn_long = math.sin(long)

        ned_2_lclf_dcm = [[-sn_lat*cs_long , -sn_long , -cs_lat*cs_long]
                          [-sn_lat*sn_long,   cs_long,  -cs_lat*sn_long]
                          [cs_lat,          0 ,      -sn_lat]] 


        lclf_2_ned_dcm = np.transpose(ned_2_lclf_dcm)
        return(lclf_2_ned_dcm)


    def quat2dcm_LULAV(self, q): # Boaz Gavriel
    # this function recieves a rotation quaternion in the form of array and gives back
    # its proper Direction Cosine Matrix
    #   if there's a problem - write it on a paper and wipe your ass with it

        if (len(q) != 4): 
            print('not a quaternion - asshole!')
            return
        elif (np.linalg.norm(q) != 1):
            q = q/np.linalg.norm(q) # if it's not a unit quaternion then it is needed to be normelized
            DCM =  [[q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1]*q[2] - q[0]*q[3]),  2*(q[0]*q[2] + q[1]*q[3])],
                    [2*(q[1]*q[2] + q[0]*q[3]), q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2,  2*(q[2]*q[3] - q[0]*q[1])],
                    [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[0]*q[1] + q[2]*q[3]),  q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]]
        else:
            DCM =  [[q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1]*q[2] - q[0]*q[3]),  2*(q[0]*q[2] + q[1]*q[3])],
                    [2*(q[1]*q[2] + q[0]*q[3]), q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2,  2*(q[2]*q[3] - q[0]*q[1])],
                    [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[0]*q[1] + q[2]*q[3]),  q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]]
        
        return(DCM)


    def dcm2quat_LULAV(self, DCM):
        quat_array = []

        eta = 0.0025 # % eta is a threshhold parameter set to help with numerical accuracy,
                    #for more on the subject read: "Accurate Computation of Quaternions from Rotation Matrices" 
                    # https://upcommons.upc.edu/bitstream/handle/2117/124384/2068-Accurate-Computation-of-Quaternions-from-Rotation-Matrices.pdf?sequence=1
        
        q = [0,0,0,0]

        r11 = DCM[0][0]
        r12 = DCM[0][1]
        r13 = DCM[0][2]
        r21 = DCM[1][0]
        r22 = DCM[1][1]
        r23 = DCM[1][2]
        r31 = DCM[2][0]
        r32 = DCM[2][1]
        r33 = DCM[2][2]

        # q0
        if ((r11 + r22 + r33) > eta):
            q[0]= 0.5*math.sqrt(1 + r11 + r22 + r33)
        else:
            num0 = (r32 - r23)**2 + (r13 - r31)**2 + (r21 - r12)**2 
            denum0 = 3 - r11 - r22 - r33
            q[0] = 0.5*math.sqrt(num0 / denum0)
        
        # q1
        if ((r11 - r22 - r33) > eta):
            q[1]= 0.5*math.sqrt(1 + r11 - r22 - r33)
        else:
            num1 = (r32 - r23)**2 + (r12 + r21)**2 + (r31 + r13)**2
            denum1 = 3 - r11 + r22 + r33
            q[1] = 0.5*math.sqrt(num1 / denum1)
        q[1] = q[1]*self._quat_sign(r23-r32)
        
        # q2
        if ((- r11 + r22 - r33) > eta):
            q[2] = 0.5*math.sqrt(1 - r11 + r22 - r33)
        else:
            num2 = (r13 - r31)**2 + (r12 + r21)**2 + (r23 + r32)**2  
            denum2 = 3 + r11 - r22 + r33
            q[2] = 0.5*math.sqrt(num2 / denum2 )
        q[2] = q[2]*self._quat_sign(r31-r13)

        # q3
        if ((- r11 - r22 + r33) > eta):
            q[3]= 0.5*math.sqrt(1 - r11 - r22 + r33)
        else:
            num3 = (r12 - r21)**2 + (r31 + r13)**2 + (r32 + r23)**2 
            denum3 = 3 + r11 + r22 - r33
            q[3] = 0.5*math.sqrt(num3 / denum3)
        q[3] = q[3]*self._quat_sign(r12-r21)

        return q


    def _quat_sign(x):
        s = 1
        if x<-1e-9:
            s =-1
        return s 


    def quat_error(self, q,p): # Boaz Gavriel
    # this function recieves a rotation quaternion in the form of array and gives back
    # its proper Direction Cosine Matrix
    #   if there's a problem - write it on a paper and wipe your ass with it

        q = Quaternion(q)
        p = Quaternion(p)
        q = q.normalised # if it's not a unit quaternion then it is needed to be normelized
        p = p.normalised 

        p = p.conjugate
        q_difference = p*q
        quat_errors = q_difference.imaginary
        angular_error = 2*np.arccos((1 - (q_difference[0])))

        return(angular_error, quat_errors)

