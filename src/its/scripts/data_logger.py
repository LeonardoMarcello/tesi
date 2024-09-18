#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import WrenchStamped, PoseStamped, TransformStamped
from std_msgs.msg import Bool
import tf

from its_msgs.msg import SoftContactSensingProblemSolution, TacTipDensity
from std_srvs.srv import Empty, EmptyResponse
import csv
import os
import time
import numpy as np
import sympy as sp
import threading
import argparse


import moveit_commander

# -- Setup TacTip ----------------------------------------------------- 
# > 0deg
#   TABLE_HEIGHT : 0.105
#   SKIN_HEIGHT : 0.020
#   INDENTATION : 0.00x
# > 15deg
#   TABLE_HEIGHT : 0.137
#   SKIN_HEIGHT : 0.020
#   INDENTATION : 0.00x
# > 30deg
#   TABLE_HEIGHT : 0.134
#   SKIN_HEIGHT : 0.020
#   INDENTATION : 0.00x
# ----------------------------------------------------------------------

# -- Setup DigiTac ----------------------------------------------------- 
# > 0deg
#   TABLE_HEIGHT : 0.0355
#   SKIN_HEIGHT : 0.0055  
#   INDENTATION : 0.00x

# > 15deg
#   TABLE_HEIGHT : 0.0445
#   SKIN_HEIGHT : 0.0055  
#   INDENTATION : 0.00x

# > 30deg
#   TABLE_HEIGHT : 0.0535
#   SKIN_HEIGHT : 0.0055  
#   INDENTATION : 0.00x
# ----------------------------------------------------------------------

# -- CMD GROUND TRUTH TABLE ------------------------------------------------
#
#           |    0 degrees   |    15 degrees     |    30 degrees
#   0 mm    |   (0,0,4.74)   |  (0,-1.27,4.73)   |  (0,-2.71,4.69)
#   2 mm    |   (0,0,2.74)   |  (0,-0.73,2.73)   |  (0,-1.57,2.72)
#   3 mm    |   (0,0,1.74)   |  (0,-0.47,1.74)   |  (0,-1,1.73)
#   4 mm    |   (0,0,0.74)   |  (0,-0.2,0.74)    |  (0,-0.43,0.74)
# ----------------------------------------------------------------------


# ======================================
#   Class for GroundTruh retrieval
# ======================================
class FrankaPosition:
    def __init__(self):
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.group_name = "panda_arm"  
        self.reference_frame = "world"  # Replace with your desired reference frame
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.move_group.set_end_effector_link("brush")
        self.end_effector_link = self.move_group.get_end_effector_link()

        self.tf_listener = tf.TransformListener()

        self.x0 = np.array([.0, .0, .0445])       # centro B in table frame, {T} [m]
                                                  # Digit 0deg - z=0.0346; 15deg - z=0.0446; 30deg - z=0.0526;
                                                  # TacTip 0deg - z=0.105; 15deg - z=0.145; 30deg - z=0.0526;
        self.c0 = np.array([.0, .0, .0])          # contact point in Table frame, {T} [mm]

        self.x = np.array([.0, .0, .0])           # Current Position in Fingertip frame, {B} [m]
        self.x_start = np.array([.0, .0, .0])     # Posizione pre-grasp in Fingertip frame, {B} [mm]

        self.contact = False                      # Condizione di contatto

    def get_end_effector_position(self):
        # get end effector position in both table frame {T} and fingertip frame {B}
        end_effector_pose = PoseStamped()
        end_effector_pose.pose = self.move_group.get_current_pose(self.end_effector_link).pose        
        end_effector_pose.header.frame_id = self.reference_frame

        ee_in_table = self.tf_listener.transformPose("/table_link", end_effector_pose)
        fingertip_id = rospy.get_param('fingertip/id')
        ee_in_fingertip = self.tf_listener.transformPose(fingertip_id, end_effector_pose)

        self.x = np.array([ee_in_fingertip.pose.position.x, ee_in_fingertip.pose.position.y, ee_in_fingertip.pose.position.z])

        return ee_in_table, ee_in_fingertip
    
    def set_center(self):
        # Set fingertip center in table frame {T}
        ee,_ = self.get_end_effector_position()
        x = ee.pose.position.x
        y = ee.pose.position.y
        z = ee.pose.position.z

        self.x0 = np.array([x,y,z])

    def set_contact_point(self):
        # Set contact point in table frame {T}
        ee,_ = self.get_end_effector_position()
        x = ee.pose.position.x
        y = ee.pose.position.y
        z = ee.pose.position.z

        self.c0 = np.array([x,y,z])
        print(f"contact detected at {x, y, z}")

    def get_indentation(self):
        # get indentation depth along z axis of table frame {T}
        ee,_ = self.get_end_effector_position()
        x = ee.pose.position.x
        y = ee.pose.position.y
        z = ee.pose.position.z

        c = np.array([x,y,z])
        delta = c - self.c0

        return delta[2]*1000.0
    

# ======================================
#   Class for data logging
# ======================================
class DataLogger:
    def __init__(self):
        
        rospy.init_node('its_data_logger_node', anonymous=True)

        self.real_cc = np.array([0,0,0.8],dtype=float)                                      # <---- Real Contact Centroid in {B} [mm]
        self.real_theta = rospy.get_param("fingertip/orientation/roll",0.0)/np.pi*180       # <---- Real Theta [deg]
        
        # Parameters (Ellipsoid size for real cc estimation)
        a = rospy.get_param("fingertip/principalSemiAxis/a", 20)
        b = rospy.get_param("fingertip/principalSemiAxis/b", 20)
        c = rospy.get_param("fingertip/principalSemiAxis/c", 20)
        self.ell_params = [a, b, c] 

        # Object for retrieval of GroundTruth
        try:
            self.franka = FrankaPosition()
            self.franka.theta = self.real_theta
        except Exception as e:
            print(e)

        self.tfBroad = tf.TransformBroadcaster()
        self.setFrame()

        # Initialize CSV file     
        DATE = time.strftime('%d_%m_%Y')
        dir = os.path.join(os.path.join("data",DATE))    
        if not os.path.exists(dir):
            os.mkdir(dir)
        # extended results of ITS solver of each experiment
        self.csv_filename_its_ext = os.path.join(dir,'soft_its_ext_data.csv')                 #<--- desired csv name here
        self.csv_file_its_ext = open(self.csv_filename_its_ext, 'w')
        self.csv_writer_its_ext = csv.writer(self.csv_file_its_ext)
        self.csv_writer_its_ext.writerow(['Experiment', 'Timestamp [ms]',
                                        'CC_x [mm]','CC_y [mm]','CC_z [mm]','Delta_d_hat [mm]',
                                        'Fn [N]','Ft_x [N]','Ft_y [N]','Ft_z [N]','T [Nmm]',
                                        'Theta_hat [deg]', 'Elapsed Time [ms]'])
        
        # extended vision based initial guess of each experiment
        self.csv_filename_vision = os.path.join(dir,'vision_data.csv')                 #<--- desired csv name here
        self.csv_file_vision = open(self.csv_filename_vision, 'w')
        self.csv_writer_vision = csv.writer(self.csv_file_vision)
        self.csv_writer_vision.writerow(['Experiment', 'Timestamp [ms]',
                                        'CC_x [mm]','CC_y [mm]','CC_z [mm]','Delta_d_hat [mm]'])
        
        # extended vision based initial guess of each experiment
        self.csv_filename_gaussian = os.path.join(dir,'gaussian_data.csv')                 #<--- desired csv name here
        self.csv_file_gaussian = open(self.csv_filename_gaussian, 'w')
        self.csv_writer_gaussian = csv.writer(self.csv_file_gaussian)
        self.csv_writer_gaussian.writerow(['Experiment', 'Timestamp [ms]','Integral'])
        
        # Detected contact of each experiment
        #self.csv_filename_contact = os.path.join(dir,'contact_data.csv')                 #<--- desired csv name here
        #self.csv_file_contact = open(self.csv_filename_contact, 'w')
        #self.csv_writer_contact = csv.writer(self.csv_file_contact)
        #self.csv_writer_contact.writerow(['Experiment', 'Timestamp [ms]'])
                
        # Sync all
        self.csv_filename_all = os.path.join(dir,'all_data.csv')                 #<--- desired csv name here
        self.csv_file_all = open(self.csv_filename_all, 'w')
        self.csv_writer_all = csv.writer(self.csv_file_all)
        self.csv_writer_all.writerow(['Experiment', 'Timestamp [ms]',
                                      'Indentation_cmd [mm]', 'Indentation_meas [mm]', 'Theta [deg]','Theta_m [deg]',
                                      'real_CC_x [mm]','real_CC_y [mm]','real_CC_z [mm]',
                                      'CC_x [mm]','CC_y [mm]','CC_z [mm]', 'e [mm]','e_norm [%]',
                                      'Fn [N]','Ft_x [N]','Ft_y [N]','Ft_z [N]','T [Nmm]',
                                      'PoC_x [mm]','PoC_y [mm]','PoC_z [mm]','Delta_d_hat [mm]',
                                      'theta_hat [deg]', 'integral',
                                      'Convergence_time [ms]', 'Solver'])
        
        # Initialize service
        self.register = False       # Enable log    
        
        self.forces = []                                                        # Force measurment norm [N]
        self.e_cc = []                                                          # Contac Centroid error array [mm]
        self.e_perc_cc = []                                                     # Contac Centroid error array percentage

        self.cc_x = []                                                          # Contac Centroid array x-value [mm]
        self.cc_y = []                                                          # Contac Centroid array y-value [mm]
        self.cc_z = []                                                          # Contac Centroid array z-value [mm]

        self.PoC_x = []                                                         # Point of Contact with vision based algorithm [mm]
        self.PoC_y = []
        self.PoC_z = []

        self.Fn = []                                                            # Normal Force at Contact Centroid [N]
        self.Ft_x = []                                                          # Tangential Force at Contact Centroid x-value [N]
        self.Ft_y = []                                                          # Tangential Force at Contact Centroid y-value [N]
        self.Ft_z = []                                                          # Tangential Force at Contact Centroid z-value [N]
        self.T = []                                                             # Torques along normal at Contact Centroid [Nmm]

        self.dd = []                                                            # Deformation [ms]
        self.theta = []                                                         # CC solution angles array [rad]
        self.times = []                                                         # Times to convergence array [ms]
        self.integrals = []                                                     # Times to convergence array [ms]
        self.slover = ""                                                        # ITS solver name

        self.experiment = 0                                                     # Num of experiment
        self.cmd_indentation = 0                                                # Commanded indentation [mm]
        self.meas_indentation = 0                                               # Measured indentation [mm]
        self.theta_m = 0                                                        # Measured tilt angle [rad]
        self.save = rospy.Service('soft_csp/save_data', Empty, self.handle_save_data)
        self.stop = rospy.Service('soft_csp/stop_save_data', Empty, self.handle_stop_save_data)
        self.set_b = rospy.Service('soft_csp/set_b', Empty, self.handle_set_B)

        # Subscribe to softITS solver and force topic
        self.vision_subscriber = rospy.Subscriber('soft_csp/initial_guess', SoftContactSensingProblemSolution, self.vision_callback) 
        self.softITS_subscriber = rospy.Subscriber('soft_csp/solution', SoftContactSensingProblemSolution, self.its_callback) 
        self.ft_subscriber = rospy.Subscriber('ft_sensor_tactip/netft_data', WrenchStamped, self.ft_callback) 
        self.density_subscriber = rospy.Subscriber("tactip/markers_density", TacTipDensity, self.density_callback)        
        self.contact_subscriber = rospy.Subscriber("/contacting", Bool, self.contact_callback) 

        self.rate = rospy.Rate(20)              # Change the rate as needed
        self.thread = threading.Thread(target=self.thread_loop)
        self.thread.daemon = True

    ##################
    # SERVICEs
    ##################
    def handle_set_B(self, request):
        try:
            self.franka.set_center()
            rospy.loginfo("Fingertip frame B setted at %.2f, %.2f, %.2f [mm]", self.franka.x0[0]*1000.0, self.franka.x0[1]*1000.0, self.franka.x0[2]*1000.0)
        except Exception as error:
            rospy.loginfo(error)
        
        return EmptyResponse()
    
    def handle_save_data(self, request):
        self.experiment = rospy.get_param('/num_exp')
        self.cmd_indentation = rospy.get_param('/indentation')
        self.solver = rospy.get_param('soft_its/algorithm/method/name')
        self.real_cc = self.real_centroid(self.real_theta, self.meas_indentation, params = self.ell_params)
        self.theta_m = (-np.arctan2(self.real_cc[2],self.real_cc[1])+np.pi/2)
        rospy.loginfo("Retrieved experiment: %d", self.experiment)
        rospy.loginfo("Real Contact centroid at %.2f, %.2f, %.2f (theta %.2f, d %.2f)", self.real_cc[0], self.real_cc[1], self.real_cc[2],
            self.theta_m/np.pi*180.0,
            self.meas_indentation)

        self.register = True
        print('=====', self.register)
        
        return EmptyResponse()
    
    def handle_stop_save_data(self, request):
        self.register = False
        self.franka.contact = False
        print('=====', self.register)  
        
        # Eval mean
        mean_cc_x = np.mean(self.cc_x)
        mean_cc_y = np.mean(self.cc_y)
        mean_cc_z = np.mean(self.cc_z)
        mean_dd = np.mean(self.dd)
        mean_theta = np.mean(self.theta)*180.0/np.pi
        mean_error = np.mean(self.e_cc)

        # print summary
        print(f"results:\n x:{mean_cc_x}\n y:{mean_cc_y}\n z: {mean_cc_z}\n Dd: {mean_dd} \n theta: {mean_theta} \n error: {mean_error}")
        
        # reset
        self.forces = []   
        self.e_cc = []  
        self.e_perc_cc = []  

        self.cc_x = []    
        self.cc_y = [] 
        self.cc_z = [] 

        self.Fn = []
        self.Ft_x = []
        self.Ft_y = []
        self.Ft_z = []  
        self.T = []

        self.dd = []                                                    
        self.theta = []                                                        
        self.times = []  

        self.PoC_x = []                       
        self.PoC_y = []
        self.PoC_z = []

        self.integrals = []

        return EmptyResponse()


    ##################
    # CALLBACKs
    ##################

    def its_callback(self, data):
        # Save Soft Contact Sensing Problem Solution
        
        # Log ITS solution
        if self.register == True:
            cc = np.array([data.PoC.x,data.PoC.y,data.PoC.z])   # ITS Contact Centroid in {B} [mm]
            normal = np.array([data.n.x,data.n.y,data.n.z])     # normal direction to fingertip surfaces in PoC
            Fn = data.Fn                                        # amplitude [N] of normal force
            Ft = np.array([data.Ft.x,data.Ft.y,data.Ft.z])      # tangential force [N]
            Ftot = Fn*normal + Ft                               # Resultant force at Contact Centroid [N]
            T = data.T                                          # amplitude [Nmm] of local torque around normal
            dd = data.D
            e = np.linalg.norm(cc-self.real_cc)

            # Contact Centroid estimated angles [rad]
            theta = -np.arctan2(data.PoC.z,data.PoC.y)+np.pi/2
            
            # Contact Centroid convergence time [ms]
            convergence_time = data.convergence_time

            # Append values
            self.e_cc.append(e)
            try:
                e_perc = e/self.meas_indentation
                self.e_perc_cc.append(e_perc)
            except:
                e_perc = 0
            self.cc_x.append(cc[0])
            self.cc_y.append(cc[1])
            self.cc_z.append(cc[2])
            self.Fn.append(Fn)
            self.Ft_x.append(Ft[0])
            self.Ft_y.append(Ft[1])
            self.Ft_z.append(Ft[2])
            self.T.append(T)
            self.dd.append(dd)
            self.theta.append(theta)
            self.times.append(convergence_time)

            # Extended table
            # ---------------
            timestamp = data.header.stamp
            row = [ self.experiment, timestamp, 
                    cc[0],cc[1],cc[2], dd,
                    Fn, Ft[0], Ft[1], Ft[2], T,
                    theta*180.0/np.pi, convergence_time]
            self.csv_writer_its_ext.writerow(row)
    

    def ft_callback(self, data):
        # Save Force measurement
        if self.register == True:
            # Force norm
            f = np.array([data.wrench.force.x,data.wrench.force.y,data.wrench.force.z])
            f = np.linalg.norm(f)

            # Append values
            self.forces.append(f)
            

    def vision_callback(self,data):
        # Save vision estimation
        if self.register == True:
            timestamp = data.header.stamp
            cc_x = data.PoC.x
            cc_y = data.PoC.y
            cc_z = data.PoC.z
            dz = data.D

            self.PoC_x.append(cc_x)
            self.PoC_y.append(cc_y)
            self.PoC_z.append(cc_z)
            
            row = [self.experiment, timestamp, cc_x, cc_y, cc_z, dz]
            self.csv_writer_vision.writerow(row)



    def contactRegion(self,density, density_at_rest, threshold=0.33):
        """
        contactRegion: 
            Estimate multiple contact regions.
            return a numpy array where  contact_region_mask[u,v] = 1 is in a contact region
                                        contact_region_mask[u,v] = 0 otherwise
                   a cv2 contours contact_region_contours
        """
        # Density variation
        DeltaZm = density_at_rest - density
        # Set true where density is decreased more than a threshold
        contact_region_mask = DeltaZm > threshold                           # 3D Array. R(u,v) = is_contact_region
        contact_region_img = 225*contact_region_mask.astype(np.uint8)      # cast to cv2 image type

        # Find Region with Determinant of Hessian procedure  (SLOW)        
        #contact_region_description = blob_doh(contact_region_img,max_sigma=480)
        #contact_region_description = contact_region_description[:,[1, 0, 2]]
        # Find Region as contours   
        #contact_region_contours, hierarchy = cv2.findContours(contact_region_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contact_region_contours = contact_region_img    # <-- cambiato

        return contact_region_mask, contact_region_contours
    
    def density_integration(self,mm2pxl,  density, density_at_rest, R, resolution = 1):
        """
        density_integration: 
            Evaluate density variation by integration over region of contact.
            return integral as float
        """
        # Density variation
        DeltaZm = density_at_rest - density
        # integrate
        integral = np.nansum(DeltaZm[R].flatten())/(mm2pxl*mm2pxl)*resolution # mm2pxl = 10
        #area = np.sum(np.array(R,dtype = int))*resolution

        return integral#/area   
    
    def density_callback(self,data):
        # Save vision estimation
        if self.register == True:
            timestamp = data.header.stamp
            density = np.array(data.density)
            delta_density = np.array(data.delta_density)

            mm2pxl = rospy.get_param("tactip/mm2pxl", 15)
            thr = rospy.get_param("markers_density/threshold", 0.33)
            R_mask,_  = self.contactRegion(density, density + delta_density, threshold=thr)
            integral = self.density_integration(mm2pxl,density, density + delta_density, R_mask)

            self.integrals.append(integral)
            row = [self.experiment, timestamp, integral]
            self.csv_writer_gaussian.writerow(row)


    def contact_callback(self,data):
        self.franka.contact = data.data
        if self.franka.contact:
            self.franka.set_contact_point()

        timestamp = rospy.Time.now()    
        exp = rospy.get_param('/num_exp')    
        #row = [exp, timestamp]
        #self.csv_writer_contact.writerow(row)
        

    ##################
    # UTILs
    ##################
    def real_centroid(self, theta, indentation, params):
        
        
        # Metodo A) intersezione con ellissoide
        #        quadratic:      y = a*x + b*x^2 (Default)
        #x, y, z = sp.symbols('x y z')
        #a,b,c = params
        #
        #print(a,b,c,indentation,theta)
        #ellipsoid = np.power(y/(b-indentation), 2) + np.power(z/(c-indentation), 2) - 1
        #vertical = z - np.tan(np.pi/2 - theta/180.0*np.pi)*y
        #
        #solution = np.array(sp.solve([ellipsoid, vertical], [y, z]))
        #solution = solution[solution[:, 1] > 0].flatten()
        #real_cc = np.array([0.0, solution[0], solution[1]],dtype=float)
        
        # Metodo B) da Franka position
        _,_ = self.franka.get_end_effector_position()
        real_cc = self.franka.x*1000.0


        return real_cc


    def setFrame(self):
        # Set fingertip frame over table link
        parent_frame_id = "table_link"
        child_frame_id = rospy.get_param('fingertip/id')
        t_stamp = rospy.Time.now()
        x,y,z = self.franka.x0
        r,p,y = (self.real_theta/180*np.pi, 0.0, 0.0)
        
        self.tfBroad.sendTransform((x, y, z),
                                    tf.transformations.quaternion_from_euler(r, p, y),
                                    t_stamp,
                                    child_frame_id,
                                    parent_frame_id)

    ##################
    # LOOP FUNCTION
    ##################

    def thread_loop(self):
        while not rospy.is_shutdown(): 
            try: 
                self.setFrame()                                                 # Broadcast {B} frame                                       
                self.meas_indentation = self.franka.get_indentation()          # get real indentation [mm]
                self.real_cc = self.real_centroid(self.real_theta,              # get current e.e. pos in {B}
                                                  self.meas_indentation, 
                                                  params = self.ell_params)

                if self.register:   
                    timestamp = rospy.Time.now()
                    indentation_cmd = self.cmd_indentation
                    solver = self.solver
                    indentation_meas = self.meas_indentation 
                    theta = self.real_theta
                    theta_m = self.theta_m/np.pi*180.0
                    real_cc_x,real_cc_y,real_cc_z = self.real_cc
                    cc_x,cc_y,cc_z = [self.cc_x[-1], self.cc_y[-1], self.cc_z[-1]]
                    e = np.linalg.norm(np.array([cc_x,cc_y,cc_z])-self.real_cc)
                    e_norm = e/indentation_meas
                    Fn = self.Fn[-1]
                    Ft_x = self.Ft_x[-1]
                    Ft_y = self.Ft_y[-1]
                    Ft_z = self.Ft_z[-1]
                    T = self.T[-1]
                    PoC_x = 0#self.PoC_x[-1]#
                    PoC_y = 0#self.PoC_y[-1]#
                    PoC_z = 0#self.PoC_z[-1]#
                    Dd = self.dd[-1]
                    hat_theta = self.theta[-1]*180/np.pi 
                    convergence_time = self.times[-1]
                    integral = 0#self.integrals[-1]#

                    row = [self.experiment, timestamp, 
                        indentation_cmd,indentation_meas,theta,theta_m,
                        real_cc_x,real_cc_y,real_cc_z,
                        cc_x,cc_y,cc_z,e,e_norm,
                        Fn,Ft_x,Ft_y,Ft_z,T,
                        PoC_x,PoC_y,PoC_z,Dd, hat_theta, integral,
                        convergence_time, solver]     
                    self.csv_writer_all.writerow(row)
            except Exception as e:
                print(e)
                pass

            self.rate.sleep()


    def run(self):
        print("Hi from Soft ITS Logger")  
        self.thread.start()
        rospy.spin()


    def __del__(self):
        # Close CSV file when the node is shutting down
        if not self.csv_file_its_ext.closed :
            self.csv_file_its_ext.close()
            self.csv_file_vision.close()
            self.csv_file_gaussian.close()
            self.csv_file_all.close()
            #self.csv_file_contact.close()


# ======================================
#   Main
# ======================================
if __name__ == '__main__':
    try:
        data_logger = DataLogger()
        data_logger.run()
    except rospy.ROSInterruptException:
        pass
