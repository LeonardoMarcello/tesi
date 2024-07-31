#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import WrenchStamped, PoseStamped, TransformStamped
import tf

from its_msgs.msg import SoftContactSensingProblemSolution, TacTipDensity
from std_srvs.srv import Empty, EmptyResponse
import csv
import os
import time
import numpy as np
import sympy as sp
import threading


import moveit_commander
import tactip_gaussian_kernel as tgk
        
# GROUND TRUTH TABLE
#
#           |    0 degrees   |    15 degrees     |    30 degrees
#   0 mm    |   (0,0,4.74)   |  (0,-1.27,4.73)   |  (0,-2.71,4.69)
#   2 mm    |   (0,0,2.74)   |  (0,-0.73,2.73)   |  (0,-1.57,2.72)
#   3 mm    |   (0,0,1.74)   |  (0,-0.47,1.74)   |  (0,-1,1.73)
#   4 mm    |   (0,0,0.74)   |  (0,-0.2,0.74)    |  (0,-0.43,0.74)

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

        self.x0 = np.array([.0, .0, .0])          # centro B in {F} [m]

        self.x = np.array([.0, .0, .0])           # Current Position in {B} [m]
        self.c0 = np.array([.0, -2.71,4.69])      # contact point B in {B} [mm]
        self.x_start = np.array([.0, .0, .0])     # Posizione pre-grasp in {B}

        self.contact = False                      # Condizione di contatto

    def get_end_effector_position(self):
        end_effector_pose = PoseStamped()
        end_effector_pose.pose = self.move_group.get_current_pose(self.end_effector_link).pose        
        end_effector_pose.header.frame_id = self.reference_frame

        ee_in_table = self.tf_listener.transformPose("/table_link", end_effector_pose)
        fingertip_id = rospy.get_param('fingertip/id')
        ee_in_fingertip = self.tf_listener.transformPose(fingertip_id, end_effector_pose)

        self.x = np.array([ee_in_fingertip.pose.position.x, ee_in_fingertip.pose.position.y, ee_in_fingertip.pose.position.z])

        return ee_in_table, ee_in_fingertip
    
    def set_center(self):
        ee,_ = self.get_end_effector_position()
        x = ee.pose.position.x
        y = ee.pose.position.y
        z = ee.pose.position.z

        self.x0 = np.array([x,y,z])

    def set_contact_point(self):
        ee,_ = self.get_end_effector_position()
        x = ee.pose.position.x
        y = ee.pose.position.y
        z = ee.pose.position.z

        self.c0 = np.array([x,y,z])

    def get_indentation(self):
        ee,_ = self.get_end_effector_position()
        x = ee.pose.position.x
        y = ee.pose.position.y
        z = ee.pose.position.z

        c = np.array([x,y,z])
        delta = c - self.c0

        return delta[2]

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

        # Object for retrieval of GroundTrut
        try:
            self.franka = FrankaPosition()
            self.franka.theta = self.real_theta
        except Exception as e:
            print(e)

        self.tfBroad = tf.TransformBroadcaster()
        #self.setFrame()                                            # <---------------- Rimettere

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
                
        # Sync all
        self.csv_filename_all = os.path.join(dir,'all_data.csv')                 #<--- desired csv name here
        self.csv_file_all = open(self.csv_filename_all, 'w')
        self.csv_writer_all = csv.writer(self.csv_file_all)
        self.csv_writer_all.writerow(['Experiment', 'Timestamp [ms]',
                                      'Indentation_cmd [mm]', 'Indentation_meas [mm]', 'Theta [deg]',
                                      'real_CC_x [mm]','real_CC_y [mm]','real_CC_z [mm]',
                                      'CC_x [mm]','CC_y [mm]','CC_z [mm]', 'e [mm]','e_norm [%]',
                                      'Fn [N]','Ft_x [N]','Ft_y [N]','Ft_z [N]','T [Nmm]',
                                      'PoC_x [mm]','PoC_y [mm]','PoC_z [mm]','Delta_d_hat [mm]',
                                      'theta_hat [mm]','Convergence_time [mm]',
                                      'integral'])
        
        # Initialize service
        self.register = False       # Enable log    

        self.real_indent = []                                                   # Measured indentation from Franka [mm]
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
        self.integrals = []                                                         # Times to convergence array [ms]
        self.slover = ""                                                        # ITS solver name

        self.experiment = 0                                                     # Num of experiment
        self.cmd_indentation = 0                                                # Commanded indentation
        self.meas_indentation = 0                                               # Measured indentation
        self.save = rospy.Service('soft_csp/save_data', Empty, self.handle_save_data)
        self.stop = rospy.Service('soft_csp/stop_save_data', Empty, self.handle_stop_save_data)
        self.set_b = rospy.Service('soft_csp/set_b', Empty, self.handle_set_B)

        # Subscribe to softITS solver and force topic
        self.vision_subscriber = rospy.Subscriber('soft_csp/initial_guess', SoftContactSensingProblemSolution, self.vision_callback) 
        self.softITS_subscriber = rospy.Subscriber('soft_csp/solution', SoftContactSensingProblemSolution, self.its_callback) 
        self.ft_subscriber = rospy.Subscriber('ft_sensor_tactip/netft_data', WrenchStamped, self.ft_callback) 
        self.density_subscriber = rospy.Subscriber("tactip/markers_density", TacTipDensity, self.density_callback) 
        self.indent_subscriber = rospy.Subscriber("indent", PoseStamped, self.indent_callback) #<--------------------------- Togliere
        rospy.set_param('/num_exp',1)                                                                   #<--------------------------- Togliere
        rospy.set_param('/indentation',4)                                                               #<--------------------------- Togliere

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
        rospy.loginfo("Retrieved parameter: %d", self.experiment)
        rospy.loginfo("Real Contact centroid at %.2f, %.2f, %.2f", self.real_cc[0], self.real_cc[1], self.real_cc[2])

        self.register = True
        print('=====', self.register)
        
        return EmptyResponse()
    
    def handle_stop_save_data(self, request):
        self.register = False
        #self.franka.contact = False                            #<------------------------- Rimettere
        rospy.set_param('/num_exp',self.experiment+1)                            #<------------------------- togliere
        print('=====', self.register)  
        
        # Eval mean
        mean_cc_x = np.mean(self.cc_x)
        mean_cc_y = np.mean(self.cc_y)
        mean_cc_z = np.mean(self.cc_z)
        mean_theta = np.mean(self.theta)*180.0/np.pi

        # print summary
        print(f"results:\n x:{mean_cc_x}\n y:{mean_cc_y}\n z: {mean_cc_z}\n theta: {mean_theta}")
        
        # reset
        self.real_indent = []
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
        #if not self.franka.contact and np.linalg.norm(np.array([data.PoC.x,data.PoC.y,data.PoC.z])>0.001):     #<---- rimettere
        #    self.franka.set_contact_point()
        if not self.register: self.handle_save_data(None)                                                            #<---- togliere
        
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
                e_perc = e/self.real_indent[-1]
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
        
    def density_callback(self,data):
        # Save vision estimation
        if self.register == True:
            timestamp = data.header.stamp
            density = data.density
            delta_density = data.delta_density

            ttgk = tgk.TacTipGaussianKernel()
            ttgk.mm2pxl = rospy.get_param("tactip/mm2pxl", 15)
            thr = rospy.get_param("markers_density/threshold", 0.33)
            R_mask,_  = ttgk.contactRegion(density, density + delta_density, threshold=thr)
            integral = ttgk.density_integration(density, density + delta_density, R_mask)

            self.integrals.append(integral)
            row = [self.experiment, timestamp, integral]
            self.csv_writer_gaussian.writerow(row)
                
    def indent_callback(self,data):                                 # <---------------------------------- Togliere
        # Save indent estim estimation
        self.meas_indentation = data.pose.position.z
        self.real_indent.append(self.meas_indentation)

    ##################
    # UTILs
    ##################
    def real_centroid(self, theta, indentation, params):
        
        
        # Metodo A) intersezione con ellissoide
        #        quadratic:      y = a*x + b*x^2 (Default)
        x, y, z = sp.symbols('x y z')
        a,b,c = params

        print(a,b,c,indentation,theta)
        ellipsoid = np.power(y/(b-indentation), 2) + np.power(z/(c-indentation), 2) - 1
        vertical = z - np.tan(np.pi/2 - theta/180.0*np.pi)*y
        
        solution = np.array(sp.solve([ellipsoid, vertical], [y, z]))
        solution = solution[solution[:, 1] > 0].flatten()
        real_cc = np.array([0.0, solution[0], solution[1]],dtype=float)
        
        # Metodo B) da Franka position
        # self.franka.get_end_effector_position()
        #real_cc = self.franka.x/1000.0


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
                #self.setFrame()                                                 #<---------------------------------- Rimettere
                #self.real_indent.append(self.franka.get_indentation())
                #self.real_cc = self.franka.x

                if self.register:   
                    timestamp = rospy.Time.now()
                    indentation_cmd = self.cmd_indentation
                    indentation_meas = self.real_indent[-1]
                    theta = self.real_theta
                    real_cc_x,real_cc_y,real_cc_z = self.real_cc
                    cc_x,cc_y,cc_z = [self.cc_x[-1], self.cc_y[-1], self.cc_z[-1]]
                    e = np.linalg.norm(np.array([cc_x,cc_y,cc_z])-self.real_cc)
                    e_norm = e/indentation_meas
                    Fn = self.Fn[-1]
                    Ft_x = self.Ft_x[-1]
                    Ft_y = self.Ft_y[-1]
                    Ft_z = self.Ft_z[-1]
                    T = self.T[-1]
                    PoC_x = self.PoC_x[-1]
                    PoC_y = self.PoC_y[-1]
                    PoC_z = self.PoC_z[-1]
                    Dd = self.dd[-1]
                    hat_theta = self.theta[-1] 
                    convergence_time = self.times[-1]
                    integral = self.integrals[-1]
                    
                    row = [self.experiment, timestamp, 
                        indentation_cmd,indentation_meas,theta,
                        real_cc_x,real_cc_y,real_cc_z,
                        cc_x,cc_y,cc_z,e,e_norm,
                        Fn,Ft_x,Ft_y,Ft_z,T,
                        PoC_x,PoC_y,PoC_z,Dd, hat_theta,
                        convergence_time, integral]       
                    
                    self.csv_writer_all.writerow(row)
            except:
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

if __name__ == '__main__':
    try:
        data_logger = DataLogger()
        data_logger.run()
    except rospy.ROSInterruptException:
        pass
