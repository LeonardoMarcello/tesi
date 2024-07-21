#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import WrenchStamped, PoseStamped
from its_msgs.msg import SoftContactSensingProblemSolution
from std_srvs.srv import Empty, EmptyResponse
import csv
import os
import numpy as np
import sympy as sp
import threading

class DataLogger:
    def __init__(self):
        
        rospy.init_node('its_data_logger_node', anonymous=True)
        
        # Parameters (Ellipsoid size for real cc estimation)
        a = rospy.get_param("fingertip/principalSemiAxis/a", 20)
        b = rospy.get_param("fingertip/principalSemiAxis/b", 20)
        c = rospy.get_param("fingertip/principalSemiAxis/c", 20)
        self.ell_params = [a, b, c] 


        # Initialize CSV file 
        # summary of ITS solver of each experiment
        self.csv_filename_its = 'soft_its_data.csv'
        self.csv_file_its = open(self.csv_filename_its, 'w')
        self.csv_writer_its = csv.writer(self.csv_file_its)
        self.csv_writer_its.writerow(['Experiment', 'Indentation [N]', 'force [N]', 
                                      'CC_x [mm]','CC_y [mm]','CC_z [mm]','Delta_d_hat [mm]',
                                      'Fn [N]','Ft_x [N]','Ft_y [N]','Ft_z [N]','T [Nmm]',
                                      'Theta_hat [deg]', 'Elapsed Time [ms]', 'Solver'])
        
        # summary of ITS solver std daviation of each experiment
        self.csv_filename_its_std = 'soft_its_std_data.csv'
        self.csv_file_its_std = open(self.csv_filename_its_std, 'w')
        self.csv_writer_its_std = csv.writer(self.csv_file_its_std)
        self.csv_writer_its_std.writerow(['Experiment', 'Indentation [N]', 'force [N]', 
                                      'CC_x [mm]','CC_y [mm]','CC_z [mm]','Delta_d_hat [mm]',
                                      'Fn [N]','Ft_x [N]','Ft_y [N]','Ft_z [N]','T [Nmm]',
                                      'Theta_hat [deg]', 'Elapsed Time [ms]', 'Solver'])
        
        # extended results of ITS solver of each experiment
        self.csv_filename_its_ext = 'soft_its_ext_data.csv'
        self.csv_file_its_ext = open(self.csv_filename_its_ext, 'w')
        self.csv_writer_its_ext = csv.writer(self.csv_file_its_ext)
        self.csv_writer_its_ext.writerow(['Experiment', 'Timestamp [ms]',
                                        'CC_x [mm]','CC_y [mm]','CC_z [mm]','Delta_d_hat [mm]',
                                        'Fn [N]','Ft_x [N]','Ft_y [N]','Ft_z [N]','T [Nmm]',
                                        'Theta_hat [deg]', 'Elapsed Time [ms]'])
        
        # extended vision based initial guess of each experiment
        self.csv_filename_vision = 'vision_data.csv'
        self.csv_file_vision = open(self.csv_filename_vision, 'w')
        self.csv_writer_vision = csv.writer(self.csv_file_vision)
        self.csv_writer_vision.writerow(['Experiment', 'Timestamp [ms]',
                                        'CC_x [mm]','CC_y [mm]','CC_z [mm]','Delta_d_hat [mm]'])
        

        
        # All
        self.csv_filename_all = 'all_data.csv'
        self.csv_file_all = open(self.csv_filename_all, 'w')
        self.csv_writer_all = csv.writer(self.csv_file_all)
        self.csv_writer_all.writerow(['Experiment', 'Timestamp [ms]',
                                      'Indentation_cmd [mm]', 'Indentation_meas [mm]', 'Theta [deg]',
                                      'real_CC_x [mm]','real_CC_y [mm]','real_CC_z [mm]',
                                      'CC_x [mm]','CC_y [mm]','CC_z [mm]', 'e [mm]','e_norm [%]'
                                      'Fn [N]','Ft_x [N]','Ft_y [N]','Ft_z [N]','T [Nmm]',
                                      'PoC_x [mm]','PoC_y [mm]','PoC_z [mm]','Delta_d_hat [mm]'])
        
        # Initialize service
        self.register = False                                                   # Enable log
        
        # GROUND TRUTH TABLE
        #
        #           |   0 degrees    |    15 degrees     |    30 degrees
        #   2 mm    |   (0,0,2.74)   | (0,-0.73,2.73)    |  (0,-1.57,2.72)
        #   3 mm    |   (0,0,1.74)   | (0,-0.47,1.74)    |  (0,-1,1.73)
        #   4 mm    |   (0,0,0.74)   |  (0,-0.2,0.74)    |  (0,-0.43,0.74)

        self.real_cc = np.array([0,0,0.8],dtype=float)                          # <---- Real Contact Centroid in {B} [mm]
        self.real_theta = 0                                                     # <---- Real Theta [deg]

        self.real_indent = []                                                   # Measured indentation from Franka [mm]
        self.forces = []                                                        # Force measurment norm [N]
        self.e_cc = []                                                          # Contac Centroid error array [mm]

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
        self.slover = ""                                                        # ITS solver name

        self.experiment = 0                                                     # Num of experiment
        self.cmd_indentation = 0                                                # Commanded indentation
        self.meas_indentation = 0                                               # Measured indentation
        self.save = rospy.Service('soft_csp/save_data', Empty, self.handle_save_data)
        self.stop = rospy.Service('soft_csp/stop_save_data', Empty, self.handle_stop_save_data)

        # Subscribe to softITS solver and force topic
        self.vision_subscriber = rospy.Subscriber('soft_csp/initial_guess', SoftContactSensingProblemSolution, self.vision_callback) 
        self.softITS_subscriber = rospy.Subscriber('soft_csp/solution', SoftContactSensingProblemSolution, self.its_callback) 
        self.ft_subscriber = rospy.Subscriber('ft_sensor_tactip/netft_data', WrenchStamped, self.ft_callback) 
        self.indent_subscriber = rospy.Subscriber('indentation', PoseStamped, self.indent_callback) 

        print("Hi from Soft ITS Logger")  
        self.rate = rospy.Rate(20)              # Change the rate as needed
        self.thread = threading.Thread(target=self.loop)
        self.thread.daemon = True
        self.thread.start()

    ##################
    # SERVICEs
    ##################
    def handle_save_data(self, request):
        self.experiment = rospy.get_param('/num_exp')
        self.cmd_indentation = rospy.get_param('/indentation')
        self.solver = rospy.get_param('soft_its/algorithm/method/name')
        self.real_cc = self.real_centroid(self.theta, self.meas_indentation, self.ell_params)
        rospy.loginfo("Retrieved parameter: %d", self.experiment)
        rospy.loginfo("Real Contact centroid at %.2f, %.2f, %.2f", self.real_cc)

        self.register = True
        print('=====', self.register)
        
        return EmptyResponse()
    
    def handle_stop_save_data(self, request):
        self.register = False
        print('=====', self.register)  
        
        # Eval mean
        mean_real_indent = np.mean(self.real_indent)
        mean_f = np.mean(self.forces)
        mean_e = np.mean(self.e_cc)
        
        mean_cc_x = np.mean(self.cc_x)
        mean_cc_y = np.mean(self.cc_y)
        mean_cc_z = np.mean(self.cc_z)
        
        mean_Fn = np.mean(self.Fn)
        mean_Ft_x = np.mean(self.Ft_x)
        mean_Ft_y = np.mean(self.Ft_y)
        mean_Ft_z = np.mean(self.Ft_z)
        mean_T = np.mean(self.T)
        
        mean_dd = np.mean(self.dd)
        mean_theta = np.mean(self.theta)*180.0/np.pi
        mean_time = np.mean(self.times)
        name = self.solver

        # Eval std. dev
        std_real_indent = np.std(self.real_indent)
        std_f = np.std(self.forces)
        std_e = np.std(self.e_cc)
        std_cc_x = np.std(self.cc_x)
        std_cc_y = np.std(self.cc_y)
        std_cc_z = np.std(self.cc_z)
        
        std_Fn = np.std(self.Fn)
        std_Ft_x = np.std(self.Ft_x)
        std_Ft_y = np.std(self.Ft_y)
        std_Ft_z = np.std(self.Ft_z)
        std_T = np.std(self.T)

        std_dd = np.std(self.dd)
        std_theta = np.std(self.theta*180.0/np.pi)
        std_time = np.std(self.times)

        # print summary
        print(f"results:\n x:{mean_cc_x}\n y:{mean_cc_y}\n z: {mean_cc_z}\n theta: {mean_theta}")

        # store mean
        row = [self.experiment, mean_real_indent, mean_f, 
               mean_cc_x,mean_cc_y,mean_cc_z, mean_dd,
               mean_Fn,mean_Ft_x,mean_Ft_y,mean_Ft_z, mean_T,
               mean_theta, mean_time, name]
        self.csv_writer_its.writerow(row)

        # Store std. dev
        row = [self.experiment, std_real_indent, std_f, 
               std_cc_x,std_cc_y,std_cc_z, std_dd,
               std_Fn, std_Ft_x, std_Ft_y, std_Ft_z, std_T,
               std_theta, std_time, name]
        self.csv_writer_its_std.writerow(row)
        
        # reset
        self.real_indent = []   
        self.forces = []   
        self.e_cc = []  

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

        return EmptyResponse()


    ##################
    # CALLBACKs
    ##################

    def its_callback(self, data):
        # Save Soft Contact Sensing Problem Solution
        if self.register == True:
            # Short Table
            # ---------------
            # Contact Centroid error in {B} [mm]
            cc = np.array([data.PoC.x,data.PoC.y,data.PoC.z])
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
    
    def indent_callback(self, data):
        # Save indentation value measurement
        self.meas_indentation = data.pose.position.z
        if self.register == True:
            # Measured indentation 
            self.real_indent.append(self.meas_indentation)

            
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
        
    
    ##################
    # UTILs
    ##################
    def real_centroid(self,theta,indentation,params):
        x, y, z = sp.symbols('x y z')
        a,b,c = params

        ellipsoid = np.power(y/(b-indentation), 2) + np.power(z/(c-indentation), 2) - 1
        vertical = z - np.tan(np.pi/2 + theta/180.0*np.pi)*y
        
        solution = np.array(sp.solve([ellipsoid, vertical], [y, z]))
        solution = solution[solution[:, 1] > 0].flatten()
        real_cc = np.array([0.0, solution[0], solution[1]])
        
        return real_cc



    ##################
    # LOOP FUNCTION
    ##################

    def thread_loop(self):
        while not rospy.is_shutdown():        
            timestamp = rospy.Time.now()
            indentation_cmd = self.cmd_indentation
            indentation_meas = self.real_indent[-1]
            theta = self.real_theta
            real_cc_x,real_cc_y,real_cc_z = self.real_cc
            cc_x,cc_y,cc_z = [self.cc_x[-1],self.cc_y[-1],self.cc_z[-1]]
            e = np.linalg.norm(np.array(cc_x,cc_y,cc_z)-self.real_cc)
            e_norm = e/indentation_meas
            Fn = self.Fn[-1]
            Ft_x = self.Ft_x[-1]
            Ft_y = self.Ft_y[-1]
            Ft_z = self.Ft_z[-1]
            T = self.T[-1]
            PoC_x =self.PoC_x[-1]
            PoC_y =self.PoC_y[-1]
            PoC_z =self.PoC_z[-1]
            Dd = self.dd[-1]
            
            row = [self.experiment, timestamp, 
                   indentation_cmd,indentation_meas,theta,
                   real_cc_x,real_cc_y,real_cc_z,
                   cc_x,cc_y,cc_z,e,e_norm,
                   Fn,Ft_x,Ft_y,Ft_z,T,
                   PoC_x,PoC_y,PoC_z,Dd]          
            self.csv_writer_all.writerow(row)

            self.rate.sleep()




    def run(self):
        rospy.spin()

    def __del__(self):
        # Close CSV file when the node is shutting down
        if not self.csv_file_its.closed :
            self.csv_file_its.close()
            self.csv_file_its_ext.close()
            self.csv_file_vision.close()
            self.csv_file_all.close()

if __name__ == '__main__':
    try:
        data_logger = DataLogger()
        data_logger.run()
    except rospy.ROSInterruptException:
        pass
