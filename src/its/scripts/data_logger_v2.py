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
import argparse


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

        self.fit_type = rospy.get_param("markers_density/fit/type", "quadratic")
        self.a = rospy.get_param("markers_density/fit/a", -0.2112)
        self.b = rospy.get_param("markers_density/fit/b", 4.1014)
        # Parsing arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("-d",type=str,dest='dir',help="directory where data are stored")
        args = parser.parse_args()
        if args.dir is None:
            dir_name = ""
        else:
            dir_name = args.dir 

        # Initialize CSV file     
        DATE = time.strftime('%d_%m_%Y')
        dir = os.path.join(os.path.join("data",DATE,dir_name))    
        if not os.path.exists(dir):
            os.makedirs(dir)

        # extended results of ITS solver of each experiment
        self.csv_filename_its_ext = os.path.join(dir,'soft_its_ext_data.csv')                 #<--- desired csv name here
        self.csv_file_its_ext = open(self.csv_filename_its_ext, 'w')
        self.csv_writer_its_ext = csv.writer(self.csv_file_its_ext)
        self.csv_writer_its_ext.writerow(['Experiment', 'Timestamp [ms]',
                                        'CC_x [mm]','CC_y [mm]','CC_z [mm]','Delta_d_hat [mm]',
                                        'Fn [N]','Ft_x [N]','Ft_y [N]','Ft_z [N]','T [Nmm]',
                                        'Theta_hat [deg]', 'Elapsed Time [ms]', 'method'])
        # Initialize service
        self.register = True       # Enable log    

        self.cc_x = []                                                          # Contac Centroid array x-value [mm]
        self.cc_y = []                                                          # Contac Centroid array y-value [mm]
        self.cc_z = []                                                          # Contac Centroid array z-value [mm]

        self.Fn = []                                                            # Normal Force at Contact Centroid [N]
        self.Ft_x = []                                                          # Tangential Force at Contact Centroid x-value [N]
        self.Ft_y = []                                                          # Tangential Force at Contact Centroid y-value [N]
        self.Ft_z = []                                                          # Tangential Force at Contact Centroid z-value [N]
        self.T = []                                                             # Torques along normal at Contact Centroid [Nmm]

        self.dd = []                                                            # Deformation [ms]
        self.theta = []                                                         # CC solution angles array [rad]
        self.times = []                                                         # Times to convergence array [ms]
        self.integrals = []                                                     # Times to convergence array [ms]
        self.solver = ""                                                        # ITS solver name

        self.experiment = 1                                                     # Num of experiment
        self.save = rospy.Service('soft_csp/save_data', Empty, self.handle_save_data)
        self.stop = rospy.Service('soft_csp/stop_save_data', Empty, self.handle_stop_save_data)

        # Subscribe to softITS solver and force topic
        self.softITS_subscriber = rospy.Subscriber('soft_csp/solution', SoftContactSensingProblemSolution, self.its_callback) 
        self.softITS_subscriber = rospy.Subscriber('/gaussian', PoseStamped, self.gaussian_callback) 
        self.initial_guess_publisher = rospy.Publisher("soft_csp/initial_guess", SoftContactSensingProblemSolution, queue_size=10) 

        self.rate = rospy.Rate(50)              # Change the rate as needed
        self.thread = threading.Thread(target=self.thread_loop)
        self.thread.daemon = True

    ##################
    # SERVICEs
    ##################
       
    def handle_save_data(self, request):
        self.experiment = rospy.get_param('/num_exp')
        self.solver = rospy.get_param('soft_its/algorithm/method/name')
        rospy.loginfo("Retrieved parameter: %d", self.experiment)

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

        self.integrals = []

        return EmptyResponse()


    ##################
    # CALLBACKs
    ##################

    def its_callback(self, data):
        
        if self.register == True:
            if data.PoC.x == data.PoC.y == data.PoC.z == 0: return

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
                    theta*180.0/np.pi, convergence_time, self.solver]
            self.csv_writer_its_ext.writerow(row)

    def gaussian_callback(self, data):
            integral = data.pose.position.z
            
            # Volume variation
            if self.fit_type=="linear":             
                deformation = self.a + self.b*integral
            elif self.fit_type=="quadratic":             
                deformation = self.a*integral + self.b*integral*integral
            elif self.fit_type=="power":             
                deformation = self.a*np.power(integral,self.b)
            elif self.fit_type=="logarithmic":             
                deformation = self.a*np.log(integral*self.b)
            else:
                deformation = self.a*integral + self.b*integral*integral

            #Broacast initial guess                    
            its_msg = SoftContactSensingProblemSolution()
            its_msg.header.stamp = rospy.Time.now()
            its_msg.PoC.x = float(0)
            its_msg.PoC.y = float(0)
            its_msg.PoC.z = float(4)
            its_msg.D = float(deformation)
            self.initial_guess_publisher.publish(its_msg)
            

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

if __name__ == '__main__':
    try:
        data_logger = DataLogger()
        data_logger.run()
    except rospy.ROSInterruptException:
        pass
