#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
from its_msgs.msg import SoftContactSensingProblemSolution
from std_srvs.srv import Empty, EmptyResponse
import csv
import cv2
import os
import threading
import numpy as np

class DataLogger:
    def __init__(self):
        
        rospy.init_node('its_data_logger_node', anonymous=True)

        # Initialize CSV file 
        self.csv_filename_its = 'indentation_data.csv'
        self.csv_file_its = open(self.csv_filename_its, 'w')
        self.csv_writer_its = csv.writer(self.csv_file_its)
        self.csv_writer_its.writerow(['Experiment', 'force [N]', 'CC_error [mm]', 'Alpha [rad]', 'Elapsed Time [ms]'])
        
        # Initialize service
        self.duration = 10000                                                   # Daration [ms]
        self.register = False                                                   # Enable log
        self.starting = True                                                    # First solution
        self.t_start = 0                                                        # First solution timestamp


        self.real_theta = np.array([0,0,0],dtype=float)                         # <---- Real Ellipsoid Inclination [rad]
        self.real_cc = np.array([0,0,0],dtype=float)                            # <---- Real Contact Centroid in {B} [mm]


        self.forces = []                                                        # Force measurment norm [N]
        self.e_cc = []                                                          # Contac Centroid error sarray [mm]
        self.theta = []                                                         # CC solution angles array [rad]
        self.times = []                                                         # Times to convergence array [ms]

        self.experiment = 0                                                     # Num of experiment
        self.save = rospy.Service('soft_csp/save_data', Empty, self.handle_save_data)
        self.stop = rospy.Service('soft_csp/stop_save_data', Empty, self.handle_stop_save_data)

        # Initialize Thread
        self.rate = rospy.Rate(20)  # Change the rate as needed
        self.thread = threading.Thread(target=self.thread_loop)
        self.thread.daemon = True
        self.thread.start()

        # Subscribe to softITS solver
        self.softITS_subscriber = rospy.Subscriber('soft_csp/solution', SoftContactSensingProblemSolution, self.its_callback) 
        self.softITS_subscriber = rospy.Subscriber('ft_sensor_tactip/netft_data', WrenchStamped, self.ft_callback) 


    def handle_save_data(self, request):
        self.register = True
        print('=====', self.register)
        self.experiment = rospy.get_param('/num_exp')
        rospy.loginfo("Retrieved parameter: %d", self.experiment)
        
        return EmptyResponse()
    
    def handle_stop_save_data(self, request):
        self.register = False
        print('=====', self.register)
        return EmptyResponse()

    def its_callback(self, data):
        # Save Soft Contact Sensing Problem Solution
        if self.register == True:
            # Contact Centroid error
            cc = np.array([data.PoC.x,data.PoC.y,data.PoC.z])
            e = np.norm(cc-self.real_cc)

            # Contact Centroid angles
            theta = np.atan2(data.PoC.z,data.PoC.x)
            
            # Contact Centroid convergence time
            time = data.convergence_time

            # Append values
            self.cc_e.append(cc)
            self.theta.append(theta)
            self.times.append(time)
    
    def ft_callback(self, data):
        # Save Soft Contact Sensing Problem Solution
        if self.register == True:
            # Contact Centroid error
            f = np.array([data.wrench.force.x,data.wrench.force.y,data.wrench.force.z])
            f = np.linalg.norm(f)

            # Append values
            self.forces.append(f)

    def thread_loop(self):
        if self.register:
            if self.starting:
                print('=====', self.register)
                self.t_start = rospy.Time.now()
            t = rospy.Time.now()
            if t - self.t_start > self.duration:
                # mean
                mean_f = np.mean(self.forces)
                mean_e = np.mean(self.cc_e)
                mean_theta = np.mean(self.theta)
                mean_time = np.mean(self.times)

                # store
                row = [self.experiment, mean_f, mean_e, mean_theta, mean_time]
                self.csv_writer_sensor.writerow(row)
                
                # reset
                self.forces = []   
                self.cc_e = []                                                     
                self.theta = []                                                       
                self.times = []    
                self.register = False  
                self.starting = True                                 


    def run(self):
        rospy.spin()

    def __del__(self):
        # Close CSV file when the node is shutting down and put frame in .tiff sequence
        if not self.csv_file_its.closed :
            self.csv_file_its.close()

if __name__ == '__main__':
    try:
        data_logger = DataLogger()
        data_logger.run()
    except rospy.ROSInterruptException:
        pass
