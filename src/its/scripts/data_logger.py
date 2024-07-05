#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import WrenchStamped
from its_msgs.msg import SoftContactSensingProblemSolution
from std_srvs.srv import Empty, EmptyResponse
import csv
import os
import numpy as np

class DataLogger:
    def __init__(self):
        
        rospy.init_node('its_data_logger_node', anonymous=True)

        # Initialize CSV file 
        self.csv_filename_its = 'soft_its_data.csv'
        self.csv_file_its = open(self.csv_filename_its, 'w')
        self.csv_writer_its = csv.writer(self.csv_file_its)
        self.csv_writer_its.writerow(['Experiment', 'force [N]', 'CC_error [mm]', 'Alpha [rad]', 'Elapsed Time [ms]'])
        
        # Initialize service
        self.register = False                                                   # Enable log

        self.real_theta = 0                                                     # <---- Real Ellipsoid Inclination [rad]
        self.real_cc = np.array([0,0,0],dtype=float)                            # <---- Real Contact Centroid in {B} [mm]


        self.forces = []                                                        # Force measurment norm [N]
        self.e_cc = []                                                          # Contac Centroid error sarray [mm]
        self.theta = []                                                         # CC solution angles array [rad]
        self.times = []                                                         # Times to convergence array [ms]

        self.experiment = 0                                                     # Num of experiment
        self.save = rospy.Service('soft_csp/save_data', Empty, self.handle_save_data)
        self.stop = rospy.Service('soft_csp/stop_save_data', Empty, self.handle_stop_save_data)

        # Subscribe to softITS solver and force topic
        self.softITS_subscriber = rospy.Subscriber('soft_csp/solution', SoftContactSensingProblemSolution, self.its_callback) 
        self.softITS_subscriber = rospy.Subscriber('ft_sensor_tactip/netft_data', WrenchStamped, self.ft_callback) 

        print("Hi from Soft ITS Logger")  


    def handle_save_data(self, request):
        self.register = True
        print('=====', self.register)
        self.experiment = rospy.get_param('/num_exp')
        rospy.loginfo("Retrieved parameter: %d", self.experiment)
        
        return EmptyResponse()
    
    def handle_stop_save_data(self, request):
        self.register = False
        print('=====', self.register)  
        
        # Eval mean
        mean_f = np.mean(self.forces)
        mean_e = np.mean(self.e_cc)
        mean_theta = np.mean(self.theta)
        mean_time = np.mean(self.times)

        # store
        row = [self.experiment, mean_f, mean_e, mean_theta, mean_time]
        self.csv_writer_its.writerow(row)
        
        # reset
        self.forces = []   
        self.e_cc = []                                                     
        self.theta = []                                                       
        self.times = []   

        return EmptyResponse()

    def its_callback(self, data):
        # Save Soft Contact Sensing Problem Solution
        if self.register == True:
            # Contact Centroid error in {B} [mm]
            cc = np.array([data.PoC.x,data.PoC.y,data.PoC.z])
            e = np.linalg.norm(cc-self.real_cc)

            # Contact Centroid estimated angles [rad]
            theta = np.arctan2(data.PoC.z,data.PoC.x)
            
            # Contact Centroid convergence time [ms]
            time = data.convergence_time

            # Append values
            self.e_cc.append(e)
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

    def run(self):
        rospy.spin()

    def __del__(self):
        # Close CSV file when the node is shutting down
        if not self.csv_file_its.closed :
            self.csv_file_its.close()

if __name__ == '__main__':
    try:
        data_logger = DataLogger()
        data_logger.run()
    except rospy.ROSInterruptException:
        pass
