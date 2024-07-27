#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import Empty, EmptyResponse
import csv
import cv2
import os
import moveit_commander
import threading
from cv_bridge import CvBridge
import numpy as np
import tifffile
#import subprocess

"""
class bagLogger:
    def __init__(self):    
        self.exp = rospy.get_param('/num_exp')
        self.bag_name = f'bag_exp_{self.exp}.bag'
        rospy.on_shutdown(self.stop_recording_handler)

        # Start recording.
        command = "source " + self.record_script
        self.p = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True, cwd=self.record_folder,
                                    executable='/bin/bash')
    
    def start_recording(self):
        global rosbag_process
        # Define the command to start recording
        command = ['rosbag', 'record', '-O', self.bag_name, '-a']
        # Start the recording process
        rosbag_process = subprocess.Popen(command)
        print(f"Started recording to {bag_name}.bag")
    
    def terminate_ros_node(self, s):
        # Adapted from http://answers.ros.org/question/10714/start-and-stop-rosbag-within-a-python-script/
        list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
        list_output = list_cmd.stdout.read()
        retcode = list_cmd.wait()
        assert retcode == 0, "List command returned %d" % retcode
        for str in list_output.split("\n"):
            if (str.startswith(s)):
                os.system("rosnode kill " + str)

    def stop_recording_handler(self):
        rospy.loginfo(rospy.get_name() + ' stop recording.')
        self.terminate_ros_node("/record")
"""
class DataLogger:
    def __init__(self):
        
        rospy.init_node('data_logger_node', anonymous=True)

        # Initialize CSV file 
        # Indentation file
        self.csv_filename_pose = 'indentation_data.csv'
        self.csv_file_pose = open(self.csv_filename_pose, 'w')
        self.csv_writer_pose = csv.writer(self.csv_file_pose)
        #self.csv_writer_pose.writerow(['Experiment','Timestamp', 'Position_Z'])
        self.csv_writer_pose.writerow(['Experiment','Timestamp', 'Indentation'])
        # Franka ATI sensor file 
        self.csv_filename_sensor = 'ati_sensor_franka_data.csv'
        self.csv_file_sensor = open(self.csv_filename_sensor, 'w')
        self.csv_writer_sensor = csv.writer(self.csv_file_sensor)
        self.csv_writer_sensor.writerow(['Experiment','Timestamp', 'Force_x', 'Force_y', 'Force_z', 'Torque_x', 'Torque_y', 'Torque_z']) 
        # second ATI sensor file
        self.csv_filename_sensor_tactip = 'ati_sensor_tactip_data.csv'
        self.csv_file_sensor_tactip = open(self.csv_filename_sensor_tactip, 'w')
        self.csv_writer_sensor_tactip = csv.writer(self.csv_file_sensor_tactip)
        self.csv_writer_sensor_tactip.writerow(['Experiment','Timestamp', 'Force_x', 'Force_y', 'Force_z', 'Torque_x', 'Torque_y', 'Torque_z']) 
        # Tactip image log file
        self.csv_filename_frame = 'tactip_data.csv'
        self.csv_file_frame = open(self.csv_filename_frame, 'w')
        self.csv_writer_frame = csv.writer(self.csv_file_frame)
        self.csv_writer_frame.writerow(['Experiment','Timestamp', 'Frame'])
        self.frame_count = 0
        self.frame_folder = 'tactip_sequence'

        # instance bag variable
        #bag = None
        
        # Initialize service
        self.register = False
        self.saveimg = False
        self.savetiff = False
        self.experiment = 0
        self.h0 = 0
        self.start = rospy.Service('start_indent', Empty, self.handle_start_indent)
        self.save = rospy.Service('save_data', Empty, self.handle_save_data)
        self.stop = rospy.Service('stop_save_data', Empty, self.handle_stop_save_data)

        # Initialize Thread
        # Initialize MoveIt Commander
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.group_name = "panda_arm"  #
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        # Get the end effector link name
        self.end_effector_link = self.move_group.get_end_effector_link()

        self.reference_frame = "world"  # Replace with your desired reference frame
        self.rate = rospy.Rate(20)  # Change the rate as needed
        self.thread = threading.Thread(target=self.save_z_position_loop)
        self.thread.daemon = True
        self.thread.start()

        self.sequence = []
        self.experiment_prec = 0

        # Subscribe to F/T sensor data from ATI sensor publisher
        #self.sensor_subscriber = rospy.Subscriber('/ft_sensor/netft_data', WrenchStamped, self.sensor_callback) 
        self.sensor_subscriber = rospy.Subscriber('/ft_sensor_franka/netft_data', WrenchStamped, self.sensor_callback) 
        self.sensor_subscriber_tactip = rospy.Subscriber('/ft_sensor_tactip/netft_data', WrenchStamped, self.sensor_tactip_callback) 
        
        # Subscrime to camera topic
        self.image_subscriber = rospy.Subscriber('/usb_cam/image_raw', Image, self.tactip_callback)

        # Indentention real publisher
        self.indentation_publisher = rospy.Publisher("/indentation", PoseStamped, queue_size=10) 

    



    def handle_start_indent(self, request):        
        self.experiment = rospy.get_param('/num_exp')
        
        # store start position
        end_effector_pose = PoseStamped()
        end_effector_pose.pose = self.move_group.get_current_pose(self.end_effector_link).pose        
        end_effector_pose.header.frame_id = self.reference_frame

        self.h0 = end_effector_pose.pose.position.z
        print(f"start indentation at h = {self.h0} m")
        
        return EmptyResponse()
    
    def handle_save_data(self, request):
        self.register = True
        print('=====', self.register)
        self.experiment = rospy.get_param('/num_exp')
        rospy.loginfo("Retrieved parameter: %d", self.experiment)
        
        return EmptyResponse()
    
    def handle_stop_save_data(self, request):
        self.register = False
        self.savetiff = True
        rospy.sleep(1) 
        self.h0 = 0
        print('=====', self.register)
        return EmptyResponse()





    def get_end_effector_position(self):
        if self.register == True:

            end_effector_pose = PoseStamped()
            end_effector_pose.pose = self.move_group.get_current_pose(self.end_effector_link).pose
            
            end_effector_pose.header.frame_id = self.reference_frame
            timestamp_pose = rospy.Time.now()
            z_position = end_effector_pose.pose.position.z
            
            #rospy.loginfo(f"End Effector Z Position: {z_position}")
            #row = [self.experiment, timestamp_pose, z_position]

            rospy.loginfo(f"Measured indentation: {self.h0-z_position} m")
            row = [self.experiment, timestamp_pose, (self.h0-z_position)*1000.0]
            self.csv_writer_pose.writerow(row)

            # Broadcast measured indentation
            indent = PoseStamped()
            indent.header.stamp = timestamp_pose
            indent.pose.position.z = (self.h0-z_position)*1000.0
            self.indentation_publisher.publish(indent)
    
    
    def sensor_callback(self, data):
        # Save Franka ATI sensor data to CSV file when the robot is moving
        if self.register == True:
            timestamp_ati = rospy.Time.now()
            force_data = data.wrench.force
            torque_data = data.wrench.torque
            row = [self.experiment, timestamp_ati, force_data.x, force_data.y, force_data.z, torque_data.x, torque_data.y, torque_data.z]
            self.csv_writer_sensor.writerow(row)

    def sensor_tactip_callback(self, data):
        # Save second ATI sensor data to CSV file when the robot is moving
        if self.register == True:
            timestamp_ati = rospy.Time.now()
            force_data = data.wrench.force
            torque_data = data.wrench.torque
            row = [self.experiment, timestamp_ati, force_data.x, force_data.y, force_data.z, torque_data.x, torque_data.y, torque_data.z]
            self.csv_writer_sensor_tactip.writerow(row)

    def tactip_callback(self, image_msg):
        # Save TacTip images
        if not self.saveimg: return

        image_folder = f"image_folder_{self.experiment}"
        if self.register == True:
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            try:
                bridge = CvBridge()
                cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            except cv2.error as e:
                rospy.logerr(f"CV Bridge error: {e}")
                return
            timestamp_frame = image_msg.header.stamp
            self.frame_count += 1
            image_filename = f"{image_folder}/frame_{self.frame_count:05d}.png"
            cv2.imwrite(image_filename, cv_image)
            print(image_filename, 'saved')
            self.csv_writer_frame.writerow([self.experiment,timestamp_frame, self.frame_count])
            self.savetiff = True
              
        if self.register == False and self.savetiff == True:
            # Write the images as a TIFF file
            self.frame_count = 0
            png_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
            png_files.sort()
            images = []

            # Read PNG images and store in the list
            for png_file in png_files:
                image_path = os.path.join(image_folder, png_file)
                img = cv2.imread(image_path)
                images.append(np.array(img))

            output_path = os.path.join(image_folder, f"tactip_sequence_{self.experiment}.tiff")
            tifffile.imwrite(output_path, images)
            print('TIFF filen', f"tactip_sequence_{self.experiment}.tiff", 'saved successfully')   
            self.savetiff = False     

    def save_z_position_loop(self):
        while not rospy.is_shutdown():
            self.get_end_effector_position()
            self.rate.sleep()

    def run(self):
        rospy.spin()

    def __del__(self):
        # Close CSV file when the node is shutting down and put frame in .tiff sequence
        if not self.csv_file_pose.closed or not self.csv_file_frame.closed :
            self.csv_file_pose.close()
            self.csv_file_sensor.close()
            self.csv_file_sensor_tactip.close()
            self.csv_file_frame.close()

if __name__ == '__main__':
    try:
        data_logger = DataLogger()
        data_logger.run()
    except rospy.ROSInterruptException:
        pass
