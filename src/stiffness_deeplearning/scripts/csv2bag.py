#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import csv
import cv2
import os
from cv_bridge import CvBridge
import numpy as np
import argparse
import rosbag

"""
class DataPub:
    def __init__(self):
        
        rospy.init_node('data_publisher_node', anonymous=True)

        self.csv_filename_pose = 'indentation_data.csv'
        self.csv_file_pose = open(self.csv_filename_pose, 'r')
        self.csv_reader_pose = csv.reader(self.csv_file_pose)
        self.pose_header = next(self.csv_reader_pose)
        self.pose_row = None
        # self.csv_writer_pose.writerow(['Experiment','Timestamp', 'Position_Z'])

        # Initialize CSV file 
        self.csv_filename_sensor = 'ati_sensor_franka_data.csv'
        self.csv_file_sensor = open(self.csv_filename_sensor, 'r')
        self.csv_reader_sensor = csv.reader(self.csv_file_sensor)
        self.sensor_header = next(self.csv_reader_sensor)
        self.sensor_row = None
        #self.csv_writer_sensor.writerow(['Experiment','Timestamp', 'Force_x', 'Force_y', 'Force_z', 'Torque_x', 'Torque_y', 'Torque_z']) 

        self.csv_filename_sensor_tactip = 'ati_sensor_tactip_data.csv'
        self.csv_file_sensor_tactip = open(self.csv_filename_sensor_tactip, 'r')
        self.csv_reader_sensor_tactip = csv.reader(self.csv_file_sensor_tactip)
        self.sensor_tactip_header = next(self.csv_reader_sensor_tactip)
        self.sensor_tactip_row = None
        #self.csv_writer_sensor_tactip.writerow(['Experiment','Timestamp', 'Force_x', 'Force_y', 'Force_z', 'Torque_x', 'Torque_y', 'Torque_z']) 

        self.csv_filename_frame = 'tactip_data.csv'
        self.csv_file_frame = open(self.csv_filename_frame, 'r')
        self.csv_reader_frame = csv.reader(self.csv_file_frame)
        self.frame_header = next(self.csv_reader_frame)
        self.frame_row = None
        self.bridge = CvBridge()
        #self.csv_writer_frame.writerow(['Experiment','Timestamp', 'Frame'])
        

        # Initialize service
        self.broadcast = False
        self.starting = True
        self.timestamp0 = None
        self.t0 = 0
        self.h0 = 0
        self.img_calib = True
        self.experiment = 0
        self.start = rospy.Service('start_data', Empty, self.handle_start)
        self.stop = rospy.Service('stop_data', Empty, self.handle_stop)

        
        # Publisher to F/T sensor data from ATI sensor publisher
        #self.sensor_subscriber = rospy.Subscriber('/ft_sensor/netft_data', WrenchStamped, self.sensor_callback) 
        self.sensor_publisher = rospy.Publisher('/ft_sensor_franka/netft_data', WrenchStamped, queue_size=10) 
        self.sensor_tactip_publisher = rospy.Publisher('/ft_sensor_tactip/netft_data', WrenchStamped, queue_size=10) 
        # Publisher to camera topic
        self.image_publisher = rospy.Publisher('/usb_cam/image_raw', Image, queue_size=10)
        # Publisher to displacement topic
        self.pose_publisher = rospy.Publisher('/ground_truth/displacement', PoseStamped, queue_size=10)

        # instance thread loop
        self.thread = threading.Thread(target=self.thread_loop)
        self.thread.daemon = True
        print("Hi from CSV Publisher")  
        self.thread.start()

    def handle_start(self, request):
        self.broadcast = True
        print('=====', self.broadcast)
        
        return EmptyResponse()
    
    def handle_stop(self, request):
        self.broadcast = False
        self.starting = True
        self.sensor_row = None
        self.sensor_tactip_row = None
        self.pose_row = None
        self.frame_row = None
        print('=====', self.broadcast)
        return EmptyResponse()

    ##################
    # Utils
    ##################
    def open_csv(self):
        self.csv_filename_pose = 'indentation_data.csv'
        self.csv_file_pose = open(self.csv_filename_pose, 'r')
        self.csv_reader_pose = csv.reader(self.csv_file_pose)
        self.pose_header = next(self.csv_reader_pose)
        self.pose_row = None
        # self.csv_writer_pose.writerow(['Experiment','Timestamp', 'Position_Z'])

        # Initialize CSV file 
        self.csv_filename_sensor = 'ati_sensor_franka_data.csv'
        self.csv_file_sensor = open(self.csv_filename_sensor, 'r')
        self.csv_reader_sensor = csv.reader(self.csv_file_sensor)
        self.sensor_header = next(self.csv_reader_sensor)
        self.sensor_row = None
        #self.csv_writer_sensor.writerow(['Experiment','Timestamp', 'Force_x', 'Force_y', 'Force_z', 'Torque_x', 'Torque_y', 'Torque_z']) 

        self.csv_filename_sensor_tactip = 'ati_sensor_tactip_data.csv'
        self.csv_file_sensor_tactip = open(self.csv_filename_sensor_tactip, 'r')
        self.csv_reader_sensor_tactip = csv.reader(self.csv_file_sensor_tactip)
        self.sensor_tactip_header = next(self.csv_reader_sensor_tactip)
        self.sensor_tactip_row = None
        #self.csv_writer_sensor_tactip.writerow(['Experiment','Timestamp', 'Force_x', 'Force_y', 'Force_z', 'Torque_x', 'Torque_y', 'Torque_z']) 

        self.csv_filename_frame = 'tactip_data.csv'
        self.csv_file_frame = open(self.csv_filename_frame, 'r')
        self.csv_reader_frame = csv.reader(self.csv_file_frame)
        self.frame_header = next(self.csv_reader_frame)
        self.frame_row = None
        self.bridge = CvBridge()
        #self.csv_writer_frame.writerow(['Experiment','Timestamp', 'Frame'])
    
    def close_csv(self):            
        self.csv_file_pose.close()
        self.csv_file_sensor.close()
        self.csv_file_sensor_tactip.close()
        self.csv_file_frame.close()


    ##################
    # LOOP FUNCTION
    ##################

    def thread_loop(self):
        while not rospy.is_shutdown():
            if (self.broadcast):      
                t = rospy.Time.now().to_nsec()

                # read forces   
                if self.sensor_row is None:
                    try:
                        self.sensor_row = next(self.csv_reader_sensor)
                    except:
                        self.close_csv()
                        self.open_csv()
                        self.starting = True
                        continue

                if self.sensor_tactip_row is None:
                    try:
                        self.sensor_tactip_row = next(self.csv_reader_sensor_tactip)
                    except:
                        self.close_csv()
                        self.open_csv()
                        self.starting = True
                        continue
                # read pose
                if self.pose_row is None:
                    try:
                        self.pose_row = next(self.csv_reader_pose) 
                    except:
                        self.close_csv()
                        self.open_csv()
                        self.starting = True
                        continue
                # read Image
                if self.frame_row is None:
                    try:
                        self.frame_row = next(self.csv_reader_frame)
                    except:
                        self.close_csv()
                        self.open_csv()
                        self.starting = True
                        continue
                if self.starting:
                    self.timestamp0 = min([int(self.sensor_row[1]),int(self.sensor_tactip_row[1]),int(self.pose_row[1]),int(self.frame_row[1])])      
                    self.t0 = rospy.Time.now().to_nsec()
                    self.h0 = float(self.pose_row[2])
                    self.starting = False
                    print("restarting files")

                # Broadcasting
                if (int(self.sensor_row[1])-self.timestamp0 < t-self.t0):			
                    force = WrenchStamped()
                    force.wrench.force.x =  float(self.sensor_row[2])
                    force.wrench.force.y =  float(self.sensor_row[3])
                    force.wrench.force.z =  float(self.sensor_row[4])
                    force.wrench.torque.x = float(self.sensor_row[5])
                    force.wrench.torque.y = float(self.sensor_row[6])
                    force.wrench.torque.z = float(self.sensor_row[7])
                    self.sensor_publisher.publish(force)
                    self.sensor_row = None

                if (int(self.sensor_tactip_row[1]) - self.timestamp0 < t - self.t0):			
                    force = WrenchStamped()
                    force.wrench.force.x =  float(self.sensor_tactip_row[2])
                    force.wrench.force.y =  float(self.sensor_tactip_row[3])
                    force.wrench.force.z =  float(self.sensor_tactip_row[4])
                    force.wrench.torque.x = float(self.sensor_tactip_row[5])
                    force.wrench.torque.y = float(self.sensor_tactip_row[6])
                    force.wrench.torque.z = float(self.sensor_tactip_row[7])
                    self.sensor_tactip_publisher.publish(force)
                    self.sensor_tactip_row = None
                
                if (int(self.pose_row[1]) - self.timestamp0 < t-self.t0):			
                    pose = PoseStamped()
                    disp = (self.h0 - float(self.pose_row[2]))*1000.0
                    pose.pose.position.z = disp
                    self.pose_publisher.publish(pose)
                    self.pose_row = None

                if (int(self.frame_row[1])-self.timestamp0 < t-self.t0):	
                    name = f"image_folder_{int(self.frame_row[0])}/frame_{int(self.frame_row[2]):05d}.png"
                    #if self.img_calib:
                    #    name = "image_folder_12/frame_00001.png"
                    #    self.img_calib=False
                    #else:
                    #    name = "image_folder_12/frame_00035.png"
                    img = cv2.imread(name)
                    image_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                    self.image_publisher.publish(image_msg)
                    self.frame_row = None


    def run(self):
        rospy.spin()

    

    def __del__(self):
        # Close CSV file when the node is shutting down and put frame in .tiff sequence
        if not self.csv_file_pose.closed or not self.csv_file_frame.closed :
            self.csv_file_pose.close()
            self.csv_file_sensor.close()
            self.csv_file_sensor_tactip.close()
            self.csv_file_frame.close()
"""
def open_csv(csv_filename):
    csv_file = open(csv_filename, 'r')
    csv_reader = csv.reader(csv_file)
    print(next(csv_reader))
    
    return csv_file,csv_reader 

def main():
    print("Converting...")

    parser = argparse.ArgumentParser()
    parser.add_argument("-o",type=str,dest='bag_name',help="Name of the output bag. It should end with .bag")
    #parser.add_argument("-f",type=str,action='append',dest='forces', help="Name of the csv files with ATI sensor measurments")
    #parser.add_argument("-p",type=str,action='append',dest='poses', help="Name of the csv files with Franka end effector position")
    #parser.add_argument("-t",type=str,action='append',dest='topics', help="Name of the topics where publishing")
    args = parser.parse_args()
    
    # check on argument parsed
    if args.bag_name is None:
        return -1
    
    # create bag
    bag = rosbag.Bag(args.bag_name, 'w')
    print("bag created")

    # write forces topics
    forces = ["ati_sensor_tactip_data.csv", "ati_sensor_franka_data.csv"]
    topics = ["/ft_sensor_tactip/netft_data", "/ft_sensor_franka/netft_data"]
    for idx,filename in enumerate(forces):
        csv_file,csv_reader = open_csv(filename)
        go = True
        while go:
            try:
                row = next(csv_reader)	

                force = WrenchStamped()
                stamp = rospy.Time(np.double(row[1])/1e9)
                force.header.stamp = stamp
                force.wrench.force.x =  float(row[2])
                force.wrench.force.y =  float(row[3])
                force.wrench.force.z =  float(row[4])
                force.wrench.torque.x = float(row[5])
                force.wrench.torque.y = float(row[6])
                force.wrench.torque.z = float(row[7])

                bag.write(topic = topics[idx], msg=force, t=stamp)
            
            except:
                csv_file.close()
                idx += 1
                go = False

    # write images topics
    images = ["tactip_data.csv"]
    topics = ["/usb_cam/image_raw"]
    bridge = CvBridge()
    for idx,filename in enumerate(images):
        csv_file,csv_reader = open_csv(filename)
        go = True
        while go:
            try:
                row = next(csv_reader)	
                #print(row)	
                exp = int(row[0])
                stamp = rospy.Time(np.double(row[1])/1e9)
                frame = int(row[2])

                name = f'image_folder_{exp}/frame_{frame:05d}.png'

                img = cv2.imread(name)
                image_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
                image_msg.header.stamp = stamp

                bag.write(topic = topics[idx], msg=image_msg, t=stamp)
            
            except:
                csv_file.close()
                idx += 1
                go = False

    print("all messages are correctly stored")
    bag.close()

    return 1 

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
