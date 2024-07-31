#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import csv
import cv2
from cv_bridge import CvBridge
import numpy as np
import argparse
import rosbag

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
    forces = ["ati_sensor_tactip_data.csv"]#, "ati_sensor_franka_data.csv"]
    topics = ["/ft_sensor_tactip/netft_data"]#, "/ft_sensor_franka/netft_data"]
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
            
            except Exception as e:
                print(e)
                csv_file.close()
                idx += 1
                go = False

    # write images topics
    images = []#["tactip_data.csv"]
    topics = []#["/usb_cam/image_raw"]
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
            
            except Exception as e:
                print(e)
                csv_file.close()
                idx += 1
                go = False


    # write indentation topics
    indent = ["indentation_data.csv"]#["tactip_data.csv"]
    topics = ["/indent"]#["/usb_cam/image_raw"]
    for idx,filename in enumerate(indent):
        csv_file,csv_reader = open_csv(filename)
        go = True
        while go:
            try:
                row = next(csv_reader)	

                pose = PoseStamped()
                stamp = rospy.Time(np.double(row[1])/1e9)
                force.header.stamp = stamp
                pose.pose.position.z =  float(row[2])

                bag.write(topic = topics[idx], msg=pose, t=stamp)
            
            except Exception as e:
                print(e)
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
