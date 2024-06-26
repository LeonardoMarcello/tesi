#!/usr/bin/env python3

import rospy
import threading
import os

import numpy as np

from stiffness_deeplearning.classifierManager import ClassifierManager
from stiffness_deeplearning.filter import Filter

from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty, EmptyResponse
import moveit_commander

class Classifier:
	def __init__(self):
		""" 
		Instance of a node to implement the specimen classify task
        """
		rospy.init_node('classifier_node', anonymous=True)
		
		# Initialize Classifier net 
		self.net_path = os.getcwd() + "/src/stiffness_deeplearning/net/mlp/mlp.h5"
		self.weight_net_path = os.getcwd() + "/src/stiffness_deeplearning/net/mlp/augmented_dataset"
		self.labels = ['grasso sottile', 'grasso spesso', 'vena', 'arteria']
		self.classifier = ClassifierManager(self.net_path, labels=self.labels)
		self.classifier.load_weights(self.net_path)
		
		# Initialize Filters
		self.force_filter = Filter(rospy.Time.now().to_sec()*1000.0, omega = 19.7392, queue_size=10)

        # Initialize Services
		self.F0 = 0.0				# z-force offset of contact point [N]
		self.h0 = 0.0				# z-position of contact point [m]
		self.threshold = -0.04		# z-force threshold for collision detection [N]
		self.save = rospy.Service('start_classifier', Empty, self.handle_contact_point)
		self.stop = rospy.Service('stop_classifier', Empty, self.handle_stop)
		
		# Subscriber to F/T sensor data from ATI sensor publisher
		self.sensor_subscriber = rospy.Subscriber('/ft_sensor/netft_data', WrenchStamped, self.sensor_callback) 
		# Publisher of F/T sensor filtered data
		self.filter_publisher = rospy.Publisher('/ft_sensor/filter_data', WrenchStamped, queue_size=10) 
		self.ori_publisher = rospy.Publisher('/ft_sensor/ori_data', WrenchStamped, queue_size=10) 
		
		# Initialize MoveIt Commander
		moveit_commander.roscpp_initialize([])
		self.robot = moveit_commander.RobotCommander()
		self.group_name = "panda_arm"  #
		self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
		self.end_effector_link = self.move_group.get_end_effector_link()
		#self.reference_frame = "world"  # Replace with your desired reference frame
		
		# Initialize Thread
		self.rate = rospy.Rate(33)  # Loop rate [Hz]
		self.thread = threading.Thread(target=self.thread_loop)
		self.thread.daemon = True
		self.thread.start()
			
	# SERVICES
	def handle_contact_point(self, request):
		_,f0 = self.force_filter.get_time_and_measure()
		_,h0 = self.get_end_effector_position()

		self.threshold = f0
		self.F0 = f0
		self.h0 = h0

		rospy.loginfo("Set collision condition with: F0 = %.2f [N], h0 = %.4f [m]", f0,h0)

		self.classifier.is_sensing = True
		print('=====', self.classifier.is_sensing)
		
		return EmptyResponse()

	def handle_stop(self, request):
		self.threshold = -0.04		
		self.classifier.is_sensing = False
		print('=====', self.classifier.is_sensing)

		return EmptyResponse()

	# SUBSCRIBER CALLBACKs
	def sensor_callback(self, data):
		# Filter raw sensor data
		time = rospy.Time.now().to_sec()*1000.0		
		force_data = data.wrench.force
		
		self.force_filter.update(force_data.z, time, method='bilinear')
	
	# UTILS
	def get_end_effector_position(self):
		end_effector_pose = PoseStamped()
		end_effector_pose.pose = self.move_group.get_current_pose(self.end_effector_link).pose		
		time = rospy.Time.now().to_sec()*1000.0
		z_pos = end_effector_pose.pose.position.z
		return (time, z_pos)
	
	def detect_collision(self, force, threshold = -0.04):
		# the collision is detected when Z-Force, in end-effector frame, is negative and in module greater than a threshold
		# Default: threshold -0.04 N 
		if force.z < threshold:
			self.classifier.is_sensing = True
			return True 
		else: 
			self.classifier.is_sensing = False
			return False
		
	def thread_loop(self):
		while not rospy.is_shutdown():
			# get filtered Force
			t_force, z_force = self.force_filter.get_time_and_measure()

			# publish filtered force
			filtered_force = WrenchStamped()
			filtered_force.header.stamp = rospy.Time.from_sec(t_force/1000.0)
			filtered_force.wrench.force.z = z_force
			self.filter_publisher.publish(filtered_force)

			# predict specimen or check for a collision
			if self.classifier.is_sensing:
				# if sensing 
				if not self.detect_collision(filtered_force.wrench.force, self.threshold):
					# if sensing ended stop classifier 
					self.classifier.is_sensing = False
					print('=====', self.classifier.is_sensing)
				else:
					# if keep sensing classify specimen
					t_pos, z_pos = self.get_end_effector_position()
					if np.abs(t_pos - t_force) < 33:
						force = -(z_force-self.F0)
						displacement = -(z_pos-self.h0)*1000.0
						rospy.loginfo("Measure (%.2f [N], %.3f [mm])", force, displacement)
						t_start = rospy.Time.now().to_sec()*1000.0
						c,p = self.classifier.predict(force, displacement)
						t_end = rospy.Time.now().to_sec()*1000.0
						rospy.loginfo("Classification results: class %s, with p = %.2f (elapsed time = %.2f [ms])", c, p, t_end-t_start)
					#else:
					#	rospy.loginfo("Position and force measures have too different timestamp")
			elif self.detect_collision(filtered_force.wrench.force, self.threshold):
				# if no sensing check for a collision 
				_, z_pos = self.get_end_effector_position()
				self.F0 = self.threshold
				self.h0 = z_pos
				print('=====', self.classifier.is_sensing)
				rospy.loginfo("Detected contact with: F0 = %.2f [N], h0 = %.4f [m]", self.F0, self.h0)

			# sleep at specified rate
			self.rate.sleep()

	def run(self):
		rospy.spin()

# MAIN
if __name__ == '__main__':
	try:
		classifier = Classifier()
		classifier.run()
	except rospy.ROSInterruptException:
		pass