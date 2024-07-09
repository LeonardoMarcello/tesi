#!/usr/bin/env python3

import rospy
import roslib

import moveit_commander
import moveit_msgs.msg
import sys
import copy
from scipy.spatial.transform import Rotation as R

import time

import actionlib

import numpy as np

from moveit_msgs.srv import GetPositionIKRequest, GetPositionIK
from moveit_msgs.msg import AllowedCollisionMatrix, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.srv import GetPositionFKRequest, GetPositionFK 

from netft_rdt_driver.srv import String_cmd, String_cmdRequest, String_cmdResponse

from franka_msgs.msg import ErrorRecoveryAction, ErrorRecoveryActionGoal

from dynamic_reconfigure.server import Server
from demo_tactip.cfg import demo_tactip_cfgConfig

import std_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped, PointStamped, PoseArray

import yaml
from std_srvs.srv import Empty #service

TABLE_HEIGHT = 0.131   # 0.016 -- Grasso sottile; 0.021 -- Arteria; 0.022 -- Grasso spesso; 0.017 -- Vena

try:
    from math import pi, tau, dist, fabs, cos
except:  
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi
    
    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

def all_close(goal, actual, tolerance):
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        d = dist((x1, y1, z1), (x0, y0, z0))
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class RobotController(object):
    def __init__(self):
        # super(RobotController, self).__init__()

        moveit_commander.roscpp_initialize(sys.argv)
        robot = moveit_commander.RobotCommander()
        robot_group_name = "panda_arm"
        move_group_robot = robot.get_group(robot_group_name)
#        move_group_robot.set_pose_reference_frame("table_link")
#        move_group_robot.set_planning_pipeline_id("pilz_industrial_motion_planner")
#        move_group_robot.set_planner_id("PTP")
        #print("--------", move_group_robot.get_planner_id())
        robot_planning_frame = move_group_robot.get_planning_frame()
        move_group_robot.set_end_effector_link("brush")
        ee_link = move_group_robot.get_end_effector_link() 

        scene = moveit_commander.PlanningSceneInterface()
        self.robot = robot
        self.scene = scene
        self.move_group_robot = move_group_robot
        self.robot_planning_frame = robot_planning_frame
        self.ee_link = ee_link

        # Commentate
        self.save_data = rospy.ServiceProxy('save_data', Empty)                     # Measurement
        self.stop_save_data = rospy.ServiceProxy('stop_save_data', Empty)           
        self.save_data_its = rospy.ServiceProxy('soft_csp/save_data', Empty)        # Soft Contact Sensing
        self.stop_save_data_its = rospy.ServiceProxy('soft_csp/stop_data', Empty)   

        #self.ft_client = rospy.ServiceProxy('/ft_sensor/bias_cmd', String_cmd)
        self.ft_client_franka = rospy.ServiceProxy('/ft_sensor_franka/bias_cmd', String_cmd)
        self.ft_client_tactip = rospy.ServiceProxy('/ft_sensor_tactip/bias_cmd', String_cmd)
        #self.srv_ft = String_cmd()
        #self.srv_ft.cmd = 'bias'      
        # ##

        #Giulia
        self.robot_group_name = robot_group_name

        self.error_recovery = actionlib.SimpleActionClient('/franka_control/error_recovery', ErrorRecoveryAction)
        self.error_recovery.wait_for_server()
        
        self.srv = Server(demo_tactip_cfgConfig, self.dyn_rec_callback)
        self.table_height = TABLE_HEIGHT - 0.004   # --ori 0.010

        self.experiment = 0
        rospy.set_param('/num_exp', self.experiment) 

    def set_bias(self):
        srv_ft = String_cmdRequest()
        srv_ft.cmd = "bias"

        try:
            response = self.ft_client_franka(srv_ft.cmd)
            rospy.loginfo(f"net_ft res: {response.res}")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call netft bias service: %s", e)

        try:
            response = self.ft_client_tactip(srv_ft.cmd)
            rospy.loginfo(f"net_ft res: {response.res}")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call netft bias service: %s", e)
    
    def dyn_rec_callback(self, config, level):
        rospy.loginfo("""Reconfigure Request: {table_height} """.format(**config))
        self.table_height = config.table_height
        return config

    def prepare_robot(self):
        self.move_group_robot.set_named_target('ready')
        self.move_group_robot.go(wait=True)
        self.move_group_robot.stop()
        self.move_group_robot.clear_pose_targets()

    def execute_plan(self, plan):
        move_group = self.move_group_robot
        executed = move_group.execute(plan, wait=True)
        if not executed:
            print("Trajectory not executed ", executed)
            goal = ErrorRecoveryActionGoal()
            goal.header = std_msgs.msg.Header()

            self.error_recovery.send_goal(goal)
            wait = self.error_recovery.wait_for_result(rospy.Duration(5.0))
            if not wait:
                rospy.logerr("ErrorRecoveryActionGoal Action server not available!")
                
    def go_to_pose(self, target_pose):
        
        move_group = self.move_group_robot
        move_group.set_pose_target(target_pose)        
        
        plan = False
        attempt = 1
        MAX_ATTEMPTS = 3
        while not plan and attempt <= MAX_ATTEMPTS:
            plan = move_group.plan()              
            plan = move_group.go()
            stop = False
            if plan:
                move_group.stop()
                move_group.clear_pose_targets()
            else:
                print("Pose not reached ")
                stop = True
                goal = ErrorRecoveryActionGoal()
                goal.header = std_msgs.msg.Header()

                self.error_recovery.send_goal(goal)
                wait = self.error_recovery.wait_for_result(rospy.Duration(5.0))
                if not wait:
                    rospy.logerr("ErrorRecoveryActionGoal Action server not available!")
            print("Attempt N: ", attempt)
            attempt = attempt + 1  	      
        #print('save_plan ', save_plan)
        return plan, stop
    
	    

def main():
    rospy.init_node("robot_controller", anonymous=True)
    robot_controller_node = RobotController()

    goal = ErrorRecoveryActionGoal()
    goal.header = std_msgs.msg.Header()

    robot_controller_node.error_recovery.send_goal(goal)
    wait = robot_controller_node.error_recovery.wait_for_result(rospy.Duration(5.0))
    
    robot_controller_node.move_group_robot.set_named_target('ready')
    executed = robot_controller_node.move_group_robot.go(wait=True)
    robot_controller_node.move_group_robot.stop()
    robot_controller_node.move_group_robot.clear_pose_targets()

    rospy.set_param('/num_exp', robot_controller_node.experiment)
    
#    goal_c = moveit_msgs.msg.Constraints()
#    orientation_c = moveit_msgs.msg.OrientationConstraint()
#    orientation_c.header = std_msgs.msg.Header()
#    orientation_c.header.frame_id = "world"
#    orientation_c.link_name = "panda_link8"
#    quat = R.from_matrix(np.identity(3)).as_euler("xyz", degrees=True)
#    quat = R.from_euler('xyz', [quat[0]+180, quat[1], quat[2]], degrees=True).as_quat()
#    orientation_c.orientation.x = quat[0]
#    orientation_c.orientation.y = quat[1]
#    orientation_c.orientation.z = quat[2]
#    orientation_c.orientation.w = quat[3]
#    orientation_c.absolute_x_axis_tolerance = 0.03
#    orientation_c.absolute_y_axis_tolerance = 0.03
#    orientation_c.absolute_z_axis_tolerance = 3.14
#    orientation_c.weight = 1.0
#    goal_c.orientation_constraints.append(orientation_c)
#    robot_controller_node.move_group_robot.set_path_constraints(goal_c)

    while True:
        choice = input("============ Press `Enter` to move the robot or q to quite ...")
        if choice == "":
            goal = ErrorRecoveryActionGoal()
            goal.header = std_msgs.msg.Header()
            robot_controller_node.experiment += 1
            rospy.set_param('/num_exp', robot_controller_node.experiment) 

            
            robot_controller_node.error_recovery.send_goal(goal)
            wait = robot_controller_node.error_recovery.wait_for_result(rospy.Duration(5.0))
            print("going to pre-grasp")
            target = PoseStamped()
            target.pose.position.x = 0
            target.pose.position.y = 0
            target.pose.position.z = TABLE_HEIGHT
            quat = R.from_matrix(np.identity(3)).as_euler("xyz", degrees=True)
            quat = R.from_euler('xyz', [quat[0]+180, quat[1], quat[2]], degrees=True).as_quat()
            target.pose.orientation.x = quat[0]
            target.pose.orientation.y = quat[1]
            target.pose.orientation.z = quat[2]
            target.pose.orientation.w = quat[3]
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time(0)
            header.frame_id = 'table_link'
            target.header = header
            robot_controller_node.go_to_pose(target)[0:1]
            rospy.sleep(1) 

            # Commentate <----
            #robot_controller_node.set_bias()
            #resp_ft = robot_controller_node.ft_client(robot_controller_node.srv_ft)
            resp = robot_controller_node.save_data()             # <----- Start log Measurement
            # ##

            ########################################
            #funzione che esegue il task princiaple
            print("going to grasp with heigh ", robot_controller_node.table_height)
            table = PoseStamped()
            table.pose.position.x = 0
            table.pose.position.y = 0
            table.pose.position.z = robot_controller_node.table_height
            quat = R.from_matrix(np.identity(3)).as_euler("xyz", degrees=True)
            quat = R.from_euler('xyz', [quat[0]+180, quat[1], quat[2]], degrees=True).as_quat()
            table.pose.orientation.x = quat[0]
            table.pose.orientation.y = quat[1]
            table.pose.orientation.z = quat[2]
            table.pose.orientation.w = quat[3]
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time(0)
            header.frame_id = 'table_link'
            table.header = header
            robot_controller_node.move_group_robot.set_max_velocity_scaling_factor(0.01)
            robot_controller_node.go_to_pose(table)[0:1]
            resp = robot_controller_node.stop_save_data()             # <----- End log Measurement
            robot_controller_node.move_group_robot.set_max_velocity_scaling_factor(1)

            ########################################
            # (LEO) funzione che esegue il log per l'ITS
            input("Press `Enter` to start ITS solver log")
            resp = robot_controller_node.save_data()                  # <----- Start log Measurement
            resp = robot_controller_node.save_data_its()              # <----- Start log ITS solution
            rospy.sleep(10)                                           #             | sleep in seconds
            resp = robot_controller_node.stop_save_data_its()         # <----- End log ITS solution
            resp = robot_controller_node.stop_save_data()             # <----- End log Measurement

            ################################
            # IO (PAOLO) HO AGGIUNTO LA RIGA DI CODICE SOTTO
            input("Press `Enter` to go back")
            
            print("returning to pre-grasp")
            target = PoseStamped()
            target.pose.position.x = 0
            target.pose.position.y = 0
            target.pose.position.z = TABLE_HEIGHT
            quat = R.from_matrix(np.identity(3)).as_euler("xyz", degrees=True)
            quat = R.from_euler('xyz', [quat[0]+180, quat[1], quat[2]], degrees=True).as_quat()
            target.pose.orientation.x = quat[0]
            target.pose.orientation.y = quat[1]
            target.pose.orientation.z = quat[2]
            target.pose.orientation.w = quat[3]
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time(0)
            header.frame_id = 'table_link'
            target.header = header
            robot_controller_node.go_to_pose(target)[0:1]
            
         
            
            #print("returing to ready")
            #robot_controller_node.move_group_robot.set_named_target('ready')
            #executed = robot_controller_node.move_group_robot.go(wait=True)

            #robot_controller_node.move_group_robot.stop()
            #robot_controller_node.move_group_robot.clear_pose_targets()
        elif choice == "q":
            print("returing to ready")
            robot_controller_node.move_group_robot.set_named_target('ready')
            executed = robot_controller_node.move_group_robot.go(wait=True)
            robot_controller_node.move_group_robot.stop()
            robot_controller_node.move_group_robot.clear_pose_targets()
            
            exit(1)
        else:
            print("wrong input. Repeat please.")
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
