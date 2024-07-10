#!/usr/bin/env python3

import rospy
from its_msgs.msg import SoftContactSensingProblemSolution, TacTipDensity, Point2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import tifffile

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
matplotlib.use("Qt5agg")
matplotlib.interactive(True)

class ITSTacTipViz:
    def __init__(self):
        
        rospy.init_node('its_tactip_viz', anonymous=True)
        
        # node rate
        self.rate = rospy.Rate(10) # 10hz

        # Estimated markers density param
        self.density = None             # density array d[u,v]
        self.delta_density = None       # density variation array delta_d[u,v]
        self.shape = None               # density shape in pixels (width, heigth)
        self.resolution = None          # density pixels resolution. Each element of d represent a square of (res x res) pixels
                                        # i.e. d(u,v) = d[u//res, v//res]
        # plot variable
        self.plot = False               # Enable density plot
        self.figure = None              # density figure
        self.axis = None                # density figur axis
        self.surface = None             # density surface
        self.heatmap = None             # density heatmap
        self.cbar = None                # density color bar

        
        # video variable
        self.video = False
        self.img_mod = None
        
        # Define the camera matrix 
        fx = 4.29201828e+03 #800
        fy = 4.35180704e+03 #800
        cx = 1.46397969e+03 #640
        cy = 6.52048204e+02 #480
        self.camera_matrix = np.array([[fx, 0, cx], 
                                        [0, fy, cy], 
                                        [0, 0, 1]], np.float32) 
        # Define the distortion coefficients 
        #self.dist_coeffs = np.zeros((5, 1), np.float32) 
        self.dist_coeffs = np.array([[ 1.20176541e+00, 
                                      -2.09036715e+01, 
                                      -2.33688961e-02,  
                                      1.48544434e-02, 
                                      -4.78303623e-01]], np.float32) 
        # Define the rotation and translation vectors 
        self.rvec = np.zeros((3, 1), np.float32) 
        self.tvec = np.zeros((3, 1), np.float32) 

        
        # image to broadcast
        self.image = None
        self.bridge = CvBridge()
        
        # Subscribe to its slution
        self.PoC_subscriber = rospy.Subscriber("tactip/poc", Point2D, self.poc_callback)
        # Subscribe to TacTip density 
        self.density_subscriber = rospy.Subscriber("tactip/markers_density", TacTipDensity, self.tactip_callback)
        # Subscribe to camera topic
        self.camera_subscriber = rospy.Subscriber("usb_cam/image_raw", Image, self.camera_callback)
        # Publisher to camera Poc topic
        self.poc_publisher = rospy.Publisher("usb_cam/image_poc", Image,queue_size=10)


    ##################
    # CALLBACKs
    ##################
    def poc_callback(self,poc):
        poc_img = self.image.copy()
        poc_img = cv2.circle(poc_img, (poc.u,poc.v), 10, color=(255, 0, 0))  
        poc_img_msg = self.bridge.cv2_to_imgmsg(poc_img, encoding="rgb8")
        self.poc_publisher.publish(poc_img_msg)

    def its_solver_callback(self, its_msg):          
        # Define the 3D point in the Body coordinate system {B} 
        PoC = np.array([[[its_msg.PoC.x*1000.0, its_msg.PoC.y*1000.0, its_msg.PoC.z*1000.0]]], np.float32) 
               
        # Map the 3D point in {B} to 2D point in {C}
        point_2d, _ = cv2.projectPoints(PoC, 
                                        self.rvec, self.tvec, 
                                        self.camera_matrix, 
                                        self.dist_coeffs) 
        
        # Display the 2D point 
        print("2D Point:", point_2d)         
        
        # Plot 2D points 
        #img = np.zeros((144,256),  
        #            dtype=np.uint8)
        #point_2d[0] = point_2d[0] 
        #img_mod = self.image.copy()
        #for point in point_2d.astype(int): 
        #    img_mod = cv2.circle(img_mod, tuple(point[0]), 40, (0,0,255) , -1)
        #
        #
        #img_mod = cv2.resize(img_mod,(640,280))
        #cv2.imshow('Image', img_mod) 
        #cv2.waitKey(1) 
        return   
    


    def tactip_callback(self, tactip_msg):
        # store new image
        h = tactip_msg.height
        w = tactip_msg.width
        res = tactip_msg.resolution
        # store density
        self.density = np.array(tactip_msg.density).reshape(w//res,h//res)
        self.delta_density = np.array(tactip_msg.delta_density).reshape(w//res,h//res)
        self.shape = (w,h)
        self.resolution = res
        self.plot = True
        return   
    

    def camera_callback(self, camera_msg):
        # store new image
        self.image = self.bridge.imgmsg_to_cv2(camera_msg, desired_encoding='passthrough')
        return   
        

    ##################
    # UTILs
    ##################

    def setPlot(self):
        # 3D Plot
        #self.figure, self.axis = plt.subplots(subplot_kw={"projection": "3d"})
        #self.figure.suptitle('Markers Density Estimation', fontsize=16)

        # Heatmap
        self.figure = plt.figure()
        self.axis = self.figure.gca()
        self.figure.suptitle('2D heatmap', fontsize=16)

        plt.gcf().canvas.manager.set_window_title('TacTipViz')
        return

    def plotDensity(self, density, shape = (640,480), resolution=1):
        """
        plotDensity: 
            Plot into a 3D graph the densities.
        """
        # Grid UV
        u_pxl = np.arange(0, shape[0], resolution)          # XY-pixels
        v_pxl = np.arange(0, shape[1], resolution)
        Xm, Ym = np.meshgrid(u_pxl, v_pxl)
        # Z values
        Zm = density[Xm//resolution, Ym//resolution]

        #self.axis.cla()
        #self.axis.set_xlabel('$v$')
        #self.axis.set_ylabel('$u$')
        #self.axis.set_zlabel('$d$')
        #if (self.surface is not None): 
        #    self.surface.remove()
        #self.surface = self.axis.plot_surface(Ym, Xm, Zm, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #
        #if (self.cbar is None): 
        #    self.cbar = self.figure.colorbar(self.surface, shrink=0.5, aspect=5)

        if (self.heatmap is not None): 
            self.heatmap.set_data(Zm)
        else:
            self.axis.set_xlabel('$v$')
            self.axis.set_ylabel('$u$')
            self.heatmap = plt.imshow(Zm, cmap=cm.coolwarm, interpolation='nearest')
            plt.colorbar(self.heatmap, shrink=0.5, aspect=5)
        
        plt.clim(0, .04)
        #plt.clim(0, 1)
        #plt.clim(0, 4)
        #plt.clim(0, 550)

        plt.draw()
        plt.pause(0.02)


    def run(self):
        print("Hi from TacTip viz") 
        self.setPlot()  
        while not rospy.is_shutdown():
            if (self.plot):
                self.plotDensity(self.delta_density.copy(), (self.image.shape[1],self.image.shape[0]), self.resolution)
                #self.plotDensity(self.density.copy(), (self.image.shape[1],self.image.shape[0]), self.resolution)
                self.plot = False

            #if (self.video):
            #    cv2.imshow('Image', self.img_mod) 
            #    cv2.waitKey(1) 
            #    self.video = False

            self.rate.sleep()

    def __del__(self):
        cv2.destroyAllWindows() 
        return

if __name__ == '__main__':
    try:
        its_tactip_viz = ITSTacTipViz()
        its_tactip_viz.run()
    except rospy.ROSInterruptException:
        pass