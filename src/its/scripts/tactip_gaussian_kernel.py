#!/usr/bin/env python3

import rospy
import threading

from its_msgs.msg import TacTipMarkers, TacTipDensity, SoftContactSensingProblemSolution, Point2D
from std_srvs.srv import Empty, EmptyResponse

import numpy as np
import cv2


##################
# CLASS DEF
##################

class TacTipGaussianKernel:
    def __init__(self):
        """ Constructor """
        rospy.init_node('Tactip_gaussian_kernel_node', anonymous=True)
        
        # Parameters
        # fx:                   intrinsic camera param x focal lenght
        # fy:                   intrinsic camera param y focal lenght
        # cx:                   intrinsic camera param x optical center
        # cy:                   intrinsic camera param y optical center
        # dist_coeff:           intrinsic camera param distortion parameter array
        # depth:                fingertip's plane distance from camera [mm]
        # mm2pxl:               conversion millimeter to pixel in fingertip's plane
        #
        # h:                    Gaussian Kernel std. dev, (Ori: 15, Fingertip: 30)  
        #
        # rshape:               Reshaped image dimension rshape = (width, height)
        # resolution:           Resolution of pixels for Density estimation
        #                       i.e. d(u,v) = density[u//resolution,v//resolution]
        #
        # threshold:            Marker Density threshold for contact detection
        #
        # fit_type:             Type of curve used in deformation estimation
        # fit_params:           Fitting curve parameters (a = stiff_params[0], b = stiff_params[1], ...)
        #
        # verbose:              Print routine elapsed time
        
        self.fx = rospy.get_param("tactip/camera/fx", 1)
        self.fy = rospy.get_param("tactip/camera/fy", 1)
        self.cx = rospy.get_param("tactip/camera/cx", 320)
        self.cy = rospy.get_param("tactip/camera/cy", 240)
        self.dist_coeff = rospy.get_param("tactip/camera/dist_coeff", [0,0,0,0,0])
        self.depth = rospy.get_param("tactip/camera/depth", 22)
        self.mm2pxl = rospy.get_param("tactip/mm2pxl", 15)


        rate = rospy.get_param("markers_density/rate", 50)
        self.h = rospy.get_param("markers_density/gaussian_kernel/h", 30)
        self.verbose = rospy.get_param("markers_density/gaussian_kernel/verbose", False)
        w = rospy.get_param("image_processing/shape/width", 640) 
        h = rospy.get_param("image_processing/shape/height", 480) 
        self.resolution = rospy.get_param("markers_density/resolution", 1)
        self.threshold = rospy.get_param("markers_density/threshold", 0.33)
        self.fit_type = rospy.get_param("markers_density/fit/type", "quadratic")
        a = rospy.get_param("markers_density/fit/a", -0.2112)
        b = rospy.get_param("markers_density/fit/b", 4.1014)
        self.rshape = (w,h)
        self.fit_params = [a,b]
        
        # Variables
        self.markers_stamp = None     # Current Markers Positions timestamp
        self.markers  = None          # Markers Positions numpy Array
        self.density = None           # Estimated Marker Density, i.e. d(u,v)
        self.density_at_rest = None   # Estimated Marker Density with no contact, i.e. d0(u,v)
        self.PoC = None               # Estimated Point of Contact in camera frame, i.e. PoC = (u,v)
        self.deformation = None       # Estimated tip deformation [mm]



        # Services
        self.calibrate = rospy.Service("calibrate_at_rest", Empty, self.handle_calibrate)
        # Subscriber to TacTip image data
        self.tactip_subscriber = rospy.Subscriber("tactip/markers_tracker", TacTipMarkers, self.markers_callback) 
        # Publisher to Contact initial guess 
        self.data_publisher = rospy.Publisher("tactip/markers_density", TacTipDensity, queue_size=10) 
        self.PoC_publisher = rospy.Publisher("tactip/poc", Point2D, queue_size=10) 
        self.initial_guess_publisher = rospy.Publisher("soft_csp/initial_guess", SoftContactSensingProblemSolution, queue_size=10) 

        # Initialize Thread
        print(rate)
        self.rate = rospy.Rate(rate)  # Loop rate [Hz]
        self.thread = threading.Thread(target=self.thread_loop)
        self.thread.daemon = True
        print("Hi from TacTip Gaussian Density Estimator")  
        self.thread.start()

        
    ##################
    # CALLBACKs
    ##################
    
    def handle_calibrate(self, request):  
        """
        handle_calibrate: 
            Store density at rest condition.
        """            
        rospy.loginfo("Calibrating density at rest...")
        
        # 1) Markers
        markers = self.markers.copy()
        
        # 2) Density Estim
        self.density = self.gaussianKernelDensityEstimation(markers, h=self.h, shape=self.rshape,resolution=self.resolution, verbose=self.verbose)
        rospy.loginfo("New density at rest setted")
        return EmptyResponse()
    
    def markers_callback(self, markers_msg):
        """
        markers_callback: 
            Store markers position.
        """            
        # store markers position in a numpy array
        self.markers = self.markersmsg_to_numpy(markers_msg)
        self.markers_stamp = markers_msg.header.stamp
        return   
        
    ##################
    # UTILs
    ##################

    def markersmsg_to_numpy(self, markers_msg):
        """
        markersmsg_to_numpy: 
            Convert markers msg into numpy array
            where   markers[m,0] = m-th marker u-coordinates
                    markers[m,1] = m-th marker v-coordinates
            return markers array
        """
        M = len(markers_msg.markers)
        markers = np.zeros((M,2),dtype=int)
        for i in range(M):
            markers[i,0] = int(markers_msg.markers[i].u)
            markers[i,1] = int(markers_msg.markers[i].v)
        return markers
    
    def gaussianKernelDensityEstimation(self, markers, h=15, shape = (640,480), resolution = 1, verbose = False):
        """
        gaussianKernelDensityEstimation: 
            Estimate density all over image's pixels.
            It use a Gaussian Kernel convolution approximation to speed up the execution.
            return a numpy array where density[u,v] = density of pixel (u*resolution,v*resolution)
        """
        t_start = rospy.Time.now().to_nsec()*1e-6
        # Create Markers position as image 
        TacTip_area = np.zeros(shape, dtype = float).T
        TacTip_area[markers[:, 1], markers[:, 0]] = 1/markers.shape[0]
        
        # Create Gaussian Kernel for convolution
        k = cv2.getGaussianKernel(6*h+1, h)
        kernel = k @ k.T
        kernel /= kernel.sum()
        
        # Evaluate Convolution
        density_img = cv2.filter2D(src=TacTip_area, ddepth=cv2.CV_32F, kernel=kernel)

        # Normalize on a scale factor (M*mm2pixel)
        #scale_factor = shape[0]*shape[1]
        #scale_factor = markers.shape[0]*100    # TacTip
        scale_factor = markers.shape[0]*self.mm2pxl*self.mm2pxl     # DigiTac

        density = scale_factor * np.array(density_img.T, dtype=float)

        t_end = rospy.Time.now().to_nsec()*1e-6
        if verbose:
            rospy.loginfo("Density estimation elapsed time: %.2f ms",(t_end-t_start))

        return density[0:shape[0]:resolution, 0:shape[1]:resolution]
    

    def contactRegion(self, density, density_at_rest, threshold=0.33):
        """
        contactRegion: 
            Estimate multiple contact regions.
            return a numpy array where  contact_region_mask[u,v] = 1 is in a contact region
                                        contact_region_mask[u,v] = 0 otherwise
                   a cv2 contours contact_region_contours
        """
        # Density variation
        DeltaZm = density_at_rest - density
        # Set true where density is decreased more than a threshold
        contact_region_mask = DeltaZm > threshold                           # 3D Array. R(u,v) = is_contact_region
        contact_region_img = 225*contact_region_mask.astype(np.uint8)      # cast to cv2 image type

        # Find Region with Determinant of Hessian procedure  (SLOW)        
        #contact_region_description = blob_doh(contact_region_img,max_sigma=480)
        #contact_region_description = contact_region_description[:,[1, 0, 2]]
        # Find Region as contours   
        #contact_region_contours, hierarchy = cv2.findContours(contact_region_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contact_region_contours = contact_region_img    # <-- cambiato

        return contact_region_mask, contact_region_contours
    
    def pointOfContact(self, density, density_at_rest, shape = (640,480), resolution = 1):
        """
        pointOfContact: 
            Estimate Centroid of Contact.
            return a tupla (u,v) of the Point of Contact in camera frame {C}
        """
        # Density variation
        DeltaZm = density_at_rest - density
        # Index with maximum decrease
        u,v = divmod(np.argmax(DeltaZm), shape[1]//resolution)
        PoC = (u*resolution, v*resolution)
        return PoC
    
    def density_integration(self,  density, density_at_rest, R, resolution = 1):
        """
        density_integration: 
            Evaluate density variation by integration over region of contact.
            return integral as float
        """
        # Density variation
        DeltaZm = density_at_rest - density
        # integrate
        integral = np.nansum(DeltaZm[R].flatten())/100*resolution # mm2pxl = 10
        #area = np.sum(np.array(R,dtype = int))*resolution

        return integral#/area
        
    def density2deformation(self, integral, params, type="quadratic"):
        """
        density2deformation: 
            Estimate tip deformation by density integral value. The desired fit type can be parsed 
            as parameter type.
                linear:         y = a + b*x
                quadratic:      y = a*x + b*x^2 (Default)
                power:          y = a*x^b
                logarithmic:    y = a*ln(b*x)
            return deformation as float
        """
        # Volume variation
        if type=="linear":             
            deformation = params[0]*integral + params[1]*integral*integral
        elif type=="quadratic":             
            deformation = params[0]*integral + params[1]*integral*integral
        elif type=="power":             
            deformation = params[0]*np.power(integral,params[1])
        elif type=="logarithmic":             
            deformation = params[0]*np.log(integral*params[1])
        else:
            deformation = params[0]*integral + params[1]*integral*integral
        return deformation
    
    def camera2body(self,PoC_C):
        """
        camera2body: 
            Coverts Centroid of Contact position (u,v) in Camera frame {C} into 3D 
            coordinates (x,y,z) in Sensor Body frame {B}.
            return a tupla (x,y,z) of the Point of Contact in body frame
        """
        u = PoC_C[0]
        v = PoC_C[1]
        u0 = self.cx
        v0 = self.cy
        fx = self.fx
        fy = self.fy
        z = self.depth

        x = (u-u0)/fx*z
        y = (v-v0)/fy*z

        PoC_B = (x,y,z)

        return PoC_B

    ##################
    # LOOP FUNCTION
    ##################

    def thread_loop(self):
        while not rospy.is_shutdown():
            if (self.markers is not None):
                # 1) Markers detection
                markers = self.markers.copy()
                stamp = self.markers_stamp
                self.markers = None
                
                # 2) Density Estimation
                self.density = self.gaussianKernelDensityEstimation(markers, h=self.h, shape=self.rshape,resolution=self.resolution, verbose=self.verbose)
                if self.density_at_rest is None:
                     self.density_at_rest = self.density.copy()                
                
                # 3) Contact regions
                R_mask, R_contours = self.contactRegion(self.density,self.density_at_rest,self.threshold)

                # 4) Point of Contact
                #if len(R_contours) > 0:
                if np.any(R_mask) > 0:
                    # estimate PoC
                    self.PoC = self.pointOfContact(self.density,self.density_at_rest, self.rshape,self.resolution)
                    PoC_B = self.camera2body(self.PoC)
                    # estimate deformation
                    integral = self.density_integration(self.density, self.density_at_rest, R_mask, self.resolution)
                    self.deformation = self.density2deformation(integral, self.fit_params,self.fit_type)

                    # broadcast initial guess
                    rospy.loginfo("Contact detected at Pixels (u, v, def) = (%i, %i, %.2f) -> (x, y, z) = (%.2f, %.2f, %.2f)",
                                  self.PoC[0],self.PoC[1], self.deformation,
                                  PoC_B[0], PoC_B[1], PoC_B[2]-self.deformation)
                    
                    its_msg = SoftContactSensingProblemSolution()
                    its_msg.header.stamp = rospy.Time.now()
                    its_msg.PoC.x = float(PoC_B[0])
                    its_msg.PoC.y = float(PoC_B[1])
                    its_msg.PoC.z = float(PoC_B[2]-self.deformation)
                    its_msg.D = float(self.deformation)
                    self.initial_guess_publisher.publish(its_msg)
                    
                    # broadcast PoC
                    PoC = Point2D()
                    PoC.u = self.PoC[0]
                    PoC.v = self.PoC[1]
                    self.PoC_publisher.publish(PoC)
                else:
                    self.PoC = None
                    
                    its_msg = SoftContactSensingProblemSolution()
                    its_msg.header.stamp = rospy.Time.now()
                    its_msg.PoC.x = 0
                    its_msg.PoC.y = 0
                    its_msg.PoC.z = self.depth
                    its_msg.D = 0
                    self.initial_guess_publisher.publish(its_msg)

                # 5) Broadcast density
                tactip_msg = TacTipDensity()
                tactip_msg.header.stamp = stamp
                tactip_msg.width = self.rshape[0]
                tactip_msg.height = self.rshape[1]
                tactip_msg.resolution = self.resolution  
                tactip_msg.density = self.density.flatten()  
                tactip_msg.delta_density = (self.density_at_rest-self.density).flatten()  
                self.data_publisher.publish(tactip_msg)
                

            #else:
            #   rospy.loginfo("No Markers tracked")

            # sleep at specified rate
            self.rate.sleep()

    def run(self):
        rospy.spin()


##################
# MAIN
##################
if __name__ == '__main__':
	try:
		tacTipGaussianKernel = TacTipGaussianKernel()
		tacTipGaussianKernel.run()
	except rospy.ROSInterruptException:
		pass