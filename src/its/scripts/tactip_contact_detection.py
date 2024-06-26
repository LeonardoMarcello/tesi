#!/usr/bin/env python3

import rospy
import threading

from its_msgs.msg import SoftContactSensingProblemSolution, TacTipDensity, Point2D
from sensor_msgs.msg import Image
from std_srvs.srv import Empty, EmptyResponse

import numpy as np
import cv2
from cv_bridge import CvBridge

from skimage.feature import blob_doh
from sklearn.neighbors import KernelDensity


##################
# CLASS DEF
##################

class TacTipDetector:
    def __init__(self):
        """ Constructor """
        rospy.init_node('Tactip_detector_node', anonymous=True)
        
        # Parameters
        # fx:                   intrinsic camera param x focal lenght
        # fy:                   intrinsic camera param y focal lenght
        # cx:                   intrinsic camera param x optical center
        # cy:                   intrinsic camera param y optical center
        # dist_coeff:           intrinsic camera param distortion parameter array
        # depth:                fingertip's plane distance from camera [mm]
        #
        # rshape:               Reshaped image dimension rshape = (width, height)
        # mask:                 Image mask (u,v,radius), (Ori: 925,540,420, Fingertip: bho)
        # blur_kernel_size:     Image noise filtering (Gaussian) kernel size   
        # blob_recog:           Determinant of Heussian Blob's recogition params (min_sigma, max_sigma, threshold) 
        #                       (Ori: 6.5,8.5,0.003, Fingertip: bho)
        #
        # h:                    Gaussian Kernel std. dev, (Ori: 15, Fingertip: 30)  
        #
        # resolution:           Resolution of pixels for Density estimation
        #                       i.e. d(u,v) = density[u//resolution,v//resolution]
        #
        # threshold:            Marker Density threshold for contact detection
        #
        # verbose:              Print routine elapsed time
        
        self.fx = rospy.get_param("camera/fx", 1)
        self.fy = rospy.get_param("camera/fy", 1)
        self.cx = rospy.get_param("camera/cx", 320)
        self.cy = rospy.get_param("camera/cy", 240)
        self.dist_coeff = rospy.get_param("camera/dist_coeff", [0,0,0,0,0])
        self.depth = rospy.get_param("camera/depth", 22)

        w = rospy.get_param("image_processing/shape/width", 640) 
        h = rospy.get_param("image_processing/shape/height", 480) 
        self.mask = rospy.get_param("image_processing/mask", "None")
        self.blur_kernel_size = tuple(rospy.get_param("image_processing/blur_kernel_size", [11,11]))
        min_sigma =  rospy.get_param("image_processing/blob_recog/min_sigma", 11)  
        max_sigma =  rospy.get_param("image_processing/blob_recog/max_sigma", 12.5)  
        blob_thresh =  rospy.get_param("image_processing/blob_recog/threshold", 0.008)  

        self.h = rospy.get_param("gaussian_kernel/h", 30)
        self.verbose = rospy.get_param("gaussian_kernel/verbose", False)
        self.resolution = rospy.get_param("markers_density/resolution", 10)
        self.threshold = rospy.get_param("markers_density/threshold", 0.33)
        self.rshape = (w,h)
        if(self.mask=="None"): self.mask=None
        self.blob_recog = (min_sigma, max_sigma, blob_thresh)
        
        # Variables
        self.image = None             # Current Raw TacTip Image
        self.image_stamp = None       # Current Raw TacTip Image timestamp
        self.bridge = CvBridge()      # Ros to OpenCV Image Converter
        self.markers  = None          # Markers Positions numpy Array
        self.M = None                 # Number of Markers in TacTip
        self.gkd = None               # Gaussian Kernel Density Estimator
        self.density = None           # Estimated Marker Density, i.e. d(u,v)
        self.density_at_rest = None   # Estimated Marker Density with no contact, i.e. d0(u,v)
        self.PoC = None               # Estimated Point of Contact in camera frame, i.e. PoC = (u,v)



        # Services
        self.calibrate = rospy.Service("calibrate_at_rest", Empty, self.handle_calibrate)
        # Subscriber to TacTip image data
        self.tactip_subscriber = rospy.Subscriber("usb_cam/image_raw", Image, self.tactip_callback) 
        # Publisher to Contact initial guess
        self.processed_image_publisher = rospy.Publisher("usb_cam/image_processed", Image, queue_size=10) 
        self.data_publisher = rospy.Publisher("tactip/markers_density", TacTipDensity, queue_size=10) 
        self.initial_guess_publisher = rospy.Publisher("soft_csp/initial_guess", SoftContactSensingProblemSolution, queue_size=10) 

        # Initialize Thread
        self.rate = rospy.Rate(1)  # Loop rate [Hz]
        self.thread = threading.Thread(target=self.thread_loop)
        self.thread.daemon = True
        print("Hi from TacTip contact detector")  
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
        raw_image = self.image.copy()
        # 1) Process image
        if (self.mask is None):
            processed_image = self.imageProcessing(raw_image,
                                                resize_shape=self.rshape,
                                                gaussian_kernel_size = self.blur_kernel_size,
                                                mask=False)
        else:
            processed_image = self.imageProcessing(raw_image,
                                                resize_shape=self.rshape,
                                                gaussian_kernel_size = self.blur_kernel_size,
                                                mask_radius=self.mask[2],mask_u_center=self.mask[0],mask_v_center=self.mask[1])
        # 2) Markers detection
        self.markers = self.markerDetection(processed_image,
                                            self.blob_recog[0],self.blob_recog[1],self.blob_recog[2]) 
        self.M = self.markers.shape[0]
        # 3) Density Estim
        self.gkd = self.gaussianKernelDensityFit(self.markers, h = self.h)
        self.density_at_rest = self.gaussianKernelDensityEvaluation(self.gkd,shape=self.rshape,resolution=self.resolution,verbose=True)
        rospy.loginfo("New density at rest setted")
        return EmptyResponse()

    def tactip_callback(self, image_msg):
        """
        tactip_callback: 
            Store raw TacTip image.
        """            
        # store raw image
        self.image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        self.image_stamp = image_msg.header.stamp
        return   
        
    ##################
    # UTILs
    ##################

    def imageProcessing(self, src, gaussian_kernel_size = (5,5), resize_shape = (640,480), mask=True, mask_radius = 420, mask_u_center = 925, mask_v_center = 540):
        """
        imageProcessing: 
            Elaborates raw image to enhance vision alghoritms.
            return elaborated image
        """
        # Mask outer Tactip Frame
        if (mask):
            mask = np.zeros_like(src)
            mask = cv2.circle(mask, (mask_u_center,mask_v_center), mask_radius, (255,255,255), -1)
            masked_img = cv2.bitwise_and(src, mask)
        else:
            masked_img = src.copy()

        # Gaussian Blur filter  
        blur_img = cv2.GaussianBlur(masked_img,gaussian_kernel_size,0)

        # GrayScaling image
        gray_img = cv2.cvtColor(blur_img, cv2.COLOR_RGB2GRAY)

        # resize image
        processed_img = cv2.resize(gray_img, resize_shape)
        
        return processed_img


    def markerDetection(self, src, min_sigma=6.5, max_sigma=8.5, threshold=0.003):
        """
        markerDetection: 
            Detects TacTip Markers' centroids.
            returns a numpy array where markers[i,0] = u-coordinate of i-th marker
                                        markers[i,1] = v-coordinate of i-th marker
        """
        # Detect blob position by Determinant of Hessian procedure
        blobs_doh = blob_doh(src, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        # store markers position
        markers = blobs_doh[:,[1, 0]]
        return markers

    def gaussianKernelDensityFit(self,markers, h=15):
        """
        gaussianKernelDensityFit: 
            Fits Gaussian density given markers position. The std. deviation h is realated to
            the width of each gaussian, it should be close to the mean distance beetwen markers. 
            return a fitted KernelDensity model
        """
        # Fit Gaussian Density Estimation by markers position
        gkd = KernelDensity(kernel='gaussian', bandwidth=h).fit(markers)
        return gkd

    def gaussianKernelDensityEvaluation(self,gkd, shape = (640,480), resolution = 1, verbose = False):
        """
        gaussianKernelDensityEvaluation: 
            Estimate density all over image's pixels.
            return a numpy array where density[u,v] = density of pixel (u*resolution,v*resolution)
        """
        # Get image pixels grid
        u_pxl = np.arange(0,shape[0],resolution)          
        v_pxl = np.arange(0,shape[1],resolution)
        pxls = np.array(np.meshgrid(u_pxl,v_pxl)).T.reshape(-1, 2)
        density_shape = (shape[0]//resolution, shape[1]//resolution)

        # Evaluate density over all pixels
        t_start = rospy.Time.now().to_nsec()*1e-6
        log_density = gkd.score_samples(pxls)
        density = np.exp(log_density.T.reshape(density_shape))*shape[0]*shape[1]   # <-- Controlla unità misura
        t_end = rospy.Time.now().to_nsec()*1e-6
        if verbose:
            rospy.loginfo("Density estimation elapsed time: %.2f ms",(t_end-t_start))
        return density

    def contactRegion(self, density, density_at_rest, threshold=0.33):
        """
        contactRegion: 
            Estimate multiple contact regions.
            return a numpy array where  contact_region_mask[u,v] = 1 is in a contact region
                                        contact_region_mask[u,v] = 0 otherwise
                   a numpy array where  contact_region_description[i,0] = u-coordinate of i-th region's center
                                        contact_region_description[i,1] = v-coordinate of i-th region's center
                                        contact_region_description[i,2] = radius of circle around i-th region
        """
        # Density variation
        DeltaZm = density_at_rest - density
        # Set true where density is decreased more than a threshold
        contact_region_mask = DeltaZm > threshold                           # 3D Array. R(u,v) = is_contact_region
        contact_region_img = 225*contact_region_mask.astype(np.uint8)      # cast to cv2 image type

        # Find Region with Determinant of Hessian procedure 
        contact_region_description = blob_doh(contact_region_img,max_sigma=480)
        contact_region_description = contact_region_description[:,[1, 0, 2]]

        return contact_region_mask, contact_region_description
    
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
            if (self.image is not None):
                # 1) Processing image
                raw_image = self.image.copy()
                stamp = self.image_stamp
                self.image = None

                if (self.mask is None):
                    processed_image = self.imageProcessing(raw_image,
                                                        resize_shape=self.rshape,
                                                        gaussian_kernel_size = self.blur_kernel_size,
                                                        mask=False)
                else:
                    processed_image = self.imageProcessing(raw_image,
                                                        resize_shape=self.rshape,
                                                        gaussian_kernel_size = self.blur_kernel_size,
                                                        mask_radius=self.mask[2],mask_u_center=self.mask[0],mask_v_center=self.mask[1])
                # 2) Markers detection
                self.markers = self.markerDetection(processed_image,
                                                    self.blob_recog[0],self.blob_recog[1],self.blob_recog[2]) 
                M = self.markers.shape[0]
                if (self.M is None):
                    self.M = M
                if (M==0):
                    rospy.logwarn("No Markers Detected")
                    continue
                if (M!=self.M):
                    rospy.logwarn("Detected %i/%i markers", M, self.M)
                
                # 3) Density Estimation
                self.gkd = self.gaussianKernelDensityFit(self.markers, h = self.h)
                self.density = self.gaussianKernelDensityEvaluation(self.gkd, shape=self.rshape,resolution=self.resolution, verbose=self.verbose)
                if self.density_at_rest is None:
                     self.density_at_rest = self.density.copy()                
                
                # 4) Contact regions
                R,R_description = self.contactRegion(self.density,self.density_at_rest,self.threshold)

                # 5) Point of Contact
                if R_description.shape[0] > 0:
                    self.PoC = self.pointOfContact(self.density,self.density_at_rest,self.rshape,self.resolution)
                    # TO DO: estim Normal Force and deformation
                    # TO DO: publish to its solver but in Sensor frame {B}
                    PoC_B = self.camera2body(self.PoC)
                    rospy.loginfo("Contact detected at Pixels (u, v) = (%i, %i) -> (x, y, z) = (%d, %d, %d)",
                                  self.PoC[0],self.PoC[1],
                                  PoC_B[0], PoC_B[1], PoC_B[2])
                    its_msg = SoftContactSensingProblemSolution()
                    its_msg.header.stamp = rospy.Time.now()
                    its_msg.PoC.x = float(PoC_B[0])
                    its_msg.PoC.y = float(PoC_B[1])
                    self.initial_guess_publisher.publish(its_msg)
                else:
                    self.PoC = None

                # 6) Broadcast density and processed image
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
                markers = [] 
                marker = Point2D()
                r = 8
                for i in range(self.markers.shape[0]):
                    marker.u = int(self.markers[i,0])
                    marker.v = int(self.markers[i,1])
                    markers.append(marker)
                    processed_image = cv2.circle(processed_image, (marker.u, marker.v), r, color=(255, 0, 0))
                if (self.PoC is not None):
                    processed_image = cv2.circle(processed_image, self.PoC, 2*r, color=(255, 0, 0))     

                tactip_msg = TacTipDensity()
                tactip_msg.header.stamp = stamp
                tactip_msg.width = self.rshape[0]
                tactip_msg.height = self.rshape[1]
                tactip_msg.resolution = self.resolution  
                tactip_msg.markers = markers
                tactip_msg.density = self.density.flatten()  
                tactip_msg.delta_density = (self.density_at_rest-self.density).flatten()  
                self.data_publisher.publish(tactip_msg)
                
                processed_image_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding="rgb8")
                self.processed_image_publisher.publish(processed_image_msg)
            else:
                rospy.loginfo("No image available")

            # sleep at specified rate
            self.rate.sleep()

    def run(self):
        rospy.spin()


##################
# MAIN
##################
if __name__ == '__main__':
	try:
		tacTipDetector = TacTipDetector()
		tacTipDetector.run()
	except rospy.ROSInterruptException:
		pass