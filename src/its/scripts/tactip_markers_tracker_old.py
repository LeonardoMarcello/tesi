#!/usr/bin/env python3

import rospy
import threading

from its_msgs.msg import Point2D, TacTipMarkers
from sensor_msgs.msg import Image

import numpy as np
import cv2
from cv_bridge import CvBridge


##################
# CLASS DEF
##################

class TacTipMarkersTracker:
    def __init__(self):
        """ Constructor """
        rospy.init_node('Tactip_markers_tracker_node', anonymous=True)
        
        # Parameters
        # M:                    number of markers in a TacTip frame (TacTip:127, DigiTac: 110)
        # mask:                 Image mask (u,v,radius), (TacTip: 925,540,420, DigiTac: bho)
        # blur_kernel_size:     Image noise filtering (Gaussian) kernel size   
        # blob_recog:           Determinant of Heussian Blob's recogition params (min_sigma, max_sigma, threshold) 
        #                       (Ori: 6.5,8.5,0.003, Fingertip: bho)
        self.M =  rospy.get_param("tactip/markers", 127)
        w = rospy.get_param("image_processing/shape/width", 640) 
        h = rospy.get_param("image_processing/shape/height", 480) 
        #self.mask = rospy.get_param("image_processing/mask", "None")
        #self.blur_kernel_size = tuple(rospy.get_param("image_processing/blur_kernel_size", [11,11]))
        #min_sigma =  rospy.get_param("image_processing/blob_recog/min_sigma", 11)  
        #max_sigma =  rospy.get_param("image_processing/blob_recog/max_sigma", 12.5)  
        self.mask = rospy.get_param("image_processing/mask", [310,240,195])
        self.blur_kernel_size = tuple(rospy.get_param("image_processing/blur_kernel_size", [5,5]))
        min_sigma =  rospy.get_param("image_processing/blob_recog/min_sigma", 6.5)  
        max_sigma =  rospy.get_param("image_processing/blob_recog/max_sigma", 8.5)  
        blob_thresh =  rospy.get_param("image_processing/blob_recog/threshold", 0.001)
        self.rshape = (w,h)
        if(self.mask=="None"): self.mask=None
        self.blob_recog = (min_sigma, max_sigma, blob_thresh)
        
        # Variables
        self.markers  = None          # Markers Positions as numpy array
        self.bridge = CvBridge()      # Ros to OpenCV Image Converter
        self.image = None             # Current Raw TacTip Image
        self.image_stamp = None       # Current Raw TacTip Image timestamp


        # Subscriber to TacTip image data
        self.tactip_subscriber = rospy.Subscriber("usb_cam/image_raw", Image, self.tactip_callback) 
        # Publisher to Markers tracker
        self.data_publisher = rospy.Publisher("tactip/markers_tracker", TacTipMarkers, queue_size=10)
        self.processed_image_publisher = rospy.Publisher("usb_cam/image_processed", Image, queue_size=10)
        

        # Initialize Thread
        self.rate = rospy.Rate(24)  # Loop rate [Hz]
        self.thread = threading.Thread(target=self.thread_loop)
        self.thread.daemon = True
        print("Hi from TacTip markers tracker")  
        self.thread.start()

        
    ##################
    # CALLBACKs
    ##################
    
    def tactip_callback(self, image_msg):
        """
        tactip_callback: 
            Store raw TacTip image.
        """            
        # 1) Store raw image
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
        # resize image
        resized_img = cv2.resize(src, resize_shape)

        # GrayScaling image
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)

        # Mask outer Tactip Frame
        if (mask):
            mask = np.zeros_like(gray_img)
            mask = cv2.circle(mask, (mask_u_center,mask_v_center), mask_radius, (255,255,255), -1)
            masked_img = cv2.bitwise_and(gray_img, mask)
        else:
            masked_img = gray_img.copy()

        # Gaussian Blur filter  
        #blur_img = cv2.GaussianBlur(masked_img,gaussian_kernel_size,0)

        #create pin mask
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2RGB)
        low_gray = np.array([20, 20, 20])
        upp_gray = np.array([65, 65, 65])
        mask_gray = cv2.inRange(masked_img, low_gray, upp_gray)

        #apply mask to select pin 
        frame_base_pin = cv2.bitwise_and(masked_img,masked_img, mask=mask_gray)
        frame_base_gray = cv2.cvtColor(frame_base_pin, cv2.COLOR_BGR2GRAY)
        (thresh, frame_base_bw) = cv2.threshold(frame_base_gray, 1, 235, cv2.THRESH_BINARY_INV)
        
        return frame_base_bw


    def markerDetection(self, src, min_sigma=6.5, max_sigma=8.5, threshold=0.003):
        """
        markerDetection: 
            Detects TacTip Markers' centroids.
            returns a numpy array where markers[i,0] = u-coordinate of i-th marker
                                        markers[i,1] = v-coordinate of i-th marker
        """
        # Detect blob position by Determinant of Hessian procedure (SLOW ca 10x)
        #blobs_doh = blob_doh(src, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        # markers = blobs_doh[:,[1, 0]]

        # Detect by OpenCv contours
        contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area_min = 30
        area_max = 130
        radius = 1
        markers = np.empty((0,2),dtype=np.float32)
        cX_old=-radius
        cY_old = -radius
        for i in range(0, len(contours)):
            c = contours[i]
            area = cv2.contourArea(c)
            if area >= area_min and area < area_max:  # select dimension of pins area
                moments = cv2.moments(c)
                # calculate x,y coordinate of center
                if moments["m00"] != 0:
                    cX = int(moments["m10"] / moments["m00"])
                    cY = int(moments["m01"] / moments["m00"])
                else:  # avoid division by zero
                    cX, cY = 0, 0
                 
                if (cX > cX_old + radius or cX < cX_old - radius) or \
                        (cY > cY_old + radius or cY < cY_old - radius):  
                    # avoid contours that do not define a pin
                    coord = np.array([cX, cY], dtype=np.float32)
                    markers = np.vstack((markers,coord))
                cX_old = cX
                cY_old = cY
        return markers
    
    ##################
    # LOOP FUNCTION
    ##################

    def thread_loop(self):
        while not rospy.is_shutdown():
            if (self.image is not None):                
                t_start = rospy.Time.now().to_nsec()*1e-6
                # 1) read raw image
                raw_image = self.image.copy()
                raw_image_stamp = self.image_stamp
                self.image = None
                
                # 2) Processing image (elap time ca 1 ms)
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
                    
                # 3) Markers detection (elap time ca 20 ms)
                self.markers = self.markerDetection(processed_image,
                                                    self.blob_recog[0],self.blob_recog[1],self.blob_recog[2]) 
                M = self.markers.shape[0]
                if (self.M is None):
                    self.M = M

                if (M==0):
                    rospy.logwarn("No Markers Detected")
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
                    processed_image_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding="rgb8")
                    processed_image_msg.header.stamp = raw_image_stamp
                    self.processed_image_publisher.publish(processed_image_msg)
                    continue
                if (M!=self.M):
                    rospy.logwarn("Detected %i/%i markers", M, self.M)

                # 4) Broadcast detected markers centroid and processed image (elap time ca 1 ms)
                markers = []
                markers_msg = TacTipMarkers()
                markers_msg.header.stamp = raw_image_stamp
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
                for i in range(self.markers.shape[0]):
                    marker = Point2D()
                    marker.u = int(self.markers[i,0])
                    marker.v = int(self.markers[i,1])
                    markers.append(marker)
                    processed_image = cv2.circle(processed_image, (marker.u, marker.v), 8, color=(0, 0, 255))
                markers_msg.markers = markers.copy()

                self.data_publisher.publish(markers_msg)
                processed_image_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding="rgb8")
                processed_image_msg.header.stamp = raw_image_stamp
                self.processed_image_publisher.publish(processed_image_msg)
                        
                t_end = rospy.Time.now().to_nsec()*1e-6
                if (t_end-t_start > self.rate.sleep_dur.to_nsec()*1e-6):
                    rospy.logwarn("Image processing is slower than node rate, elapsed time: %.2f ms",(t_end-t_start))

            # sleep at specified rate
            self.rate.sleep()

    def run(self):
        rospy.spin()


##################
# MAIN
##################
if __name__ == '__main__':
	try:
		tacTipMarkersTracker = TacTipMarkersTracker()
		tacTipMarkersTracker.run()
	except rospy.ROSInterruptException:
		pass