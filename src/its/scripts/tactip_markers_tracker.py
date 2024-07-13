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
        # mask:                 Image mask type. None, circle, polygon
        #                       (u,v,radius), (TacTip: 925,540,420, DigiTac: bho)
        # blur_kernel_size:     Image noise filtering (Gaussian) kernel size   
        # circle_recog:         Hough Gradient Circle recogition params (min_radius, max_radius, param1, param2, min_distance)
        #                       (Ori: 6.5,8.5,0.003, Fingertip: bho)
        self.M =  rospy.get_param("tactip/markers", None)
        w = rospy.get_param("image_processing/shape/width", 640) 
        h = rospy.get_param("image_processing/shape/height", 480) 
        
        self.mask = rospy.get_param("image_processing/mask", "polygon")
        self.blur_kernel_size = rospy.get_param("image_processing/blur_kernel_size", 5)
        min_radius =  rospy.get_param("image_processing/circle_recog/min_radius", 7)  
        max_radius =  rospy.get_param("image_processing/circle_recog/max_radius", 12)  
        param1 =  rospy.get_param("image_processing/circle_recog/param1", 250)  
        param2 =  rospy.get_param("image_processing/circle_recog/param2", 8)  
        min_distance =  rospy.get_param("image_processing/circle_recog/min_distance", 27)  
        self.rshape = (w,h)
        self.markers_detection_params = (min_radius, max_radius, param1, param2, min_distance)
        
        # Variables
        self.markers  = None                        # Markers Positions as numpy array
        self.colors = 255*np.random.rand(127,3)     # Markers centroid visualise color
        self.bridge = CvBridge()                    # Ros to OpenCV Image Converter
        self.image = None                           # Current Raw TacTip Image
        self.old_image = None                       # Last Raw TacTip Image (used in Optic Flow evale)
        self.image_stamp = None                     # Current Raw TacTip Image timestamp


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
        # 1) store old raw frame
        if self.image is not None:
            self.old_image = self.image.copy()
        else:
            self.old_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        # 2) Store new raw frame
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
        processed_frame = cv2.resize(src, resize_shape)

        # blur
        processed_frame = cv2.medianBlur(processed_frame, 5)

        # GrayScaling image
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2GRAY)

        # Mask outer Tactip Frame
        if (mask=="circle"):
            mask = np.zeros_like(processed_frame)
            mask_u_center = 320
            mask_v_center = 240
            mask_radius = 240
            mask = cv2.circle(mask, (mask_u_center, mask_v_center), mask_radius, (255,255,255), -1)
            processed_frame = cv2.bitwise_and(processed_frame, mask)
        elif (mask=="polygon"):
            borders = np.array([[144, 456],
                        [476, 456],
                        [605, 365],
                        [615,117],
                        [495, 4],
                        [161, 1],
                        [44, 108],
                        [34, 348]])
            mask = np.zeros_like(processed_frame)
            mask = cv2.fillConvexPoly(mask, borders, (255, 255, 255))
            processed_frame = cv2.bitwise_and(processed_frame, mask)
        else:
            processed_frame = processed_frame.copy()

        # Gaussian Blur filter  
        #blur_img = cv2.GaussianBlur(masked_img,gaussian_kernel_size,0)

        # Median Blur filter  
        #blur_img = cv2.medianBlur(masked_img,gaussian_kernel_size,0)

        #create pin mask
        #masked_img = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2RGB)
        #low_gray = np.array([20, 20, 20])
        #upp_gray = np.array([65, 65, 65])
        #mask_gray = cv2.inRange(masked_img, low_gray, upp_gray)

        #apply mask to select pin 
        #frame_base_pin = cv2.bitwise_and(masked_img,masked_img, mask=mask_gray)
        #frame_base_gray = cv2.cvtColor(frame_base_pin, cv2.COLOR_BGR2GRAY)
        #(thresh, frame_base_bw) = cv2.threshold(frame_base_gray, 1, 235, cv2.THRESH_BINARY_INV)
        
        return processed_frame


    def markerDetection(self, src, params):
        """
        markerDetection: 
            Detects TacTip Markers' centroids in a frame.
            returns a numpy array where markers[i,0] = u-coordinate of i-th marker
                                        markers[i,1] = v-coordinate of i-th marker
        """
        # Detect blob position by Determinant of Hessian procedure (SLOW ca 10x)
        #blobs_doh = blob_doh(src, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        # markers = blobs_doh[:,[1, 0]]

        # Detect by OpenCv contours
        #contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #area_min = 30
        #area_max = 130
        #radius = 1
        #markers = np.empty((0,2),dtype=np.float32)
        #cX_old=-radius
        #cY_old = -radius
        #for i in range(0, len(contours)):
        #    c = contours[i]
        #    area = cv2.contourArea(c)
        #    if area >= area_min and area < area_max:  # select dimension of pins area
        #        moments = cv2.moments(c)
        #        # calculate x,y coordinate of center
        #        if moments["m00"] != 0:
        #            cX = int(moments["m10"] / moments["m00"])
        #            cY = int(moments["m01"] / moments["m00"])
        #        else:  # avoid division by zero
        #            cX, cY = 0, 0
        #         
        #        if (cX > cX_old + radius or cX < cX_old - radius) or \
        #                (cY > cY_old + radius or cY < cY_old - radius):  
        #            # avoid contours that do not define a pin
        #            coord = np.array([cX, cY], dtype=np.float32)
        #            markers = np.vstack((markers,coord))
        #        cX_old = cX
        #        cY_old = cY
        
        
        # Detect by OpenCv Circle
        min_radius = params[0]
        max_radius = params[1]
        param1 = params[2]
        param2 = params[3]
        min_distance = params[4]
        circles = np.array(cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, min_distance,
                                param1=param1, param2=param2,
                                minRadius=min_radius, maxRadius=max_radius))
        markers = circles[0,:,0:2]
        return markers

    def markerDetectionManual(self,src, params):
        """
        markerDetectionManual: 
            Detects TacTip Markers' centroids routine with manual mode. It first use automatic methods. 
            Then it opens a windows to select/remove points.
                > left click:    Add centroid point
                > right click:   Remove nearest centroid
                > q:             Quit manual selection windows
        """
        markers = self.markerDetection(src,params)
        
        def manual_select_points_callback(event, x, y, flags, param):
            nonlocal markers
            if event == cv2.EVENT_LBUTTONDOWN:
                markers = np.vstack((markers, np.array([x,y],dtype=np.float32)))
                print("Adding markers")
            if event == cv2.EVENT_RBUTTONDOWN:
                clicked = np.repeat(np.array([[x,y]]), markers.shape[0],axis = 0)
                distances = (markers[:,0]-clicked[:,0])*(markers[:,0]-clicked[:,0]) + (markers[:,1]-clicked[:,1])*(markers[:,1]-clicked[:,1])
                idx = distances.argmin()
                markers = np.delete(markers,idx,axis=0)
                print("Removing marker at index " + str(idx))


        cv2.imshow('Select Markers', src)
        cv2.setMouseCallback('Select Markers', manual_select_points_callback)
        cv2.waitKey(1)
        while(1):
            markers_img = src.copy()
            markers_img_rgb = cv2.cvtColor(markers_img,cv2.COLOR_GRAY2RGB)
            for i in range(markers.shape[0]):
                x = markers[i, 0]
                y = markers[i, 1]
                markers_img = cv2.circle(markers_img_rgb, (int(x), int(y)), 5, self.colors[i,:], -1)
            cv2.imshow('Select Markers',markers_img_rgb)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyWindow('Select Markers')
                break
        return markers
    
    def markersTracking(self, old_gray_frame, gray_frame, old_markers):
        """
        markersTracking: 
            Track TacTip Markers' centroids by using Sparse Optic Flow on last markers known position between
            two consecutive frames.
            returns a numpy array where markers[i,0] = u-coordinate of i-th marker
                                        markers[i,1] = v-coordinate of i-th marker
        """
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,  # Termination method
                10,                                                 # max count
                0.03)                                               # epsilon

        parameter_lucas_kanade = dict(winSize=(15,15),
                                    maxLevel=2,
                                    criteria=criteria)

        new_points, status, errors = cv2.calcOpticalFlowPyrLK(old_gray_frame, gray_frame,
                                                            old_markers.astype(np.float32), None,
                                                            **parameter_lucas_kanade)

        new_markers = new_points.reshape(-1, 2)

        return new_markers


    ##################
    # LOOP FUNCTION
    ##################
    def thread_loop(self):
        while not rospy.is_shutdown():
            if (self.old_image is not None):                
                t_start = rospy.Time.now().to_nsec()*1e-6
                # 1) read raw images
                raw_image = self.image.copy()
                raw_image_stamp = self.image_stamp
                old_raw_image = self.old_image.copy()
                self.old_image = None
                
                # 2) Processing images (elap time ca 1 ms)
                processed_image = self.imageProcessing(raw_image,
                                                        resize_shape=self.rshape,
                                                        gaussian_kernel_size = self.blur_kernel_size,
                                                        mask=self.mask)
                old_processed_image = self.imageProcessing(old_raw_image,
                                                        resize_shape=self.rshape,
                                                        gaussian_kernel_size = self.blur_kernel_size,
                                                        mask=self.mask)
                    
                # 3) Markers detection (elap time ca 20 ms)
                if self.markers is None:
                    self.markers = self.markerDetectionManual(processed_image, self.markers_detection_params) 
                else:
                    self.markers = self.markersTracking(old_processed_image,processed_image, self.markers)

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
                    processed_image = cv2.circle(processed_image, (marker.u, marker.v), 5, self.colors[i,:], -1)
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