#!/usr/bin/env python3

import rospy
import threading

from its_msgs.msg import TacTipMarkers, TacTipDensity, SoftContactSensingProblemSolution, Point2D
from std_srvs.srv import Empty, EmptyResponse

import numpy as np
from scipy.spatial import ConvexHull, Voronoi 
from scipy.interpolate import CloughTocher2DInterpolator


##################
# CLASS DEF
##################

class TacTipVoronoi:
    def __init__(self):
        """ Constructor """
        rospy.init_node('Tactip_voronoi_node', anonymous=True)
        
        # Parameters
        # fx:                   intrinsic camera param x focal lenght
        # fy:                   intrinsic camera param y focal lenght
        # cx:                   intrinsic camera param x optical center
        # cy:                   intrinsic camera param y optical center
        # dist_coeff:           intrinsic camera param distortion parameter array
        # depth:                fingertip's plane distance from camera [mm]
        #
        # vertices:             List of points to delimit area with markers  
        #                       (they must be in clockwise or counterclockwise order)
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
        #
        # verbose:              Print routine elapsed time
        
        self.fx = rospy.get_param("tactip/camera/fx", 1)
        self.fy = rospy.get_param("tactip/camera/fy", 1)
        self.cx = rospy.get_param("tactip/camera/cx", 320)
        self.cy = rospy.get_param("tactip/camera/cy", 240)
        self.dist_coeff = rospy.get_param("tactip/camera/dist_coeff", [0,0,0,0,0])
        self.depth = rospy.get_param("tactip/camera/depth", 22)

        w = rospy.get_param("image_processing/shape/width", 640) 
        h = rospy.get_param("image_processing/shape/height", 480)
        rate = rospy.get_param("markers_volume/rate", 30)
        self.verbose = rospy.get_param("markers_volume/voronoi/verbose", False)
        ver = rospy.get_param("markers_volume/voronoi/vertices",    [[305,50],
                                                                    [480,153],
                                                                    [480,343],
                                                                    [305,440],
                                                                    [130,343],
                                                                    [130,153]])            
        self.vertices = np.array(ver)   
        self.resolution = rospy.get_param("markers_volume/resolution", 1)
        self.threshold = rospy.get_param("markers_volume/threshold", 500)
        self.fit_type = rospy.get_param("markers_volume/fit/type", "quadratic")
        a = rospy.get_param("markers_volume/fit/a", -0.2112)
        b = rospy.get_param("markers_volume/fit/b", 4.1014)
        self.rshape = (w,h)
        self.fit_params = [a,b]

        # Variables
        self.markers = None             # Markers Positions numpy Array
        self.markers_stamp = None       # Markers Positions timestamp
        self.Voronoi = None             # Voronoi structure
        self.volume = None              # Voronoi volume in numpy array 
                                        # where volume[m,0] = m-th marker u-coordinates
                                        #       volemu[m,1] = m-th marker v-coordinates
                                        #       volume[m,2] = m-th marker cell area
        self.interpolation = None       # Struct for cubi interpolation
        self.volume_fit = None          # Voronoi volume fitted, i.e. Vol(u,v)
        self.volume_fit_at_rest = None  # Voronoi volume fitted with no contact, i.e. Vol0(u,v)
        self.PoC = None                 # Estimated Point of Contact in camera frame, i.e. PoC = (u,v)
        self.deformation = None         # Estimated tip deformation [mm]
        self.times = None                 # Elapsed time array (None to avoid log)

        # Services
        self.calibrate = rospy.Service("calibrate_at_rest", Empty, self.handle_calibrate)
        # Subscriber to TacTip markers tracker
        self.tactip_subscriber = rospy.Subscriber("tactip/markers_tracker", TacTipMarkers, self.markers_callback) 
        # Publisher to Contact initial guess
        self.data_publisher = rospy.Publisher("tactip/markers_density", TacTipDensity, queue_size=10) 
        self.PoC_publisher = rospy.Publisher("tactip/poc", Point2D, queue_size=10) 
        self.initial_guess_publisher = rospy.Publisher("soft_csp/initial_guess", SoftContactSensingProblemSolution, queue_size=10) 
        #  
        # Initialize Thread
        self.rate = rospy.Rate(rate)  # Loop rate [Hz]
        self.thread = threading.Thread(target=self.thread_loop)
        self.thread.daemon = True
        print("Hi from TacTip Voronoi")  
        self.thread.start()


    def __del__(self):
        # Close CSV file when the node is shutting down
        if self.times is not None:
            print(f"Voronoi Elapsed time: {np.mean(self.times)} (± {np.std(self.times)}) ms")
        
    ##################
    # CALLBACKs
    ##################
        
    def handle_calibrate(self, request):  
        """
        handle_calibrate: 
            Store volume at rest condition.
        """            
        rospy.loginfo("Calibrating volume at rest...")
        
        # 1) Markers
        markers = self.markers.copy()
        
        # 2) Density Estim
        self.interpolation = self.cubic_interpolation(markers)
        self.volume_fit_at_rest = self.volume_evaluation(self.interpolation,shape=self.rshape,resolution=self.resolution,verbose=True)
        rospy.loginfo("New volume at rest setted")
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

    def voronoi_cell(self, markers, vertices):
        """
        voronoi_cell: 
            Create voronoi cells from markers centroid position as a numpy array
            where   markers[m,0] = m-th marker u-coordinates
                    markers[m,1] = m-th marker v-coordinates
            return Voronoi structure
        """
        # 1) Extend border to enclose each marker cell
        #u_min = min(markers[:, 0])
        #u_max = max(markers[:, 0])
        #v_min = min(markers[:, 1])
        #v_max = max(markers[:, 1])
        #
        #south_border = np.array([np.arange(u_min-5, u_max+5, 20),                               (v_max+5)*np.ones(len(np.arange(u_min-5, u_max+5, 20)))]).T
        #right_border = np.array([(u_max+5)*np.ones(len(np.arange(v_min-5, v_max+5, 20))),       np.arange(v_min-5, v_max+5, 20)]).T
        #north_border = np.array([np.arange(u_min-5, u_max+5, 20),                               (v_min-5)*np.ones(len(np.arange(u_min-5, u_max+5, 20)))]).T
        #left_border = np.array([(u_min-5)*np.ones(len(np.arange(v_min-5, v_max+5, 20))),        np.arange(v_min-5, v_max+5, 20)]).T
        #border = np.vstack((south_border,right_border,north_border,left_border))

        border = np.empty((0,2), dtype=int)
        for i in range(vertices.shape[0]):
            vi = vertices[i, :]
            v_next = vertices[np.mod(i+1, vertices.shape[0]), :]
            alpha = np.arange(0, 1, 1/3) #1/6
            points = np.array([alpha*vi[0]+(1-alpha)*v_next[0], alpha*vi[1]+(1-alpha)*v_next[1]]).T
            border = np.vstack((border, points))

        # 2) Build Voronoi graph        
        voronoi = Voronoi(np.vstack((markers,border)))
        
        return voronoi
    
    def voronoi_area(self, voronoi):
        """
        voronoi_area: 
            Evaluate each marker's voronoi cell area and return the 3D volume in a numpy array
            where   vol[m,0] = m-th marker u-coordinates
                    vol[m,1] = m-th marker v-coordinates
                    vol[m,2] = m-th marker cell area
            return vol
        """
        vol = np.empty((0,3), float)
        for i, reg_num in enumerate(voronoi.point_region):
            # get indices of vertices of a voronoi cell
            indices = voronoi.regions[reg_num]
            if -1 not in indices: 
                # evaluate area only on regions that are closed
                A = ConvexHull(voronoi.vertices[indices]).volume
                c = voronoi.points[i,:]
                # stack in a numpay array cell center (marker centroid) and cell area
                vol = np.vstack((vol, np.hstack((c,A))))
        return vol
    
    def cubic_interpolation(self,volume):
        """
        cubic_interpolation: 
            Evaluate cubic interpolation of voronoi volume
            return interpolation
        """
        # interpolate by cubic fitting
        interp = CloughTocher2DInterpolator(volume[:, 0:2], volume[:, 2])

        return interp

    def volume_evaluation(self,interp, shape = (640,480), resolution = 1, verbose = False):
        """
        volume_evaluation: 
            Evaluate volume fitting all over image's pixels.
            return a numpy array where volume_fit[u,v] = volaume at pixel (u*resolution,v*resolution)
        """
        # Get image pixels grid
        u_pxls = np.arange(0,shape[0],resolution)          
        v_pxls = np.arange(0,shape[1],resolution)
        u, v = np.meshgrid(u_pxls,v_pxls)
        density_shape = (shape[0]//resolution, shape[1]//resolution)

        # Evaluate volume over all pixels
        t_start = rospy.Time.now().to_nsec()*1e-6
        volume_fit = interp(u,v).T.reshape(density_shape)                       # <-- Controlla unità misura
        t_end = rospy.Time.now().to_nsec()*1e-6
        if verbose:
            rospy.loginfo("Volume evaluation elapsed time: %.2f ms",(t_end-t_start))
        return volume_fit
    
    
    def pointOfContact(self, volume, volume_at_rest, shape = (640,480), resolution = 1):
        """
        pointOfContact: 
            Estimate Centroid of Contact.
            return a tupla (u,v) of the Point of Contact in camera frame {C}
        """
        # Volume variation
        DeltaZm = volume - volume_at_rest
        # Index with maximum decrease
        u,v = divmod(np.nanargmax(DeltaZm), shape[1]//resolution)
        PoC = (u*resolution, v*resolution)
        return PoC
    
    def volume_integration(self,  volume, volume_at_rest, shape = (640,480)):
        """
        volume_integration: 
            Evaluate volume variation by integration over image.
            return integral as float
        """
        # Volume variation
        DeltaZm = volume - volume_at_rest
        # integrate
        integral = np.nansum(DeltaZm.flatten())/(shape[0]*shape[1])
        return integral
        
    def volume2deformation(self, integral, params, type="quadratic"):
        """
        volume2deformation: 
            Estimate tip deformation by volume integral value.. The desired fit type can be parsed 
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
            if self.markers is not None:
                t_start = rospy.Time.now().to_nsec()*1e-6
                # 1) Markers detection
                markers = self.markers.copy()
                markers_stamp = self.markers_stamp
                self.markers = None

                # 2) build Voronoi cells
                self.Voronoi = self.voronoi_cell(markers, self.vertices)   

                # 3) Evaluate area of markers
                self.volume = self.voronoi_area(self.Voronoi)

                # 4) Fit Surface by cubic interpolation
                self.interpolation = self.cubic_interpolation(self.volume)     
                self.volume_fit = self.volume_evaluation(self.interpolation, shape=self.rshape,resolution=self.resolution, verbose=self.verbose)  
                if self.volume_fit_at_rest is None:
                     self.volume_fit_at_rest = self.volume_fit.copy()                
                
                # 5) Point of Contact and deformation
                if np.nanmax(self.volume_fit - self.volume_fit_at_rest) > self.threshold:
                    # estimate PoC
                    self.PoC = self.pointOfContact(self.volume_fit, self.volume_fit_at_rest, self.rshape, self.resolution)
                    PoC_B = self.camera2body(self.PoC)
                    # estimate deformation
                    integral = self.volume_integration(self.volume_fit, self.volume_fit_at_rest, self.rshape)
                    self.deformation = self.volume2deformation(integral, self.fit_params,self.fit_type)
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
                    

                # 6) Broadcast density
                tactip_msg = TacTipDensity()
                tactip_msg.header.stamp = markers_stamp
                tactip_msg.width = self.rshape[0]
                tactip_msg.height = self.rshape[1]
                tactip_msg.resolution = self.resolution  
                tactip_msg.density = self.volume_fit.flatten()  
                tactip_msg.delta_density = (self.volume_fit-self.volume_fit_at_rest).flatten()  
                self.data_publisher.publish(tactip_msg)

                
                t_end = rospy.Time.now().to_nsec()*1e-6
                if self.times is not None: self.times.append(t_end-t_start) 
            #else:
            #    rospy.loginfo("No Markers tracked")
            
            # sleep at specified rate
            self.rate.sleep()
    
    def run(self):
        rospy.spin()


##################
# MAIN
##################
if __name__ == '__main__':
	try:
		tacTipVoronoi = TacTipVoronoi()
		tacTipVoronoi.run()
	except rospy.ROSInterruptException:
		pass