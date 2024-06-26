# This message encode information on TacTip markers density in a frame

Header header				    # stamp
its_msgs/Point2D[]  markers	    # Markers centroids list
uint64  width   	            # Frame width 
uint64  height	                # Frame height
uint64  resolution	            # Pixels resolution
float64[]  density	            # Markers density of each pixel in row major encoding
                                # (i.e. d(u,v) = density[u*resolution + v*width*resolution])
float64[]  delta_density        # Markers density variation of each pixel in row major encoding
                                # (i.e. Delta_d(u,v) = delta_density[u*resolution + v*width*resolution])