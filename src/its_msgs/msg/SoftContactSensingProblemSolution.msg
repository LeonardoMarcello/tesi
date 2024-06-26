# All values are referenced in the fingertip core center frame {B}

Header header				# stamp
geometry_msgs/Point  PoC	# Point of Contact [mm]
geometry_msgs/Vector3 n     # normal direction to fingertip surfaces in PoC
float64 Fn					# amplitude [N] of normal force
geometry_msgs/Vector3 Ft    # tangential force [N]
float64 T					# amplitude [Nmm] of local torque around normal
float64 D					# deformation [mm] of fingertip surfaces along normal
