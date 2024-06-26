Comandi Implementati 
=====================

## _Interazione Franka_ per esperimenti di indentazione:
##
	roslaunch demo_tactip real_robot.launch					- 	launch Franka reale
	rosrun stiffness_deeplearning robot_controller.py		-	controllo indentazione Franka
	rosrun stiffness_deeplearning Sensor.py					-	Log su csv forze, indentazione, immagini
	rosrun stiffness_deeplearning csv_publisher.py			-	Ri-pubblica dati salvati su csv

## Driver Sensori Force/Torque:
##
	roslaunch netft_rdt_driver ft_sensors.launch		- 	Solo sensore su fingertip
	roslaunch netft_rdt_driver ft_2_sensors.launch		-	Sensore fingertip e validazione

## Algoritmi ITS:
##
	rosrun its its_node					-	Algoritmo ITS
	rosrun its soft_its_node			-	Algoritmo Soft ITS (con modello stiffness TacTip)
	rosrun its soft_its_viz				-	Visualizzatore ITS
##
###	roslaunch its soft_its.launch		-	Algoritmo, Visualizzatore e load parametri ITS
#### per versione con modello rigidezza
##
		roslaunch its soft_its.launch algorithm:=soft
#### per versione senza modello rigidezza (*Default*)
		roslaunch its soft_its.launch algorithm:=standard
	
## Driver TacTip:
##
	roslaunch tactip_driver tactip_camera.launch		-	Camera TacTip

## Algoritmi TacTip Density:
##
	rosrun its tactip_markers_tracker.py		-	Algoritmo tracking markers TacTip
	rosrun its tactip_voronoi.py				-	Algoritmo superficie con aree voronoi
	rosrun its tactip_gaussian_kernel.py		-	Algoritmo superficie con kernel gaussiano
	rosrun its tactip_contact_detection.py		-	Algoritmo TacTip Density (vecchia versione con tracking unificata RIMOVIBILE)
	rosrun its its_tactip_viz.py				-	Visualizzatore TacTip Density

###	roslaunch its tactip.launch					-	Algoritmo, Visualizzatore e load parametri TacTip Density
####	per versione con densita' gaussiana (*Default*)
##
		roslaunch its tactip.launch algorithm:=gaussian

####	per versione volume superficie a celle di voronoi
##
		roslaunch its tactip.launch algorithm:=voronoi
