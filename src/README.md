Comandi Implementati 
=====================
## 1) _Interazione Franka_ per esperimenti di indentazione:
```	
roslaunch demo_tactip real_robot.launch					# launch Franka reale

rosrun stiffness_deeplearning robot_controller.py		# controllo indentazione Franka

rosrun stiffness_deeplearning Sensor.py					# Log su csv forze, indentazione, immagini

rosrun its data_logger.py								# Log su csv indentazione, forza verticale, errore centroide, angolo indentazione, tempo di convergenza, solver


rosrun stiffness_deeplearning csv_publisher.py			# Ri-pubblica dati salvati su csv

rosrun stiffness_deeplearning csv2bag.py -o <bag_name>	# Costruzione bag a partire da misure su csv
```
## 1.a) Driver Sensori Force/Torque:
```
roslaunch netft_rdt_driver ft_sensors.launch		# Solo sensore su fingertip
roslaunch netft_rdt_driver ft_2_sensors.launch		# Sensore fingertip e validazione
```

## 1.b) Algoritmi ITS:
```
rosrun its its_node					# Algoritmo ITS
rosrun its soft_its_node			# Algoritmo Soft ITS (con caratteristica rigidezza TacTip)
rosrun its soft_its_viz				# Visualizzatore ITS
```

### Launcher (Algoritmo, Visualizzatore e load parametri ITS)
#### Versione senza caratteristica rigidezza (*Default*)
##
		roslaunch its its.launch algorithm:=standard
#### Versione con caratteristica rigidezza
##
		roslaunch its its.launch algorithm:=soft 


## 2.a) Driver TacTip:
##
	roslaunch tactip_driver tactip_camera.launch		# Camera TacTip


## 2.b) Algoritmi TacTip Density:
##
	rosrun its tactip_markers_tracker.py		# Algoritmo tracking markers TacTip
	rosrun its tactip_voronoi.py				# Algoritmo superficie con aree voronoi
	rosrun its tactip_gaussian_kernel.py		# Algoritmo densità con kernel gaussiano
	rosrun its tactip_contact_detection.py		# Algoritmo TacTip Density (vecchia versione con tracking unificata RIMOVIBILE)
	rosrun its its_tactip_viz.py				# Visualizzatore superficie/densità

###	Launcher (Algoritmo, Visualizzatore e load parametri TacTip Density)
####	Versione con densita' gaussiana (*Default*)
##
		roslaunch its tactip.launch algorithm:=gaussian 

####	Versione superficie celle di voronoi
##
		roslaunch its tactip.launch algorithm:=voronoi

####	Versione parametri DigiTac (*Default*)
##
		roslaunch its tactip.launch tactip:=DigiTac

####	Versione parametri TacTip
##
		roslaunch its tactip.launch tactip:=TacTip