#ifndef SOFT_ITS_H
#define SOFT_ITS_H

#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Dense>
#include <Eigen/LU>

#include "its/Fingertip.hpp"
#include "its/IntrinsicTactileSensing.hpp"

using namespace fingertip;

namespace soft_its {


    /* 
    *   TO DO: Descrizione libreria
    */
// Contact Sensing problem minimal solution
struct ContactSensingProblemSolution{
		Eigen::Vector3d c;      // Point of Contact position (PoC),[m]
		double K;               // Torque scale factor
        double Dd; 	            // Surface Deformation, [mm]
};

// Contact Sensing problem extended solution
struct ExtendedContactSensingProblemSolution{
    Eigen::Vector3d PoC;    // Point of Contact (PoC) w.r.t Fingertip Frame {B}, [m]
    Eigen::Vector3d n;      // Normal in the point of contact (PoC) w.r.t Fingertip Frame {B}, [-]
    double fn;              // Normal force w.r.t Surface in the point of contact (PoC), [N]
    Eigen::Vector3d ft;     // Tangential force w.r.t Surface in the point of contact (PoC), [N]
    double t;               // Torque along normal in the point of contact (PoC), [Nm]
    double Dd;              // Surface Deformation, [mm]
};
class SoftIntrinsicTactileSensing: public its::IntrinsicTactileSensing{
    public:
        //std::string sensor_id;                  // Sensor id
        //Fingertip fingertip;                    // Fingertip description (i.e. surface model, pose)
        //Eigen::Vector3d f;                      // Force measured from ATI sensor [N]
        //Eigen::Vector3d m;                      // Torque measured from ATI sensor [N]
        ContactSensingProblemSolution X;        // Soft ITS Solution X = [c', K, Dd']


        // Constructor/Deconstructor
		SoftIntrinsicTactileSensing();
		~SoftIntrinsicTactileSensing();

        // SET FINGERTIP DECRIPTION
        //bool setFingertipSurface(std::string id, double a = 1.0, double b = 1.0, double c = 1.0);
        /* setFingertipSurface:
        *       The Fingertip surface ellipsoid model  
        *       (x^2)/(a^2) + (y^2)/(b^2) + (z^2)/(c^2) = 1
        */

		//bool setFingertipDisplacement(double dispX, double dispY, double dispZ);
        /* setFingertipDisplacement:
        *       The displacment w.r.t Force Sensor frame {S}
        */
        //bool setFingertipOrientation(double roll, double pitch, double yaw);
        /* setFingertipOrientation:
        *       The orientation w.r.t Force sensor frame {S} is achieved by roll-pitch-yaw encoding  
        *       Rot = Rz(yaw) * Ry(pitch) * Rz(roll)
        */


        bool setFingertipStiffness(double a = 0.0, double b = 0.0);
        /* setFingertipStiffness: 
        *       Two way to describe the surface elastic behaviour:
        *           1. Hooke's law:                 
        *                                   F = a*Dx --> homogeneous and isotropic material
        *                                   stiffnessCoefficients(0) = a
        *                                   stiffnessCoefficients(1) = 0
        *           2. Quadratic's law : 
        *                                   F = a*Dx + b*Dd^2
        *                                   stiffnessCoefficients(0) = a
        *                                   stiffnessCoefficients(1) = b
        *           3. Rigid surface:
        *                                   stiffnessCoefficients(0) = 0
        *                                   stiffnessCoefficients(1) = 0
        */
       


        // Solve contact Sensing Problem
        int solveContactSensingProblem(Eigen::Vector3d f, Eigen::Vector3d t, double forceThreshold = 0.0,
                                    its::ContactSensingProblemMethod method = its::ContactSensingProblemMethod::Levenberg_Marquardt,
                                    ContactSensingProblemSolution X0 = ContactSensingProblemSolution(), int count_max = 100, double stop_th = 0.005, 
                                    double epsilon = 0.1, bool verbose = false);
        /* solveContactSensingProblem: 
        *       This routine implements different solver for Soft Contact Sensing Problem:
        *           Iterative Method:
        *                   - Levenberg-Marquardt (=LM)
        *                   - Gauss-Newton (=GN)
        *           Iterative Method:
        *                   - Closed-Form (=CF)
        *                   - Wrench-Method (=WM)
        */
        
        int solveContactSensingProblemLM(ContactSensingProblemSolution X0 ,Eigen::Vector3d f, Eigen::Vector3d t, double forceThreshold = 0.0,
                                    int count_max = 100, double stop_th = 0.005, double epsilon = 0.1, bool verbose = false);
        int solveContactSensingProblemGN(ContactSensingProblemSolution X0 ,Eigen::Vector3d f, Eigen::Vector3d t, double forceThreshold = 0.0,
                                    int count_max = 100, double stop_th = 0.005, double epsilon = 0.005, bool verbose = false);
        int solveContactSensingProblemCF(Eigen::Vector3d f, Eigen::Vector3d t, double forceThreshold = 0.0);
        int solveContactSensingProblemWM(Eigen::Vector3d f, Eigen::Vector3d t, double forceThreshold = 0.0);
      

        // retrieve the extended solution
        ExtendedContactSensingProblemSolution getExtendedSolution();
};

}  // namespace soft_its
#endif // SOFT_ITS_H