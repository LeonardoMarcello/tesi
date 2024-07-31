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
        // INHERITED ATTRS:
        //std::string sensor_id;                  // Sensor id
        //Fingertip fingertip;                    // Fingertip description (i.e. surface model, pose)
        //Eigen::Vector3d f;                      // Force measured from ATI sensor [N]
        //Eigen::Vector3d m;                      // Torque measured from ATI sensor [N]OptimLM params;                         // Parameters for solving optimized LM by levmar.h 
        //OptimLM params;                         // Parameters for solving optimized LM by levmar.h 

        ContactSensingProblemSolution X;        // Soft ITS Solution X = [c', K, Dd']


        // Constructor/Deconstructor
		SoftIntrinsicTactileSensing();
		~SoftIntrinsicTactileSensing();

        // SET FINGERTIP DECRIPTION
        
        // INHERITED ROUTINES:
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

        // retrieve the extended solution
        ExtendedContactSensingProblemSolution getExtendedSolution();
       


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
    private: 
        virtual int solveContactSensingProblemLM(ContactSensingProblemSolution X0 ,Eigen::Vector3d f, Eigen::Vector3d t, double forceThreshold = 0.0,
                                    int count_max = 100, double stop_th = 0.005, double epsilon = 0.1, bool verbose = false);
        virtual int solveContactSensingProblemGN(ContactSensingProblemSolution X0 ,Eigen::Vector3d f, Eigen::Vector3d t, double forceThreshold = 0.0,
                                    int count_max = 100, double stop_th = 0.005, double epsilon = 0.005, bool verbose = false);
        virtual int solveContactSensingProblemCF(Eigen::Vector3d f, Eigen::Vector3d t, double forceThreshold = 0.0);
        virtual int solveContactSensingProblemWM(Eigen::Vector3d f, Eigen::Vector3d t, double forceThreshold = 0.0);
      
      
        /* Routines for Optimized Solver by using levmar.h
         * For more details of the library refer to lecture: 
         *      levmar: Levenberg-Marquardt nonlinear least squares algorithms in {C}/{C}++}, Lourakis, 2004,
         *      https://users.ics.forth.gr/~lourakis/levmar/
         * 
         * Non-Linear equation to solve with Levenberg-Marquartd:
         *          g(x) = 0
         *          dgdx: Jacobian of partial derivative
         * Parameters:
         *      double *x:      pointer to solution, it starts with initial guess value and terminate with founded solution 
         *      double *g:      System equations
         *      double *jac:    Jiacobian of system equations
         *      int m:          Solution dimension
         *      int n:          System dimension
         *      void *data:     Pointer to additional input,e.g. F/T measurements 
         * 
         *      int itmax:                  Max number of iteration
         *      double opt[5]:              Minimization options, [\tau, \epsilon1, \epsilon2, \epsilon3, \delta]. Respectively the
         *                                  scale factor for initial \mu, stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2 and the
         *                                  step used in difference approximation to the Jacobian. If \delta<0, the Jacobian is approximated
         *                                  with central differences which are more accurate (but slower!) compared to the forward differences
         *                                  employed by default. Set to NULL for defaults to be used.
         *      double info[10]:            Information regarding the minimization. Set to NULL if don't care
         *                                      info[0]= ||e||_2 at initial p.
         *                                      info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, \mu/max[J^T J]_ii ], all computed at estimated p.
         *                                      info[5]= num of iterations,
         *                                      info[6]=reason for terminating: 1 - stopped by small gradient J^T e
         *                                                                      2 - stopped by small Dp (solution variation)
         *                                                                      3 - stopped by itmax
         *                                                                      4 - singular matrix. Restart from current p with increased \mu
         *                                                                      5 - no further error reduction is possible. Restart with increased mu
         *                                                                      6 - stopped by small ||e||_2
         *                                                                      7 - stopped by invalid (i.e. NaN or Inf) "func" values; a user error
         *                                      info[7]= num of function evaluations  
         *                                      info[8]= num of Jacobian evaluations
         *                                      info[9]= num of linear systems solved, i.e. num of attempts for reducing error
         *  
         */
    private:
        static void soft_contact_sensing_problem(double *x, double *g, int m, int n, void *data);
        static void jacobian_soft_contact_sensing_problem(double *x, double *jac, int m, int n, void *data);
    
    public:
        int solveContactSensingProblemOptim(ContactSensingProblemSolution X0 ,Eigen::Vector3d f, Eigen::Vector3d m, its::OptimLM params);
        void setLMParameters(double forceThreshold = 0.0, int count_max = 100, double stop_th = 0.005, double epsilon = 0.005, bool verbose = false) override;    

        // INHERITHED ROUTINE
        //int solveContactSensingProblemOptim(its::ContactSensingProblemSolution X0 ,Eigen::Vector3d f, Eigen::Vector3d m, OptimLM params);
        
};

}  // namespace soft_its
#endif // SOFT_ITS_H