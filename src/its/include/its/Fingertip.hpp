#ifndef FINGERTIP_H
#define FINGERTIP_H


#include <string>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/LU>

namespace fingertip {

    /* 
    *   TO DO: Descrizione libreria
    */
    
    // Surfaces model types
    enum class SurfaceType: unsigned short int {
        Plane = 1,          // S(x,y,z): z = R    
        Sphere = 2,         // S(x,y,z): x^2 + y^2 + z^2 = R          
        Cylinder = 3,       // S(x,y,z): (x^2)/(a^2) + (y^2)/(b^2) = 1       
        Ellipsoid = 4,      // S(x,y,z): (x^2)/(a^2) + (y^2)/(b^2) + (z^2)/(c^2) = 1
        NURBS = 5
    };
    // Surfaces Stiffness types
    enum class StiffnessType: unsigned short int {
        Rigid = 1,          // Rigid:      Surface NO DEFORMABLE
        Hooke = 2,          // Hooke:      F = K * Dx
        Quadratic = 3,      // Quadratic:  F = p2 * Dx ^ 2 + p1 * Dx + p0
        Power = 4           // Power:      F = a * Dx ^ b
    };
    class Surface{
    /* Object implementation of Fingertip Surface */
        public:
            SurfaceType surfaceType;                        // Surface type
            std::vector<double>  principalAxisCoeff;        // Ellipsoid Principal-Axis coefficients
            StiffnessType stiffnessType;                    // Surface stiffness type
            std::vector<double> stiffnessCoefficients;      // Surface stiffness coefficients 


            Eigen::Vector3d getNormal(double x, double y, double z, double d = 0){   
                /* getNormal:
                *    Eval Surface normal at point p. n(p) = \/S. 
                */               
                Eigen::Vector3d n; 
                double a,b,c;
                switch (this->surfaceType){
                    
                    case SurfaceType::Ellipsoid:
                        a = this->principalAxisCoeff[0];
                        b = this->principalAxisCoeff[1];
                        c = this->principalAxisCoeff[2];
                        n(0) = 2*x/((a-d)*(a-d));
                        n(1) = 2*y/((b-d)*(b-d));
                        n(2) = 2*z/((c-d)*(c-d));
                        break;

                    case SurfaceType::Sphere:
                        a = this->principalAxisCoeff[0];
                        n(0) = 2*x/((a-d)*(a-d));
                        n(1) = 2*y/((a-d)*(a-d));
                        n(2) = 2*z/((a-d)*(a-d));
                        break;

                    default:
                        // to do. Excepiton
                        break;
                }

                return n;
            }
    };



    class Fingertip{
    /* Object implementation of Fingertip model */
        public:
            std::string id;                             // Fingertip name

			Eigen::Matrix3d orientation;                // Fingertip Frame {B} orientation w.r.t Sensor frame {S}
			Eigen::Vector3d displacement;               // Fingertip Frame {B} displacement [mm] w.r.t Sensor frame {S}
            
            Surface model;                              // Fingertip surface model            
    };

}  // namespace fingertip
#endif // FINGERTIP_H