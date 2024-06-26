#include "its/SoftIntrinsicTactileSensing.hpp"

using namespace soft_its;
using namespace fingertip;

SoftIntrinsicTactileSensing::SoftIntrinsicTactileSensing(){
	/* do nothing */
}
SoftIntrinsicTactileSensing::~SoftIntrinsicTactileSensing(){
    /* do nothing */
}


bool SoftIntrinsicTactileSensing::setFingertipSurface(std::string id, double a, double b, double c){
    this->fingertip.id = id;
    this->fingertip.model.principalAxisCoeff = {a, b, c};
    if (a == b == c){
        this->fingertip.model.surfaceType = SurfaceType::Sphere;
    }
    else{
        this->fingertip.model.surfaceType = SurfaceType::Ellipsoid;
    }
    return true;
}
bool SoftIntrinsicTactileSensing::setFingertipDisplacement(double dispX, double dispY, double dispZ){
    this->fingertip.displacement(0) = dispX;
	this->fingertip.displacement(1) = dispY;
	this->fingertip.displacement(2) = dispZ;
    return true;
}
bool SoftIntrinsicTactileSensing::setFingertipOrientation(double roll, double pitch, double yaw){
    Eigen::Matrix3d Roll, Pitch, Yaw;
    Roll <<     1,     0,          0,
                0,  cos(roll),   sin(roll),
                0, -sin(roll),  cos(roll);

    Pitch <<    cos(pitch), 0,  -sin(pitch),
                0,          1,      0,
                sin(pitch), 0,  cos(pitch);

    Yaw <<      cos(yaw),   sin(yaw),   0,
                -sin(yaw),  cos(yaw),   0,
                0,              0,      1;

    this->fingertip.orientation = Yaw*Pitch*Roll;
    return true;
}

bool SoftIntrinsicTactileSensing::setFingertipStiffness(double a, double b){
    if (a == 0 and b == 0){
        this->fingertip.model.stiffnessType = StiffnessType::Rigid;
    }
    else if (b==0){
        this->fingertip.model.stiffnessType = StiffnessType::Hooke;
    }
    else{
        this->fingertip.model.stiffnessType = StiffnessType::Quadratic;
    }
    
    this->fingertip.model.stiffnessCoefficients = {a, b};
    return true;

}


// Solve contact Sensing Problem
int SoftIntrinsicTactileSensing::solveContactSensingProblem(ContactSensingProblemSolution X0, Eigen::Vector3d f, Eigen::Vector3d m, double forceThreshold,
                                                            ContactSensingProblemMethod method, int count_max, double stop_th, double epsilon, bool verbose){
    
    switch (method)
    {
    case ContactSensingProblemMethod::Levenberg_Marquardt:
        return solveContactSensingProblemLM(X0,f,m,forceThreshold,count_max,stop_th,epsilon,verbose);

    case ContactSensingProblemMethod::Gauss_Newton:
        return solveContactSensingProblemGN(X0,f,m,forceThreshold,count_max,stop_th,epsilon,verbose);

    case ContactSensingProblemMethod::Closed_Form:
        return solveContactSensingProblemCF(f,m,forceThreshold);

    default:
        return solveContactSensingProblemLM(X0,f,m,forceThreshold,count_max,stop_th,epsilon,verbose);
    }
}

int SoftIntrinsicTactileSensing::solveContactSensingProblemLM(ContactSensingProblemSolution X0, Eigen::Vector3d f, Eigen::Vector3d m, double forceThreshold,
                                                            int count_max, double stop_th, double epsilon, bool verbose){
    // store measure
    this->f = f;                // [N]
    this->m = m;                // [Nm]
    
    // check threshold condition
    if(f.norm()<forceThreshold){
        return -1;
    }
    // Initial guess
    this->X = X0;                    // Contact sensing solution initialization
    double x, y, z, k, d;            // Store solution into more readable variables
    x = this->X.c(0);                // PoC x [mm]
    y = this->X.c(1);                // PoC y [mm]
    z = this->X.c(2);                // PoC z[mm]
    k = this->X.K;                   // Scale Factor
    d = this->X.Dd;                  // Deformation [mm]
    

    // Fingertip model
    const double a = this->fingertip.model.principalAxisCoeff[0];      // Ellipsoid Principal Axis Coefficients [mm]
    const double b = this->fingertip.model.principalAxisCoeff[1];
    const double c = this->fingertip.model.principalAxisCoeff[2];

    const Eigen::Vector3d d_sb =  this->fingertip.displacement;
    const Eigen::Matrix3d R_sb =  this->fingertip.orientation;

    const std::vector<double> E = this->fingertip.model.stiffnessCoefficients;   // Surface Stiffness Coefficients [N/mm]
    
    
    // F/T Sensor measures
    Eigen::Vector3d tmp = R_sb*f;       // Force Measure [N] w.r.t Fingertip Frame, {B}
    const double fx = tmp(0);      
    const double fy = tmp(1);
    const double fz = tmp(2);

    tmp = R_sb*(m + f.cross(d_sb/1000.0));     // Torque Measure [Nm] w.r.t Fingertip Frame, {B}
    const double mx = tmp(0)*1000.0;           // converting in [Nmm]
    const double my = tmp(1)*1000.0;
    const double mz = tmp(2)*1000.0;
    // LM Algorithm params
    double lambda = 100;                // init damping parameter 


    // Contact Problem Sensing definition
    Eigen::VectorXd g(5);                                       // contact problem function. g(x) = 0
    Eigen::MatrixXd J(5,5);                                     // Jacobian matrix. J = dg/dx
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(5, 5);        // eye matrix
    Eigen::VectorXd h(5);                                       // Update variable

    // Contact Problem Sensing iterative routine
    int count = 0;                                                                         // iteration step counter      
    Eigen::Vector3d n =  this->fingertip.model.getNormal(x,y,z,d);                         // normal direction in PoC
    // eval g(x0)
    g(0) = 2*k*x/((a-d)*(a-d)) - fy*z + fz*y - mx;                                         // torque direction condition 
    g(1) = 2*k*y/((b-d)*(b-d)) - fz*x + fx*z - my;                                         // ...
    g(2) = 2*k*z/((c-d)*(c-d)) - fx*y + fz*x - mz;                                         // ...
    g(3) = (x*x)/((a-d)*(a-d)) + (y*y)/((b-d)*(b-d)) + (z*z)/((c-d)*(c-d)) - 1;            // point on surface 
    if(this->fingertip.model.stiffnessType == StiffnessType::Hooke){                       // surface deformation model
        g(4) = -f.transpose()*n.normalized() - E[0]*d;                    
    }else if (this->fingertip.model.stiffnessType == StiffnessType::Quadratic){
        g(4) = -f.transpose()*n.normalized() - E[0]*d - E[1]*d*d;  
    }                         

    double Xi_square = g(0)*g(0) + g(1)*g(1) + g(2)*g(2) + g(3)*g(3) + g(4)*g(4);       // Minimizing function cost
    double Xi_square_old = Xi_square;                                                   // Minimizing function cost old step
    
    while (Xi_square >= stop_th && count <= count_max){

            // Eval J = dg/dx(x)
            // VERSIONE 1
            // dg1/dx
            J(0,0) = 2*k/((a-d)*(a-d));                                       
            J(0,1) = fz;                                       
            J(0,2) = -fy;                                       
            J(0,3) = 2*x/((a-d)*(a-d));         
            J(0,4) = 4*k*x/((a-d)*(a-d)*(a-d));    
            // dg2/dx
            J(1,0) = -fz;                                       
            J(1,1) = 2*k/((b-d)*(b-d));                                       
            J(1,2) = fx;                                       
            J(1,3) = 2*y/((b-d)*(b-d));         
            J(1,4) = 4*k*y/((b-d)*(b-d)*(b-d));   
            // dg3/dx
            J(2,0) = fy;                                       
            J(2,1) = -fx;                                       
            J(2,2) = 2*k/((c-d)*(c-d));                                       
            J(2,3) = 2*z/((c-d)*(c-d));         
            J(2,4) = 4*k*z/((c-d)*(c-d)*(c-d));    
            // dg4/dx
            J(3,0) = 2*x/((a-d)*(a-d));                                       
            J(3,1) = 2*y/((b-d)*(b-d));                                       
            J(3,2) = 2*z/((c-d)*(c-d));                                       
            J(3,3) = 0;         
            J(3,4) = 2*x*x/((a-d)*(a-d)*(a-d)) + 2*y*y/((b-d)*(b-d)*(b-d)) + 2*z*z/((c-d)*(c-d)*(c-d));    
            // dg5/dx
            /*
            J(4,0) = - 2*fx/(a*a) - E*d*8*x/(a*a*a*a);                                       
            J(4,1) = - 2*fy/(b*b) - E*d*8*y/(b*b*b*b);                                       
            J(4,2) = - 2*fz/(c*c) - E*d*8*z/(c*c*c*c);                                       
            J(4,3) = 0;         
            J(4,4) = - E*(4*x*x/(a*a*a*a) + 4*y*y/(b*b*b*b) + 4*z*z/(c*c*c*c));    
            */

            // VERSIONE 2 - normale uguale tra S e S'
            /*
            // dg1/dx
            J(0,0) = 2*k/(a*a);                                       
            J(0,1) = fz;                                       
            J(0,2) = -fy;                                       
            J(0,3) = 2*x/(a*a);         
            J(0,4) = 0;    
            // dg2/dx
            J(1,0) = -fz;                                       
            J(1,1) = 2*k/(b*b);                                       
            J(1,2) = fx;                                       
            J(1,3) = 2*y/(b*b);         
            J(1,4) = 0;   
            // dg3/dx
            J(2,0) = fy;                                       
            J(2,1) = -fx;                                       
            J(2,2) = 2*k/(c*c);                                       
            J(2,3) = 2*z/(c*c);         
            J(2,4) = 0;    
            // dg4/dx
            J(3,0) = 2*x/((a-d)*(a-d));                                       
            J(3,1) = 2*y/((b-d)*(b-d));                                       
            J(3,2) = 2*z/((c-d)*(c-d));                                       
            J(3,3) = 0;         
            J(3,4) = 2*x*x/((a-d)*(a-d)*(a-d)) + 2*y*y/((b-d)*(b-d)*(b-d)) + 2*z*z/((c-d)*(c-d)*(c-d));    
            */
            // dg5/dx
            J(4,0) = - 2*fx/(a*a);                                       
            J(4,1) = - 2*fy/(b*b);                                       
            J(4,2) = - 2*fz/(c*c);                                       
            J(4,3) = 0;        
            if(this->fingertip.model.stiffnessType == StiffnessType::Hooke){                        
                J(4,4) = - E[0];                    
            }else if (this->fingertip.model.stiffnessType == StiffnessType::Quadratic){
                J(4,4) = - E[0] - 2*E[1];  
            }     
            
            // update solution step
            h = -(J.transpose()*J + lambda*I).inverse()*J.transpose()*g; 
            x += h(0); y += h(1); z += h(2); k += h(3); d += h(4);
            
            // eval g(x_new)   
            n =  this->fingertip.model.getNormal(x,y,z,d);
            g(0) = 2*k*x/((a-d)*(a-d)) - fy*z + fz*y - mx;
            g(1) = 2*k*y/((b-d)*(b-d)) - fz*x + fx*z - my;
            g(2) = 2*k*z/((c-d)*(c-d)) - fx*y + fy*x - mz;
            g(3) = (x*x)/((a-d)*(a-d)) + (y*y)/((b-d)*(b-d)) + (z*z)/((c-d)*(c-d)) - 1;
            if(this->fingertip.model.stiffnessType == StiffnessType::Hooke){                        
                g(4) = -f.transpose()*n.normalized() - E[0]*d;                    
            }else if (this->fingertip.model.stiffnessType == StiffnessType::Quadratic){
                g(4) = -f.transpose()*n.normalized() - E[0]*d - E[1]*d*d;  
            }                         

            //print step
            if (verbose){
                const Eigen::IOFormat fmt(5, Eigen::DontAlignCols, "\t\t|", " ", "", "", "", "");
                std::cout << "Step:\t" << count << std::endl;
                std::cout << "g(0)\t" << "\t|" << "g(1)\t" << "\t|" << "g(2)\t" << "\t|" << "g(3)\t" << "\t|" << "g(4)" << std::endl;
                std::cout << g.transpose().format(fmt) << std::endl;    
                std::cout << "h(0)\t" << "\t|" << "h(1)\t" << "\t|" << "h(2)\t" << "\t|" << "h(3)\t" << "\t|" << "h(4)" << std::endl;
                std::cout << h.transpose().format(fmt) << std::endl;    
                std::cout << "x\t" << "\t|" << "y\t" << "\t|" << "z\t" << "\t|" << "k\t" << "\t|" << "Dd" << std::endl;
                std::cout << x << "\t\t|" << y << "\t\t|" << z << "\t\t|" << k << "\t\t|" << d << std::endl;
            }
            
            // update lambda step
            Xi_square_old = Xi_square;
            Xi_square = g(0)*g(0) + g(1)*g(1) + g(2)*g(2) + g(3)*g(3) + g(4)*g(4);   
            if ((Xi_square_old - Xi_square) > epsilon*h.transpose()*(lambda*h - J.transpose()*g))
                lambda /=10;
            else
                lambda *= 10;
            
            // update counter
            count++; 
    }

    // store result and return number of iteration
    const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, "\t\t|", " ", "", "", "", "");
    std::cout << "\t|g(0)\t" << "\t|" << "g(1)\t" << "\t|" << "g(2)\t" << "\t|" << "g(3)\t" << "\t|" << "g(4)" << std::endl;
    std::cout << "\t|" << g.transpose().format(fmt) << std::endl; 

    this->X.c(0) = x;          
    this->X.c(1) = y;
    this->X.c(2) = z;
    this->X.K = k;
    this->X.Dd = d;
    
    return count;
}
int SoftIntrinsicTactileSensing::solveContactSensingProblemGN(ContactSensingProblemSolution X0, Eigen::Vector3d f, Eigen::Vector3d m, double forceThreshold,
                                                            int count_max, double stop_th, double epsilon, bool verbose){
    // store measure
    this->f = f;                // [N]
    this->m = m;                // [Nm]
    
    // check threshold condition
    if(f.norm()<forceThreshold){
        return -1;
    }
    // Initial guess
    this->X = X0;                    // Contact sensing solution initialization
    double x, y, z, k, d;            // Store solution into more readable variables
    x = this->X.c(0);                // PoC x [mm]
    y = this->X.c(1);                // PoC y [mm]
    z = this->X.c(2);                // PoC z[mm]
    k = this->X.K;                   // Scale Factor
    d = this->X.Dd;                  // Deformation [mm]
    

    // Fingertip model
    const double a = this->fingertip.model.principalAxisCoeff[0];      // Ellipsoid Principal Axis Coefficients [mm]
    const double b = this->fingertip.model.principalAxisCoeff[1];
    const double c = this->fingertip.model.principalAxisCoeff[2];

    const Eigen::Vector3d d_sb =  this->fingertip.displacement;
    const Eigen::Matrix3d R_sb =  this->fingertip.orientation;

    const std::vector<double> E = this->fingertip.model.stiffnessCoefficients;   // Surface Stiffness Coefficients [N/mm]
    
    
    // F/T Sensor measures
    Eigen::Vector3d tmp = R_sb*f;       // Force Measure [N] w.r.t Fingertip Frame, {B}
    const double fx = tmp(0);      
    const double fy = tmp(1);
    const double fz = tmp(2);

    tmp = R_sb*(m + f.cross(d_sb/1000.0));     // Torque Measure [Nm] w.r.t Fingertip Frame, {B}
    const double mx = tmp(0)*1000.0;           // converting in [Nmm]
    const double my = tmp(1)*1000.0;
    const double mz = tmp(2)*1000.0;

    // Contact Problem Sensing definition
    Eigen::VectorXd g(5);                                       // contact problem function. g(x) = 0
    Eigen::MatrixXd J(5,5);                                     // Jacobian matrix. J = dg/dx
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(5, 5);        // eye matrix
    Eigen::VectorXd h(5);                                       // Update variable

    // GN param
    double eps = epsilon;
    // Contact Problem Sensing iterative routine
    int count = 0;                                                                         // iteration step counter      
    Eigen::Vector3d n =  this->fingertip.model.getNormal(x,y,z,d);                         // normal direction in PoC
    // eval g(x0)
    g(0) = 2*k*x/((a-d)*(a-d)) - fy*x + fz*y - mx;                                         // torque direction condition 
    g(1) = 2*k*y/((b-d)*(b-d)) - fz*x + fx*z - my;                                         // ...
    g(2) = 2*k*z/((c-d)*(c-d)) - fx*y + fz*x - mz;                                         // ...
    g(3) = (x*x)/((a-d)*(a-d)) + (y*y)/((b-d)*(b-d)) + (z*z)/((c-d)*(c-d)) - 1;            // point on surface 
    if(this->fingertip.model.stiffnessType == StiffnessType::Hooke){                       // surface deformation model
        g(4) = -f.transpose()*n.normalized() - E[0]*d;                    
    }else if (this->fingertip.model.stiffnessType == StiffnessType::Quadratic){
        g(4) = -f.transpose()*n.normalized() - E[0]*d - E[1]*d*d;  
    }                         

    double Xi_square = g(0)*g(0) + g(1)*g(1) + g(2)*g(2) + g(3)*g(3) + g(4)*g(4);       // Minimizing function cost
    double Xi_square_old = Xi_square;                                                   // Minimizing function cost old step
    
    while (Xi_square >= stop_th && count <= count_max){

            // Eval J = dg/dx(x)
            // VERSIONE 1
            // dg1/dx
            J(0,0) = 2*k/((a-d)*(a-d));                                       
            J(0,1) = fz;                                       
            J(0,2) = -fy;                                       
            J(0,3) = 2*x/((a-d)*(a-d));         
            J(0,4) = 4*k*x/((a-d)*(a-d)*(a-d));    
            // dg2/dx
            J(1,0) = -fz;                                       
            J(1,1) = 2*k/((b-d)*(b-d));                                       
            J(1,2) = fx;                                       
            J(1,3) = 2*y/((b-d)*(b-d));         
            J(1,4) = 4*k*y/((b-d)*(b-d)*(b-d));   
            // dg3/dx
            J(2,0) = fy;                                       
            J(2,1) = -fx;                                       
            J(2,2) = 2*k/((c-d)*(c-d));                                       
            J(2,3) = 2*z/((c-d)*(c-d));         
            J(2,4) = 4*k*z/((c-d)*(c-d)*(c-d));    
            // dg4/dx
            J(3,0) = 2*x/((a-d)*(a-d));                                       
            J(3,1) = 2*y/((b-d)*(b-d));                                       
            J(3,2) = 2*z/((c-d)*(c-d));                                       
            J(3,3) = 0;         
            J(3,4) = 2*x*x/((a-d)*(a-d)*(a-d)) + 2*y*y/((b-d)*(b-d)*(b-d)) + 2*z*z/((c-d)*(c-d)*(c-d));    
            // dg5/dx
            J(4,0) = - 2*fx/(a*a);                                       
            J(4,1) = - 2*fy/(b*b);                                       
            J(4,2) = - 2*fz/(c*c);                                       
            J(4,3) = 0;        
            if(this->fingertip.model.stiffnessType == StiffnessType::Hooke){                        
                J(4,4) = - E[0];                    
            }else if (this->fingertip.model.stiffnessType == StiffnessType::Quadratic){
                J(4,4) = - E[0] - 2*E[1];  
            }     
            
            // update solution step
            h = -eps*J.transpose()*g; 
            x += h(0); y += h(1); z += h(2); k += h(3); d += h(4);
            
            // eval g(x_new)   
            n =  this->fingertip.model.getNormal(x,y,z,d);
            g(0) = 2*k*x/((a-d)*(a-d)) - fy*z + fz*y - mx;
            g(1) = 2*k*y/((b-d)*(b-d)) - fz*x + fx*z - my;
            g(2) = 2*k*z/((c-d)*(c-d)) - fx*y + fy*x - mz;
            g(3) = (x*x)/((a-d)*(a-d)) + (y*y)/((b-d)*(b-d)) + (z*z)/((c-d)*(c-d)) - 1;
            if(this->fingertip.model.stiffnessType == StiffnessType::Hooke){                        
                g(4) = -f.transpose()*n.normalized() - E[0]*d;                    
            }else if (this->fingertip.model.stiffnessType == StiffnessType::Quadratic){
                g(4) = -f.transpose()*n.normalized() - E[0]*d - E[1]*d*d;  
            }                         

            //print step
            if (verbose){
                const Eigen::IOFormat fmt(5, Eigen::DontAlignCols, "\t\t|", " ", "", "", "", "");
                std::cout << "Step:\t" << count << std::endl;
                std::cout << "g(0)\t" << "\t|" << "g(1)\t" << "\t|" << "g(2)\t" << "\t|" << "g(3)\t" << "\t|" << "g(4)" << std::endl;
                std::cout << g.transpose().format(fmt) << std::endl;    
                std::cout << "h(0)\t" << "\t|" << "h(1)\t" << "\t|" << "h(2)\t" << "\t|" << "h(3)\t" << "\t|" << "h(4)" << std::endl;
                std::cout << h.transpose().format(fmt) << std::endl;    
                std::cout << "x\t" << "\t|" << "y\t" << "\t|" << "z\t" << "\t|" << "k\t" << "\t|" << "Dd" << std::endl;
                std::cout << x << "\t\t|" << y << "\t\t|" << z << "\t\t|" << k << "\t\t|" << d << std::endl;
            }        
            // update cost function   
            Xi_square_old = Xi_square;
            Xi_square = g(0)*g(0) + g(1)*g(1) + g(2)*g(2) + g(3)*g(3) + g(4)*g(4);    
            // update counter
            count++; 
    }

    // store result and return number of iteration
    const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, "\t\t|", " ", "", "", "", "");
    std::cout << "\t|g(0)\t" << "\t|" << "g(1)\t" << "\t|" << "g(2)\t" << "\t|" << "g(3)\t" << "\t|" << "g(4)" << std::endl;
    std::cout << "\t|" << g.transpose().format(fmt) << std::endl; 

    this->X.c(0) = x;          
    this->X.c(1) = y;
    this->X.c(2) = z;
    this->X.K = k;
    this->X.Dd = d;
    
    return count;
}

int SoftIntrinsicTactileSensing::solveContactSensingProblemCF(Eigen::Vector3d f, Eigen::Vector3d m, double forceThreshold){
    double alpha, beta, gamma, invR, R, D, K, sigma, detG;
	Eigen::Vector3d Ap, invAt, AAp, psa;
	Eigen::Matrix3d A;
	
    // Store measure 
    this->f = f;    // Force [N]
    this->m = m;    // Torque [Nm]
    
    // check threshold condition
    if(f.norm()<forceThreshold){
        return -1;
    }
    
    // Fingertip model
    const double a = this->fingertip.model.principalAxisCoeff[0];      // Ellipsoid Principal Axis Coefficients [mm]
    const double b = this->fingertip.model.principalAxisCoeff[1];
    const double c = this->fingertip.model.principalAxisCoeff[2];

    const Eigen::Vector3d d_sb =  this->fingertip.displacement;
    const Eigen::Matrix3d R_sb =  this->fingertip.orientation;

    const std::vector<double> E = this->fingertip.model.stiffnessCoefficients;   // Surface Stiffness Coefficients [N/mm]

    // F/T Sensor measures
    Eigen::Vector3d tmp = R_sb*f;       // Force Measure [N] w.r.t Fingertip Frame, {B}
    const double px = tmp(0);      
    const double py = tmp(1);
    const double pz = tmp(2);
    const Eigen::Vector3d p =  {px, py, pz};
    
    tmp = R_sb*(m + f.cross(d_sb/1000.0));     // Torque Measure [Nm] w.r.t Fingertip Frame, {B}
    const double tx = tmp(0)*1000.0;           // convering in [Nmm]
    const double ty = tmp(1)*1000.0;
    const double tz = tmp(2)*1000.0;
    const Eigen::Vector3d t =  {tx, ty, tz};

	switch(this->fingertip.model.stiffnessType){
		case StiffnessType::Hooke:
			psa(0) = a - (fabs(px)/E[0]);
			psa(1) = b - (fabs(py)/E[0]);
			psa(2) = c - (fabs(pz)/E[0]);
			break;
		case StiffnessType::Quadratic:
            psa(0) = a - (-E[0] - sqrt(pow(E[0],2) + 4*E[1]*fabs(px))/(2*E[1]));
            psa(1) = b - (-E[0] - sqrt(pow(E[0],2) + 4*E[1]*fabs(py))/(2*E[1]));
            psa(2) = c - (-E[0] - sqrt(pow(E[0],2) + 4*E[1]*fabs(pz))/(2*E[1]));
			break;
		default:
			psa(0) = a; psa(1) = b; psa(2) = c;
			break;
	}

	alpha = psa(0);
	beta = psa(1);
	gamma = psa(2);
	invR = 1;

	while((alpha < 1.0) || (beta < 1.0) || (gamma < 1.0)){
		invR *= 10.0;
		alpha = a*invR;
		beta = b*invR;
		gamma = c*invR;
	}

	R = 1.0/invR;
	A.row(0) = Eigen::Vector3d(1.0/alpha, 0.0, 0.0);
    A.row(1) = Eigen::Vector3d(0.0, 1.0/beta, 0.0);
    A.row(2) = Eigen::Vector3d(0.0, 0.0, 1.0/gamma);
    
    Ap = A * p;
    AAp = A * A * p;
    invAt = A.inverse() * t;
    D = A.determinant();
    sigma = (D * D * invAt.squaredNorm()) - (R * R * Ap.squaredNorm()); 
    K = (-(p.dot(t) / sqrt(p.dot(t) * p.dot(t))) / (sqrt(2.0) * R * D)) * sqrt(sigma + sqrt((sigma * sigma) + (4.0 * D * D * R * R * p.dot(t) * p.dot(t))));
    detG = K * ((K * K * D * D) + Ap.squaredNorm());


    Eigen::Vector3d PoC;
    if(K != 0){
    	PoC = (1.0 / detG) * ((K * K * D * D * A.inverse() * A.inverse() * t) + (K * AAp.cross(t)) + (p.dot(t) * p));
    }
	else{
		return solveContactSensingProblemCF(f, m, forceThreshold);//TO DO: Wrench Axis Method
    }
    // store results
	this->X.c = PoC;
    this->X.K = K;
    this->X.Dd = c-psa(2);

	return 1;
}


int SoftIntrinsicTactileSensing::solveContactSensingProblemWM(Eigen::Vector3d f, Eigen::Vector3d m, double forceThreshold){
    // To Do?
    return -1;
}

ExtendedContactSensingProblemSolution SoftIntrinsicTactileSensing::getExtendedSolution(){
    ExtendedContactSensingProblemSolution csps;
    
    Eigen::Vector3d n = this->fingertip.model.getNormal(this->X.c(0),this->X.c(1),this->X.c(2),this->X.Dd);

    csps.PoC = this->X.c;                           // Point of Contact (PoC) w.r.t Fingertip Frame {B}
    csps.n = n.normalized();                        // Normal in the point of contact (PoC) w.r.t Fingertip Frame {B}

    csps.fn = this->f.dot(csps.n);                  // Normal force w.r.t Surface in the point of contact (PoC) [N]


    csps.ft = this->f - csps.fn*csps.n;             // Tangential force w.r.t Surface in the point of contact (PoC) [N]
    csps.t = (this->X.K*n).norm();                  // Torque along normal in the point of contact (PoC) [Nmm]

    csps.Dd = this->X.Dd;

    return csps; 
}