#include "its/IntrinsicTactileSensing.hpp"

using namespace its;
using namespace fingertip;

IntrinsicTactileSensing::IntrinsicTactileSensing(){
	/* do nothing */
}
IntrinsicTactileSensing::~IntrinsicTactileSensing(){
    /* do nothing */
}


bool IntrinsicTactileSensing::setFingertipSurface(std::string id, double a, double b, double c){
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
bool IntrinsicTactileSensing::setFingertipDisplacement(double dispX, double dispY, double dispZ){
    this->fingertip.displacement(0) = dispX;
	this->fingertip.displacement(1) = dispY;
	this->fingertip.displacement(2) = dispZ;
    return true;
}
bool IntrinsicTactileSensing::setFingertipOrientation(double roll, double pitch, double yaw){
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

// Solve contact Sensing Problem
int IntrinsicTactileSensing::solveContactSensingProblem(Eigen::Vector3d f, Eigen::Vector3d m, double forceThreshold, ContactSensingProblemMethod method, 
                                                        ContactSensingProblemSolution X0, int count_max, double stop_th, double epsilon, bool verbose){
    
    switch (method)
    {
    case ContactSensingProblemMethod::Levenberg_Marquardt:
        //return solveContactSensingProblemLM(X0,f,m,forceThreshold,count_max,stop_th,epsilon,verbose);
        return solveContactSensingProblemOptim(X0,f,m,this->params);

    case ContactSensingProblemMethod::Gauss_Newton:
        return solveContactSensingProblemGN(X0,f,m,forceThreshold,count_max,stop_th,epsilon,verbose);

    case ContactSensingProblemMethod::Closed_Form:
        return solveContactSensingProblemCF(f,m,forceThreshold);

    default:
        return solveContactSensingProblemLM(X0,f,m,forceThreshold,count_max,stop_th,epsilon,verbose);
    }
}

int IntrinsicTactileSensing::solveContactSensingProblemLM(ContactSensingProblemSolution X0, Eigen::Vector3d f, Eigen::Vector3d m, double forceThreshold,
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
    double x, y, z, k;            // Store solution into more readable variables
    x = this->X.c(0);                // PoC x [mm]
    y = this->X.c(1);                // PoC y [mm]
    z = this->X.c(2);                // PoC z [mm]
    k = this->X.K;                   // Scale Factor
    

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
    
    // store measure
    this->f = {fx,fy,fz};                // [N]
    this->m = {mx,my,mz};                // [Nm]

    // LM Algorithm params
    double lambda = 100;                // init damping parameter 


    // Contact Problem Sensing definition
    Eigen::VectorXd g(4);                                       // contact problem function. g(x) = 0
    Eigen::MatrixXd J(4,4);                                     // Jacobian matrix. J = dg/dx
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(4, 4);        // eye matrix
    Eigen::VectorXd h(4);                                       // Update variable

    // Contact Problem Sensing iterative routine
    int count = 0;                                                                         // iteration step counter      
    Eigen::Vector3d n =  this->fingertip.model.getNormal(x,y,z);                         // normal direction in PoC
    // eval g(x0)
    g(0) = 2*k*x/(a*a) - fy*z + fz*y - mx;                                         // torque direction condition 
    g(1) = 2*k*y/(b*b) - fz*x + fx*z - my;                                         // ...
    g(2) = 2*k*z/(c*c) - fx*y + fy*x - mz;                                         // ...
    g(3) = (x*x)/(a*a) + (y*y)/(b*b) + (z*z)/(c*c) - 1;                            // point on surface 
    

    double Xi_square = g(0)*g(0) + g(1)*g(1) + g(2)*g(2) + g(3)*g(3);       // Minimizing function cost
    double Xi_square_old = Xi_square;                                       // Minimizing function cost old step
    
    while (Xi_square >= stop_th && count <= count_max){

            // Eval J = dg/dx(x)
            // VERSIONE 1
            // dg1/dx
            J(0,0) = 2*k/(a*a);                                       
            J(0,1) = fz;                                       
            J(0,2) = -fy;                                       
            J(0,3) = 2*x/(a*a);            
            // dg2/dx
            J(1,0) = -fz;                                       
            J(1,1) = 2*k/(b*b);                                       
            J(1,2) = fx;                                       
            J(1,3) = 2*y/(b*b);           
            // dg3/dx
            J(2,0) = fy;                                       
            J(2,1) = -fx;                                       
            J(2,2) = 2*k/(c*c);                                       
            J(2,3) = 2*z/(c*c);             
            // dg4/dx
            J(3,0) = 2*x/(a*a);                                       
            J(3,1) = 2*y/(b*b);                                       
            J(3,2) = 2*z/(c*c);                                       
            J(3,3) = 0;        
            
            // update solution step
            h = -(J.transpose()*J + lambda*I).inverse()*J.transpose()*g; 
            x += h(0); y += h(1); z += h(2); k += h(3);
            
            // eval g(x_new)   
            n =  this->fingertip.model.getNormal(x,y,z);
            g(0) = 2*k*x/(a*a) - fy*z + fz*y - mx;
            g(1) = 2*k*y/(b*b) - fz*x + fx*z - my;
            g(2) = 2*k*z/(c*c) - fx*y + fy*x - mz;
            g(3) = (x*x)/(a*a) + (y*y)/(b*b) + (z*z)/(c*c) - 1;                    

            //print step
            if (verbose){
                const Eigen::IOFormat fmt(5, Eigen::DontAlignCols, "\t\t|", " ", "", "", "", "");
                std::cout << "Step:\t" << count << std::endl;
                std::cout << "g(0)\t" << "\t|" << "g(1)\t" << "\t|" << "g(2)\t" << "\t|" << "g(3)\t"  << std::endl;
                std::cout << g.transpose().format(fmt) << std::endl;    
                std::cout << "h(0)\t" << "\t|" << "h(1)\t" << "\t|" << "h(2)\t" << "\t|" << "h(3)\t"  << std::endl;
                std::cout << h.transpose().format(fmt) << std::endl;    
                std::cout << "x\t" << "\t|" << "y\t" << "\t|" << "z\t" << "\t|" << "k\t"  << std::endl;
                std::cout << x << "\t\t|" << y << "\t\t|" << z << "\t\t|" << k << std::endl;
            }
            
            // update lambda step
            Xi_square_old = Xi_square;
            Xi_square = g(0)*g(0) + g(1)*g(1) + g(2)*g(2) + g(3)*g(3);   
            if ((Xi_square_old - Xi_square) > epsilon*h.transpose()*(lambda*h - J.transpose()*g))
                lambda /=10;
            else
                lambda *= 10;
            
            // update counter
            count++; 
    }

    // store result and return number of iteration
    const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, "\t\t|", " ", "", "", "", "");
    std::cout << "\t|g(0)\t" << "\t|" << "g(1)\t" << "\t|" << "g(2)\t" << "\t|" << "g(3)\t"  << std::endl;
    std::cout << "\t|" << g.transpose().format(fmt) << std::endl; 

    this->X.c(0) = x;          
    this->X.c(1) = y;
    this->X.c(2) = z;
    this->X.K = k;
    
    return count;
}
int IntrinsicTactileSensing::solveContactSensingProblemGN(ContactSensingProblemSolution X0, Eigen::Vector3d f, Eigen::Vector3d m, double forceThreshold,
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
    double x, y, z, k;            // Store solution into more readable variables
    x = this->X.c(0);                // PoC x [mm]
    y = this->X.c(1);                // PoC y [mm]
    z = this->X.c(2);                // PoC z[mm]
    k = this->X.K;                   // Scale Factor
    

    // Fingertip model
    const double a = this->fingertip.model.principalAxisCoeff[0];      // Ellipsoid Principal Axis Coefficients [mm]
    const double b = this->fingertip.model.principalAxisCoeff[1];
    const double c = this->fingertip.model.principalAxisCoeff[2];

    const Eigen::Vector3d d_sb =  this->fingertip.displacement;
    const Eigen::Matrix3d R_sb =  this->fingertip.orientation;    
    
    // F/T Sensor measures
    Eigen::Vector3d tmp = R_sb*f;       // Force Measure [N] w.r.t Fingertip Frame, {B}
    const double fx = tmp(0);      
    const double fy = tmp(1);
    const double fz = tmp(2);

    tmp = R_sb*(m + f.cross(d_sb/1000.0));     // Torque Measure [Nm] w.r.t Fingertip Frame, {B}
    const double mx = tmp(0)*1000.0;           // converting in [Nmm]
    const double my = tmp(1)*1000.0;
    const double mz = tmp(2)*1000.0;   
    
    // store measure
    this->f = {fx,fy,fz};                // [N]
    this->m = {mx,my,mz};                // [Nm]

    // Contact Problem Sensing definition
    Eigen::VectorXd g(4);                                       // contact problem function. g(x) = 0
    Eigen::MatrixXd J(4,4);                                     // Jacobian matrix. J = dg/dx
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(4, 4);        // eye matrix
    Eigen::VectorXd h(4);                                       // Update variable

    // GN param
    double eps = epsilon;
    // Contact Problem Sensing iterative routine
    int count = 0;                                                                         // iteration step counter      
    Eigen::Vector3d n =  this->fingertip.model.getNormal(x,y,z);                         // normal direction in PoC
    // eval g(x0)
    g(0) = 2*k*x/(a*a) - fy*x + fz*y - mx;                                         // torque direction condition 
    g(1) = 2*k*y/(b*b) - fz*x + fx*z - my;                                         // ...
    g(2) = 2*k*z/(c*c) - fx*y + fy*x - mz;                                         // ...
    g(3) = (x*x)/(a*a) + (y*y)/(b*b) + (z*z)/(c*c) - 1;                            // point on surface 

    double Xi_square = g(0)*g(0) + g(1)*g(1) + g(2)*g(2) + g(3)*g(3);              // Minimizing function cost
    double Xi_square_old = Xi_square;                                              // Minimizing function cost old step
    
    while (Xi_square >= stop_th && count <= count_max){

            // Eval J = dg/dx(x)
            // VERSIONE 1
            // dg1/dx
            J(0,0) = 2*k/(a*a);                                       
            J(0,1) = fz;                                       
            J(0,2) = -fy;                                       
            J(0,3) = 2*x/(a*a);         
            // dg2/dx
            J(1,0) = -fz;                                       
            J(1,1) = 2*k/(b*b);                                       
            J(1,2) = fx;                                       
            J(1,3) = 2*y/(b*b);         
            // dg3/dx
            J(2,0) = fy;                                       
            J(2,1) = -fx;                                       
            J(2,2) = 2*k/(c*c);                                       
            J(2,3) = 2*z/(c*c);          
            // dg4/dx
            J(3,0) = 2*x/(a*a);                                       
            J(3,1) = 2*y/(b*b);                                       
            J(3,2) = 2*z/(c*c);                                       
            J(3,3) = 0;         
            
            // update solution step
            h = -eps*J.transpose()*g; 
            x += h(0); y += h(1); z += h(2); k += h(3);
            
            // eval g(x_new)   
            n =  this->fingertip.model.getNormal(x,y,z);
            g(0) = 2*k*x/(a*a) - fy*z + fz*y - mx;
            g(1) = 2*k*y/(b*b) - fz*x + fx*z - my;
            g(2) = 2*k*z/(c*c) - fx*y + fy*x - mz;
            g(3) = (x*x)/(a*a) + (y*y)/(b*b) + (z*z)/(c*c) - 1;                   

            //print step
            if (verbose){
                const Eigen::IOFormat fmt(5, Eigen::DontAlignCols, "\t\t|", " ", "", "", "", "");
                std::cout << "Step:\t" << count << std::endl;
                std::cout << "g(0)\t" << "\t|" << "g(1)\t" << "\t|" << "g(2)\t" << "\t|" << "g(3)\t" << std::endl;
                std::cout << g.transpose().format(fmt) << std::endl;    
                std::cout << "h(0)\t" << "\t|" << "h(1)\t" << "\t|" << "h(2)\t" << "\t|" << "h(3)\t"  << std::endl;
                std::cout << h.transpose().format(fmt) << std::endl;    
                std::cout << "x\t" << "\t|" << "y\t" << "\t|" << "z\t" << "\t|" << "k\t"  << std::endl;
                std::cout << x << "\t\t|" << y << "\t\t|" << z << "\t\t|" << k << std::endl;
            }        
            // update cost function   
            Xi_square_old = Xi_square;
            Xi_square = g(0)*g(0) + g(1)*g(1) + g(2)*g(2) + g(3)*g(3);    
            // update counter
            count++; 
    }

    // store result and return number of iteration
    const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, "\t\t|", " ", "", "", "", "");
    std::cout << "\t|g(0)\t" << "\t|" << "g(1)\t" << "\t|" << "g(2)\t" << "\t|" << "g(3)\t"  << std::endl;
    std::cout << "\t|" << g.transpose().format(fmt) << std::endl; 

    this->X.c(0) = x;          
    this->X.c(1) = y;
    this->X.c(2) = z;
    this->X.K = k;
    
    return count;
}

int IntrinsicTactileSensing::solveContactSensingProblemCF(Eigen::Vector3d f, Eigen::Vector3d m, double forceThreshold){
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

    // Store measure 
    this->f = p;    // Force [N]
    this->m = t;    // Torque [Nm]

	alpha = a;
	beta = b;
	gamma = c;
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

	return 1;
}


int IntrinsicTactileSensing::solveContactSensingProblemWM(Eigen::Vector3d f, Eigen::Vector3d m, double forceThreshold){
    // To Do?
    return -1;
}

ExtendedContactSensingProblemSolution IntrinsicTactileSensing::getExtendedSolution(){
    ExtendedContactSensingProblemSolution csps;
    
    Eigen::Vector3d n = this->fingertip.model.getNormal(this->X.c(0),this->X.c(1),this->X.c(2));

    csps.PoC = this->X.c;                           // Point of Contact (PoC) w.r.t Fingertip Frame {B}
    csps.n = n.normalized();                        // Normal in the point of contact (PoC) w.r.t Fingertip Frame {B}

    csps.fn = this->f.dot(csps.n);                  // Normal force w.r.t Surface in the point of contact (PoC) [N]


    csps.ft = this->f - csps.fn*csps.n;             // Tangential force w.r.t Surface in the point of contact (PoC) [N]
    csps.t = (this->X.K*n).norm();                  // Torque along normal in the point of contact (PoC) [Nmm]

    return csps; 
}


/* Routines for Optimized Solver by using levmar.h */

void IntrinsicTactileSensing::contact_sensing_problem(double *x, double *g, int m, int n, void *data){
    double fx,fy,fz,mx,my,mz,a,b,c;    
    double input[9];
	memcpy(&input, data, 9*sizeof(double));
    fx = input[0]; fy = input[1]; fz = input[2]; 
    mx = input[3]; my = input[4]; mz = input[5];
    a = input[6]; b = input[7]; c = input[8]; 
    
    // Solution: x = [cx, cy, cz, k, Dd]
    // Soft Contact Sensing Ptoblem: x t.c. g(x) = 0
    g[0] = 2*x[3]*x[0]/(a*a) - fy*x[2] + fz*x[1] - mx;                                         // torque direction condition 
    g[1] = 2*x[3]*x[1]/(b*b) - fz*x[0] + fx*x[2] - my;                                         // ...
    g[2] = 2*x[3]*x[2]/(c*c) - fx*x[1] + fy*x[0] - mz;                                         // ...
    g[3] = (x[0]*x[0])/(a*a) + (x[1]*x[1])/(b*b) + (x[2]*x[2])/(c*c) - 1;                   // point on surface 
}
void IntrinsicTactileSensing::jacobian_contact_sensing_problem(double *x, double *jac, int m, int n, void *data){
    double fx,fy,fz,mx,my,mz,a,b,c;    
    double input[9];
	memcpy(&input, data, 9*sizeof(double));
    fx = input[0]; fy = input[1]; fz = input[2]; 
    mx = input[3]; my = input[4]; mz = input[5];
    a = input[6]; b = input[7]; c = input[8]; 
    
    // Eval J = dg/dx(x)
    // dg1/dx
    jac[0] = 2*x[3]/(a*a);                                       
    jac[1] = fz;                                       
    jac[2] = -fy;                                       
    jac[3] = 2*x[0]/(a*a);         
    // dg2/dx
    jac[4] = -fz;                                       
    jac[5] = 2*x[3]/(b*b);                                       
    jac[6] = fx;                                       
    jac[7] = 2*x[1]/(b*b);         
    // dg3/dx
    jac[8] = fy;                                       
    jac[9] = -fx;                                       
    jac[10] = 2*x[3]/(c*c);                                       
    jac[11] = 2*x[2]/(c*c);          
    // dg4/dx
    jac[12] = 2*x[0]/(a*a);                                       
    jac[13] = 2*x[1]/(b*b);                                       
    jac[14] = 2*x[2]/(c*c);                                       
    jac[15] = 0; 
}

int IntrinsicTactileSensing::solveContactSensingProblemOptim(ContactSensingProblemSolution X0 ,Eigen::Vector3d f, Eigen::Vector3d m, OptimLM params){
    // store measure
    this->f = f;                // [N]
    this->m = m;                // [Nm]
    
    // check threshold condition
    if(f.norm()< params.force_threshold){
        return -1;
    }
    // Fingertip model
    const double a = this->fingertip.model.principalAxisCoeff[0];      // Ellipsoid Principal Axis Coefficients [mm]
    const double b = this->fingertip.model.principalAxisCoeff[1];
    const double c = this->fingertip.model.principalAxisCoeff[2];

    const Eigen::Vector3d d_sb =  this->fingertip.displacement;
    const Eigen::Matrix3d R_sb =  this->fingertip.orientation;

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

    // Store measure 
    this->f = p;    // Force [N]
    this->m = t;    // Torque [Nm]

    // Set variable for LM solver 
    double x[4] = {X0.c(0),X0.c(1),X0.c(2),X0.K};                   // initial guess x0
    double data[9] = {p(0),p(1),p(2), t(0),t(1),t(2), a, b, c};     // measures and params
  
	int step = dlevmar_der(this->contact_sensing_problem, this->jacobian_contact_sensing_problem, x, NULL, params.m, params.n, params.itmax, params.opt, params.info, NULL, NULL, data);
    //int step = dlevmar_dif(this->contact_sensing_problem, x, NULL, params.m, params.n, params.itmax, params.opt, params.info, NULL, NULL, data);


    this->X.c(0) = x[0]; this->X.c(1) = x[1]; this->X.c(2) = x[2]; this->X.K = x[3];

    if (params.verbose){
        std::cout << "Reason of termination:\t" << params.info[6] << std::endl;
        std::cout << "Num of Step:\t" << params.info[5] << std::endl;
        std::cout << "Error ||e||_2 at initial guess ("<< X0.c(0) << ", " << X0.c(1) << ", " << X0.c(2) << ", " << X0.K << "):\t" << params.info[0] << std::endl;
        std::cout << "Error ||e||_2 at solution:\t" << params.info[1] << std::endl;
        std::cout << "Last value of ||J^T g ||_inf:\t" << params.info[2] << std::endl;
        std::cout << "Last step ||delta_x||_2:\t" << params.info[3] << std::endl;
        std::cout << "lambda at solution:\t" << params.info[4] << std::endl;
    }    
    return step; 

}


void IntrinsicTactileSensing::setLMParameters(double forceThreshold, int count_max, double stop_th, double epsilon, bool verbose){

    this->params.force_threshold = forceThreshold;
    this->params.itmax = count_max;
    this->params.m = 4; this->params.n = 4;
    this->params.verbose = verbose;

    this->params.opt[0] =  1e2;                                // mu init (lambda = mu*max{J^T J} )
    this->params.opt[1] = 1e-15;                                // stop threshold on ||J^T e||_inf 
    this->params.opt[2] = 1e-15;                                // stop threshold on ||Dp||_2
    this->params.opt[3] = stop_th;                              // stop threshold on ||e||_2
}