#include "ros/ros.h"
#include <tf/transform_broadcaster.h>
#include "geometry_msgs/WrenchStamped.h"
#include "its_msgs/SoftContactSensingProblemSolution.h"
#include "its/IntrinsicTactileSensing.hpp"


using namespace its;


// Soft Intrinsic Tactile Sensing variables
IntrinsicTactileSensing ITS;                        // ITS Object
ContactSensingProblemSolution X0;                   // Initial Guess
ExtendedContactSensingProblemSolution solution;     // Extended CSP Solution
std::vector<double> psa_at_rest(3);                 // principal axis with no contact

// Measures variables
bool new_measure = false;               // Force measured from ATI sensor [N]
Eigen::Vector3d f;                      // Force measured from ATI sensor [N]
Eigen::Vector3d m;                      // Torque measured from ATI sensor [N]

// Force/Torque Sensor callback
void ftCallback(const geometry_msgs::WrenchStamped::ConstPtr& msg){
  f(0) = msg->wrench.force.x;
  f(1) = msg->wrench.force.y;
  f(2) = msg->wrench.force.z;
  m(0) = msg->wrench.torque.x;
  m(1) = msg->wrench.torque.y;
  m(2) = msg->wrench.torque.z;
  new_measure = true;
}

// Initial Guess callback
void igCallback(const its_msgs::SoftContactSensingProblemSolution::ConstPtr& msg){
  // Point Of Contact
  X0.c(0) = msg->PoC.x; X0.c(1) = msg->PoC.y; X0.c(2) = msg->PoC.z;
  // Set surface with estimated Normal Deformation
  std::string finger_id = ITS.fingertip.id;
  double Dd = msg->D;
  ITS.setFingertipSurface(finger_id, psa_at_rest[0]-Dd, psa_at_rest[1]-Dd, psa_at_rest[2]-Dd); 
  // Torque Scale Factor
  double norm = ITS.fingertip.model.getNormal(X0.c(0), X0.c(1), X0.c(2)).norm();
  X0.K = msg->T/norm;

  ROS_INFO("New Initial Guess setted");
}



// Main
int main(int argc, char **argv){
  ros::init(argc, argv, "its_node");
  ros::NodeHandle nh;

  ROS_INFO("Hi from its_node");

  // node rate
  double rate;

  // sensor parameters
  std::string sensor_id, finger_id;
  double dispX,dispY,dispZ;
  double roll,pitch,yaw;
  std::vector<double> psa(3);
  std::vector<double> stiff_coeff(3);

  // SoftIntrinsicTactileSensing parameters
  std::string solver_name;
  int count_max;
  double force_th, eps, stop_th;
  bool verbose;
  ContactSensingProblemMethod solver;

  // Transform object and transform broadcaster
  static tf::TransformBroadcaster br;
  tf::Transform transform;
  /*
  transform.setOrigin( tf::Vector3(msg->x, msg->y, 0.0) );
  tf::Quaternion q;
  q.setRPY(0, 0, msg->theta);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "turtle_name"));
  */


  // Declaring parameters
  nh.param<std::string>("sensor/id",sensor_id, "mySensor");
  ITS.sensor_id = sensor_id;

  nh.param<std::string>("fingertip/id",finger_id, "myFinger");
  nh.param<double>("fingertip/displacement/x",dispX,.0);
  nh.param<double>("fingertip/displacement/y",dispY,.0);
  nh.param<double>("fingertip/displacement/z",dispZ,.0);
  nh.param<double>("fingertip/orientation/roll",roll,.0);
  nh.param<double>("fingertip/orientation/pitch",pitch,.0);
  nh.param<double>("fingertip/orientation/yaw",yaw,.0);
  nh.param<double>("fingertip/principalSemiAxis/a",psa[0], 1);
  nh.param<double>("fingertip/principalSemiAxis/b",psa[1], 1);
  nh.param<double>("fingertip/principalSemiAxis/c",psa[2], 1);  
  psa_at_rest[0]=psa[0]; psa_at_rest[1]=psa[1]; psa_at_rest[2]=psa[2];
  ITS.setFingertipSurface(finger_id,psa[0],psa[1],psa[2]);
  ITS.setFingertipDisplacement(dispX,dispY,dispZ);
  ITS.setFingertipOrientation(roll,pitch,yaw);

  nh.param<bool>("soft_its/algorithm/verbose", verbose, false);
  nh.param<double>("soft_its/algorithm/force_threshold",force_th, 0.0);
  nh.param<std::string>("soft_its/algorithm/method/name",solver_name, "Levenberg-Marquardt");
  nh.param<int>("soft_its/algorithm/method/params/count_max",count_max, 100);
  nh.param<double>("soft_its/algorithm/method/params/stop_threshold",stop_th, 0.005);
  nh.param<double>("soft_its/algorithm/method/params/epsilon",eps, 0.01);

  nh.param<double>("soft_its/rate",rate, 0.5);
  
  const std::string green("\033[1;32m");
  const std::string reset("\033[0m");
  if (solver_name=="Levenberg-Marquardt"){
    solver = ContactSensingProblemMethod::Levenberg_Marquardt;
  }else if(solver_name=="Gauss-Newton"){
    solver = ContactSensingProblemMethod::Gauss_Newton;
  }else if(solver_name=="Closed-Form"){
    solver = ContactSensingProblemMethod::Closed_Form;
  }else{
    ROS_WARN("Desired solver does not exist. Levenberg_Marquardt method will be used");
    solver_name = "Lavenberg-Marquardt";
    solver = ContactSensingProblemMethod::Levenberg_Marquardt;
  }
	std::cout<<green<<"Solver: "<<solver_name<<reset<<std::endl;


  // Set initial guess
  //X0.c = {0,0,psa[2]};
  X0.c = {0,0,psa[2]/2};
  X0.K = 0;
  
  // Declaring Subscriber
  ros::Subscriber ft_sub = nh.subscribe(ITS.sensor_id+"/netft_data", 100, ftCallback);
  ros::Subscriber ig_sub = nh.subscribe("soft_csp/initial_guess", 100, igCallback);

  // Declaring Publisher
  ros::Publisher solution_pub = nh.advertise<its_msgs::SoftContactSensingProblemSolution>("soft_csp/solution", 100);

  // Set node rate
  ros::Rate ros_rate(rate);



  while (ros::ok()) {
    
    if (new_measure){
        // solve ContactSensingProblem
        ros::Time start_time(ros::Time::now());
        int step = ITS.solveContactSensingProblem(X0, f, m, force_th, solver, count_max, stop_th, eps, verbose);
        ros::Time stop_time(ros::Time::now());
        new_measure = false;

        if(step > 0){
          solution = ITS.getExtendedSolution();

          // report INFO 
          ROS_INFO("%s",std::string(40, '-').c_str());
          ROS_INFO("ITS input from %s: F = (%.3f,%.3f,%.3f); M = (%.3f,%.3f,%.3f);", ITS.sensor_id.c_str(),ITS.f(0),ITS.f(1),ITS.f(2),ITS.m(0),ITS.m(1),ITS.m(2));
          ROS_INFO("ITS solution X = [x,y,z,k] = [%.2f, %.2f, %.2f, %.2f]", ITS.X.c(0),ITS.X.c(1),ITS.X.c(2),ITS.X.K);
          ROS_INFO("ITS Extended solution [x,y,z,fn,t] = [%.2f, %.2f, %.2f, %.2f,%.2f]", 
                      solution.PoC(0), solution.PoC(1), solution.PoC(2),
                      solution.fn, solution.t);
          if(step < count_max){
            ROS_INFO("ITS algorithm runs for %i step. Elapsed time = %f ms", step, (stop_time.nsec - start_time.nsec)/1e6);
          }
          else{
            ROS_WARN("ITS algorithm needs more than %i step. Elapsed time = %f ms", count_max, (stop_time.nsec - start_time.nsec)/1e6);
          }
          ROS_INFO("%s",std::string(40, '-').c_str());

          // publish result
          Eigen::Vector3d n = ITS.fingertip.model.getNormal(solution.PoC(0),solution.PoC(1),solution.PoC(2)).normalized();
          its_msgs::SoftContactSensingProblemSolution sol_msg;
          sol_msg.header.frame_id = ITS.fingertip.id;
          sol_msg.header.stamp = ros::Time::now();
          sol_msg.PoC.x = solution.PoC(0); sol_msg.PoC.y = solution.PoC(1); sol_msg.PoC.z = solution.PoC(2);
          sol_msg.n.x = n(0); sol_msg.n.y = n(1); sol_msg.n.z = n(2);
          sol_msg.Fn = solution.fn; 
          sol_msg.Ft.x = solution.ft(0); sol_msg.Ft.y = solution.ft(1); sol_msg.Ft.z = solution.ft(2);
          sol_msg.T = solution.t;
          sol_msg.D = psa_at_rest[0]-ITS.fingertip.model.principalAxisCoeff[0];
          sol_msg.convergence_time = (stop_time.nsec - start_time.nsec)/1e6;

          solution_pub.publish(sol_msg);

          // broadcast tf
          transform.setOrigin( tf::Vector3(solution.PoC(0)/1000.0, solution.PoC(1)/1000.0, solution.PoC(2)/1000.0) );
          tf::Quaternion q;
          q.setRPY(0, 0, 0);
          transform.setRotation(q);
          br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), ITS.fingertip.id, "PoC"));
        }
        else{
          
          // publish empty msg
          its_msgs::SoftContactSensingProblemSolution sol_msg;
          sol_msg.header.frame_id = ITS.fingertip.id;
          sol_msg.header.stamp = ros::Time::now();
          sol_msg.PoC.x = 0; sol_msg.PoC.y = 0; sol_msg.PoC.z = 0;
          sol_msg.n.x = 0; sol_msg.n.y = 0; sol_msg.n.z = 0;
          sol_msg.Fn = 0; 
          sol_msg.Ft.x = 0; sol_msg.Ft.y = 0; sol_msg.Ft.z = 0;
          sol_msg.T = 0;
          sol_msg.D = 0;

          solution_pub.publish(sol_msg);
        }
    }
    else{
        /*
        ROS_INFO("F/T Measures not available");
        // publish empty msg
        its_msgs::SoftContactSensingProblemSolution sol_msg;
        sol_msg.header.frame_id = SITS.fingertip.id;
        sol_msg.header.stamp = ros::Time::now();
        sol_msg.PoC.x = 0; sol_msg.PoC.y = 0; sol_msg.PoC.z = 0;
        sol_msg.n.x = 0; sol_msg.n.y = 0; sol_msg.n.z = 0;
        sol_msg.Fn = 0; 
        sol_msg.Ft.x = 0; sol_msg.Ft.y = 0; sol_msg.Ft.z = 0;
        sol_msg.T = 0;
        sol_msg.D = 0;

        solution_pub.publish(sol_msg);
        */
    }

    ros::spinOnce();
    ros_rate.sleep();

  }

  return 0;
}