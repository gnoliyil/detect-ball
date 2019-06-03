#include "Sai2Model.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "Sai2Primitives.h"

#include <iostream>
#include <string>
#include <vector>

#include <signal.h>
bool runloop = true;
void sighandler(int sig)
{ runloop = false; }

using namespace std;
using namespace Eigen;

const string robot_file = "./resources/panda_arm.urdf";

#define JOINT_CONTROLLER      0
#define POSORI_CONTROLLER     1

int state = JOINT_CONTROLLER;

#define y_max	0.5
#define y_min	-0.5
#define z_max	0.7
#define z_min	0.25
#define x_max	0.4
#define x_min	0.25
#define yx_a	0.55
#define yx_c	1.2
#define zx_a	0.7
#define zx_b	0.88

void clip_des_pos(Sai2Primitives::PosOriTask*& posori_task){
	
	posori_task->_desired_position(1) = min(max(posori_task->_desired_position(1), -y_max), y_max);
	posori_task->_desired_position(2) = min(max(posori_task->_desired_position(2), z_min), z_max);
	// posori_task->_desired_position(0) = zx_b - zx_a * posori_task->_desired_position(2);
	// posori_task->_desired_position(0) = min(max(posori_task->_desired_position(0), x_min), x_max);
	posori_task->_desired_position(0) = x_max;
}

std::string CALI_FLAG;
std::string READ_POS_FLAG;
const std::string READ_POS_Y = "Y";
const std::string READ_POS_N = "N";
const std::string CALI_Y = "Y";
const std::string CALI_N = "N";

// redis keys:
// - read:
std::string JOINT_ANGLES_KEY;
std::string JOINT_VELOCITIES_KEY;
std::string JOINT_TORQUES_SENSED_KEY;

std::string DES_POS_KEY;
std::string READ_POS_KEY;
std::string CALI_KEY;

// - write
std::string JOINT_TORQUES_COMMANDED_KEY;
std::string ROBOT_POSITION_KEY;

// - model
std::string MASSMATRIX_KEY;
std::string CORIOLIS_KEY;
std::string ROBOT_GRAVITY_KEY;

unsigned long long controller_counter = 0;

const bool flag_simulation = false;
// const bool flag_simulation = true;

const bool inertia_regularization = true;

int main() {

	if(flag_simulation)
	{
		JOINT_ANGLES_KEY = "sai2::cs225a::panda_robot::sensors::q";
		JOINT_VELOCITIES_KEY = "sai2::cs225a::panda_robot::sensors::dq";
		JOINT_TORQUES_COMMANDED_KEY = "sai2::cs225a::panda_robot::actuators::fgc";
	}
	else
	{
		JOINT_TORQUES_COMMANDED_KEY = "sai2::FrankaPanda::actuators::fgc";

		JOINT_ANGLES_KEY  = "sai2::FrankaPanda::sensors::q";
		JOINT_VELOCITIES_KEY = "sai2::FrankaPanda::sensors::dq";
		JOINT_TORQUES_SENSED_KEY = "sai2::FrankaPanda::sensors::torques";
		MASSMATRIX_KEY = "sai2::FrankaPanda::sensors::model::massmatrix";
		CORIOLIS_KEY = "sai2::FrankaPanda::sensors::model::coriolis";
		ROBOT_GRAVITY_KEY = "sai2::FrankaPanda::sensors::model::robot_gravity";		
	}

	//----------------------Sync------------------------------
	DES_POS_KEY = "sai2::FrankaPanda::control::target_position";
	READ_POS_KEY = "sai2::FrankaPanda::control::read_position";
	CALI_KEY = "sai2::FrankaPanda::control::cali";
	ROBOT_POSITION_KEY = "sai2::FrankaPanda::sensors::current_position";
	//----------------------Sync------------------------------


	// start redis client
	auto redis_client = RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load robots
	auto robot = new Sai2Model::Sai2Model(robot_file, false);
	robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
	VectorXd initial_q = robot->_q;
	robot->updateModel();

	// prepare controller
	int dof = robot->dof();
	VectorXd command_torques = VectorXd::Zero(dof);
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);

	// pose task
	const string control_link = "link7";
	const Vector3d control_point = Vector3d(0,0,0.05);
	auto posori_task = new Sai2Primitives::PosOriTask(robot, control_link, control_point);

#ifdef USING_OTG
	posori_task->_use_interpolation_flag = true;
	posori_task->_otg->setMaxLinearVelocity(0.5);
	posori_task->_otg->setMaxLinearAcceleration(1.0);
	posori_task->_otg->setMaxLinearJerk(3.0);

	posori_task->_otg->setMaxAngularVelocity(M_PI/3);
	posori_task->_otg->setMaxAngularAcceleration(M_PI);
	posori_task->_otg->setMaxAngularJerk(3*M_PI);
#else
	posori_task->_use_velocity_saturation_flag = true;
	posori_task->_linear_saturation_velocity = 0.5;
	posori_task->_angular_saturation_velocity = M_PI/3;
#endif
	
	VectorXd posori_task_torques = VectorXd::Zero(dof);
	posori_task->_kp_pos = 200.0;
	posori_task->_kv_pos = 20.0;
	posori_task->_kp_ori = 200.0;
	posori_task->_kv_ori = 20.0;

	// joint task
	auto joint_task = new Sai2Primitives::JointTask(robot);
	auto ori_task = new Sai2Primitives::OrientationTask(robot, control_link, control_point);

#ifdef USING_OTG
	joint_task->_use_interpolation_flag = true;
	joint_task->_otg->setMaxVelocity(M_PI/3);
	joint_task->_otg->setMaxAcceleration(M_PI);
	joint_task->_otg->setMaxJerk(3*M_PI);
#else
	joint_task->_use_velocity_saturation_flag = true;
	joint_task->_saturation_velocity = M_PI/3.0*Eigen::VectorXd::Ones(dof);
#endif

	VectorXd joint_task_torques = VectorXd::Zero(dof);
	joint_task->_kp = 50.0;
	joint_task->_kv = 10.0;

	VectorXd q_init_desired = initial_q;
	q_init_desired << 0.0, 0.0, 0.0, -90.0, 0.0, 107.0, 0.0;
	q_init_desired *= M_PI/180.0;
	joint_task->_desired_position = q_init_desired;

	// create a timer
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(1000); 
	double start_time = timer.elapsedTime(); //secs
	bool fTimerDidSleep = true;



	//-------------------------------------Check Sweep-----------------------
	// vector<Vector3d> corners = {Vector3d(0.25,0.53,0.9), Vector3d(0.25,0.53,0.1), Vector3d(0.35,0.0,0.1), Vector3d(0.25,-0.53,0.1), Vector3d(0.25,-0.53,0.9), Vector3d(0.35,0.0,0.9)};
	// vector<Vector3d> corners = {Vector3d(-0.25,0.45,1.0), Vector3d(-0.25,0.45,0.1),Vector3d(-0.25,-0.45,0.1), Vector3d(-0.25,-0.45,1.0), Vector3d(0, 0, 0.55)};
	// vector<Vector3d> corners = {Vector3d(0.25,0.45,0.7), Vector3d(0.25,0.45,0.15),Vector3d(0.25,-0.45,0.15), Vector3d(0.25,-0.45,0.7), Vector3d(0.45, 0, 0.55)};

	// unsigned int cor = 0;

	// posori_task->_desired_position = corners[0];
	//----------------------------------------------------------------------

	// posori_task->_desired_position = Vector3d(0.45,0.0,0.1);

	//--------------------------------------Face forward--------------------------
	posori_task->_desired_orientation.setIdentity();
	// posori_task->_desired_orientation = AngleAxisd(-M_PI, Vector3d::UnitZ()).toRotationMatrix() * posori_task->_desired_orientation;
	// posori_task->_desired_orientation = AngleAxisd(M_PI/2, Vector3d::UnitY()).toRotationMatrix() * posori_task->_desired_orientation;
	posori_task->_desired_orientation = AngleAxisd(M_PI, Vector3d::UnitX()).toRotationMatrix() * posori_task->_desired_orientation;
	//----------------------------------------------------------------------

	Vector3d pos, delta_phi;
	Matrix3d rot;

	while (runloop) {
		// wait for next scheduled loop
		timer.waitForNextLoop();
		double time = timer.elapsedTime() - start_time;

		// read robot state from redis
		robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
		robot->_dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);

		// update model
		if(flag_simulation)
		{
			robot->updateModel();
		}
		else
		{
			robot->updateKinematics();
			robot->_M = redis_client.getEigenMatrixJSON(MASSMATRIX_KEY);
			// if(inertia_regularization)
			// {
			// 	robot->_M(4,4) += 0.07;
			// 	robot->_M(5,5) += 0.07;
			// 	robot->_M(6,6) += 0.07;
			// }
			robot->_M_inv = robot->_M.inverse();
		}

		if(state == JOINT_CONTROLLER)
		{
			// update task model and set hierarchy
			N_prec.setIdentity();
			joint_task->updateTaskModel(N_prec);

			if(inertia_regularization)
			{
				robot->_M += 0.1 * MatrixXd::Identity(dof,dof);
			}

			// compute torques
			joint_task->computeTorques(joint_task_torques);

			command_torques = joint_task_torques;

			if (redis_client.exists(DES_POS_KEY) && redis_client.exists(READ_POS_KEY)){
				READ_POS_FLAG = redis_client.get(READ_POS_KEY);
				if (READ_POS_FLAG == READ_POS_Y){
					state = POSORI_CONTROLLER;
				}
			}
		}

		else if(state == POSORI_CONTROLLER)
		{

			robot->position(pos, control_link, control_point);
			robot->rotation(rot, control_link);

			if (redis_client.exists(DES_POS_KEY)){
				posori_task->_desired_position = redis_client.getEigenMatrixJSON(DES_POS_KEY);
			} else {
				state = JOINT_CONTROLLER;
				posori_task->_desired_position = pos;
			}

			clip_des_pos(posori_task);

			N_prec.setIdentity();
			posori_task->updateTaskModel(N_prec);
			N_prec = posori_task->_N;
			joint_task->updateTaskModel(N_prec);

			if(inertia_regularization)
			{
				posori_task->_Lambda += 0.1 * MatrixXd::Identity(6,6);
			}

			// compute torques
			posori_task->computeTorques(posori_task_torques);
			joint_task->computeTorques(joint_task_torques);

			command_torques = posori_task_torques + joint_task_torques;

			
			Sai2Model::orientationError(delta_phi,posori_task->_desired_orientation, rot);

			if (redis_client.exists(CALI_KEY)){
				CALI_FLAG = redis_client.get(CALI_KEY);
				if (CALI_FLAG == CALI_Y){
					redis_client.setEigenMatrixJSON(ROBOT_POSITION_KEY, pos);
				}
			}

			if (redis_client.exists(READ_POS_KEY)){
				READ_POS_FLAG = redis_client.get(READ_POS_KEY);
				if (READ_POS_FLAG == READ_POS_N){
					state = JOINT_CONTROLLER;
				}
			}
		}

		// send to redis
		redis_client.setEigenMatrixJSON(JOINT_TORQUES_COMMANDED_KEY, command_torques);
		
		controller_counter++;

	}

	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Controller Loop run time  : " << end_time << " seconds\n";
    std::cout << "Controller Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Controller Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";

	return 0;
}
