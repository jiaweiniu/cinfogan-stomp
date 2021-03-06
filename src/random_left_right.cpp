#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/geometric/PathGeometric.h>

#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/LBTRRT.h>
#include <ompl/geometric/planners/rrt/LazyRRT.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/TRRT.h>
#include <ompl/geometric/planners/rrt/pRRT.h>
#include <ompl/geometric/planners/est/EST.h>

#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <memory>

// All obstacles are considered as circles
struct Obstacle {
  std::vector<float> q;
  float r_squared; // radius squared
};

namespace ob = ompl::base;
namespace og = ompl::geometric;

// Return true if the state is valid, false if the state is invalid
bool isStateValid(const int n_obstacles, const int n_dimensions,const ob::State *state, std::vector<Obstacle> l_o)
{
  const ob::RealVectorStateSpace::StateType* state_nD=state->as<ob::RealVectorStateSpace::StateType>();

  // Extract the robot's (x,y) position from its state
  std::vector<double> state_q;
  for(int i=0;i<n_dimensions;i++) {
    state_q.push_back(state_nD->values[i]);
  }
  
  for (int i=0;i<n_obstacles;i++) {
    double sum=0;
    for(int j=0;j<n_dimensions;j++) {
      sum+=pow(state_q[j]-l_o[i].q[j],2);
    }
    if(sum<l_o[i].r_squared) {return false;}
  }
  // Otherwise, the state is valid:
  return true;
}


bool data_generator(unsigned int k, unsigned int n_obstacles,
		    unsigned int n_dimensions, float radius, bool total_random)
{
  bool found_path=false;
  std::random_device rdev;
  std::mt19937 engine(rdev());
  
  std::uniform_real_distribution<> dist_s_x = std::uniform_real_distribution<>(0.02, 0.15);
  std::uniform_real_distribution<> dist_g_x = std::uniform_real_distribution<>(0.85, 0.98);
  std::uniform_real_distribution<> dist_obstacle_x = std::uniform_real_distribution<>(0.4, 0.6);
  std::uniform_real_distribution<> dist_y = std::uniform_real_distribution<>(0.02, 0.98);
  
  std::vector<float> start_q;
  std::vector<float> goal_q;
  
  start_q.push_back(dist_s_x(engine));
  start_q.push_back(dist_y(engine));
  goal_q.push_back(dist_g_x(engine));
  goal_q.push_back(dist_y(engine));
  
  // Construct the state space where we are planning
  ob::StateSpacePtr space(new ob::RealVectorStateSpace(n_dimensions));

  ob::RealVectorBounds bounds(n_dimensions);
  bounds.setLow(0);
  bounds.setHigh(1);
  space->as<ob::RealVectorStateSpace>()->setBounds(bounds);

  // Instantiate SimpleSetup
  og::SimpleSetup ss(space);

  std::vector<Obstacle> list_obstacles;
  for(unsigned int i=0;i<n_obstacles;i++) {
    struct Obstacle o;
    o.q.push_back(dist_obstacle_x(engine));
    o.q.push_back(dist_y(engine));
    o.r_squared=radius*radius; 
    list_obstacles.push_back(o);
  }
  
  ob::PlannerPtr planner(new og::RRTConnect(ss.getSpaceInformation()));
  ss.setPlanner(planner);
  //the [&] is important, it means to use the list_obstacles outside of the lambda function
  ss.setStateValidityChecker([&](const ob::State *state) { return isStateValid(n_obstacles, n_dimensions,state,list_obstacles); });
    
  // Setup Start and Goal
  ob::ScopedState<> start(space);
  for(unsigned int j=0;j<n_dimensions;j++){
    start->as<ob::RealVectorStateSpace::StateType>()->values[j] = start_q[j];
  }
  ob::ScopedState<> goal(space);
  for(unsigned int j=0;j<n_dimensions;j++){
    goal->as<ob::RealVectorStateSpace::StateType>()->values[j] = goal_q[j];
  }
  ss.setStartAndGoalStates(start, goal);

  // Execute the planning algorithm
  ob::PlannerStatus solved = ss.solve();
  unsigned int n_states=45;
  
  if (solved)
  {
    srand(time(NULL));
    // Simplify the solution 
    ss.simplifySolution();
    
    // Print the solution path to a file
    std::ofstream ofs("../dataset/random_left_right/rrtconnect_12obs/path_"+std::to_string(k+1)+".dat", std::ios::trunc);
    og::PathGeometric path=ss.getSolutionPath();
    path.interpolate(n_states);
    
    if(path.getStateCount()==n_states){
      ofs << start_q[0]<<" "<<start_q[1] << " " << goal_q[0]<<" "<<goal_q[1];

      for(unsigned int i=0;i<n_obstacles;i++){
	for(unsigned int j=0;j<2;j++){
	  ofs << " " << list_obstacles[i].q[j];
	}
      }
      ofs << '\n';
      
      ofs << path.getState(15)->as<ob::RealVectorStateSpace::StateType>()->values[0] << " ";
      ofs << path.getState(15)->as<ob::RealVectorStateSpace::StateType>()->values[1] << " ";
      ofs << path.getState(23)->as<ob::RealVectorStateSpace::StateType>()->values[0] << " ";
      ofs << path.getState(23)->as<ob::RealVectorStateSpace::StateType>()->values[1] << " ";
      ofs << path.getState(31)->as<ob::RealVectorStateSpace::StateType>()->values[0] << " ";
      ofs << path.getState(31)->as<ob::RealVectorStateSpace::StateType>()->values[1];
      found_path=true;
    }
  }
  else {std::cout << "No solution found" << std::endl;}

  return found_path;
}


int main()
{
  unsigned int counter=0;
  unsigned int n_paths=100000;
  
  unsigned int n_obstacles=12;
  unsigned int n_dimensions=2;

  float radius=0.08;

  bool total_random=true;
  while(counter<n_paths) {
    if(data_generator(counter, n_obstacles, n_dimensions, radius, total_random)) {counter++;}
    std::cout<<std::endl<<std::endl<<std::endl<<counter<<std::endl<<std::endl;
  }
  return 0;
}


