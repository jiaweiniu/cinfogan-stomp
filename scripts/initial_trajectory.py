from scipy import interpolate
import numpy as np
import extensions
from cinfogan_initial_traj import gen_points
from models import Generator
from chainer import serializers, Variable
#import train


def cinfogan_initial_traj(q_start, q_goal, n_timesteps):
    t = np.linspace(0,1,n_timesteps)       # divided by timesteps times between 0 to 1
    gen = Generator(60,12,6,2,100)
    serializers.load_npz("../results/models/40.model",gen)
    
    #n_tests = 500
    #n_z = 60
    #n_continuous = 2
    q_start = np.random.random(2)
    q_goal = np.random.random(2)
    obstacles = np.random.random((3,2))
    xi = gen_points(gen,obstacles,q_start,q_goal)
    point_1 = np.array([xi[0], xi[1]])
    point_2 = np.array([xi[2], xi[3]])
    point_3 = np.array([xi[4], xi[5]])

    #print("Initial trajectory points1")
    #print (point_1,point_2,point_3) 
    ξ_x = np.array([q_start[0],point_1[0],point_2[0],point_3[0],q_goal[0]])
    ξ_y = np.array([q_start[1],point_1[1],point_2[1],point_3[1],q_goal[1]])

    xnew = np.linspace(q_start[0], q_start[1], n_timesteps)  
    ynew = interpolate.spline(ξ_x,ξ_y,xnew)

    ξ_0 = np.array([xnew,ynew])
    return ξ_0


def linear_initial_traj(q_start, q_goal, n_timesteps):
    t = np.linspace(0,1,n_timesteps)       

    ξ_0 = np.zeros((2,n_timesteps))
    ξ_0[0] = (1-t)*q_start[0]+t*q_goal[0]
    ξ_0[1] = (1-t)*q_start[1]+t*q_goal[1]

    return ξ_0
