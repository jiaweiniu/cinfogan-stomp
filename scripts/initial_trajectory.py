from scipy import interpolate
import numpy as np
import extensions
from cinfogan_initial_traj import gen_points
from models import Generator
from chainer import serializers, Variable
from tqdm import tqdm
    

def cinfogan_initial_traj(q_start, q_goal, n_timesteps):
    t = np.linspace(0,1,n_timesteps)# divided by timesteps times between 0 to 1
    gen = Generator(60,12,6,2,100)
    serializers.load_npz("../results/models/40.model",gen)
    x = gen_points(gen)
    point_1 = np.array([x[0], x[1]])
    point_2 = np.array([x[2], x[3]])
    point_3 = np.array([x[4], x[5]])
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
