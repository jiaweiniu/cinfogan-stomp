# cinfogan initial trjaectory
import os
from scipy import interpolate
from chainer import serializers
import numpy as np
from models import Generator
from chainer import serializers, Variable
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def gen_points(gen,obstacles,q_start,q_goal):
    n_tests = 1
    n_z = 60
    n_continuous = 2
    z = np.random.uniform(-2, 2, (n_tests,n_z+n_continuous)) # noise and latent code
    z = Variable(np.array(z, dtype=np.float32))
    x = np.random.uniform(0.0, 1.0,(n_tests,12));  # condition(problem) 
    x = Variable(x.astype(np.float32))
    #x = Variable(np.array(x, dtype=np.float32))
    xi = gen(z,x)  # generated points
    xi = xi.data[0]
    return xi

        
'''
if __name__ == '__main__':
    radius = 0.1
    obstacles = np.random.random((3,2))    
    wrong_setup=True
    while wrong_setup:
        q_start = np.random.random(2)
        q_goal = np.random.random(2)
        collide = False
        for obs in obstacles:
            if np.linalg.norm(q_start-obs)<radius or np.linalg.norm(q_goal-obs)<radius:
                collide=True
        if not(collide):
            wrong_setup=False
            
or np.linalg.norm(obs[0]-obs[1])<radius or np.linalg.norm(obs[1]-obs[2])<radius or np.linalg.norm(obs[2]-obs[0])<radius:
            


gen = Generator(60,12,6,2,100)
serializers.load_npz("../results/models/50.model",gen)

 #print obstacles, start, goal, generated points.
    print("obstacles")
    print(obstacles)
    print("start")
    print(q_start)
    print("goal")
    print(q_goal)
    print("gen")
    print(gen)

xi = gen_points(gen,n_tests,n_z,obstacles,q_start,q_goal)
    #print("generated points")
    #print (gen_points(gen,n_tests,n_z,obstacles,q_start,q_goal))   
    
# set obstacles
    fig,ax = plt.subplots(1)  
    circle =patches.Circle((obstacles[0]),radius=0.1,color='red',alpha=0.6)
    ax.add_patch(circle)
    circle = patches.Circle((obstacles[1]),radius=0.1,color='red',alpha=0.6)
    ax.add_patch(circle)
    circle = patches.Circle((obstacles[2]),radius=0.1,color='red',alpha=0.6)
    ax.add_patch(circle)

  
    ax.plot([q_start[0],xi[0],xi[2],xi[4],q_goal[0]],[q_start[1],xi[1],xi[3],xi[5],q_goal[1]],marker='*')
    ax.scatter(q_start[0],q_start[1],marker='s',s=130,color='black',zorder=20)
    ax.scatter(q_goal[0],q_goal[1],marker='*',s=130,color='black',zorder=21)
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    plt.show()

def cinfogan_initial_traj(q_start, q_goal, n_timesteps):
    t = np.linspace(0,1,n_timesteps)       # divided by timesteps times between 0 to 1    
     
    point_1 = np.array([xi[0], xi[1]])
    point_2 = np.array([xi[2], xi[3]])
    point_3 = np.array([xi[4], xi[5]])

    print("Initial trajectory points")
    print (point_1,point_2,point_3) 
    ξ_x = np.array([q_start[0],point_1[0],point_2[0],point_3[0],q_goal[0]])
    ξ_y = np.array([q_start[1],point_1[1],point_2[1],point_3[1],q_goal[1]])

    xnew = np.linspace(q_start[0], q_start[1], n_timesteps)  
    ynew = interpolate.spline(ξ_x,ξ_y,xnew)

    ξ_0 = np.array([xnew,ynew])
    return ξ_0
'''
