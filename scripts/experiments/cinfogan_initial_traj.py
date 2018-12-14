# cinfogan initial trjaectory
import os
from scipy import interpolate
from chainer import serializers
import numpy as np
from models import Generator
from chainer import serializers, Variable
import matplotlib.patches as patches
import matplotlib.pyplot as plt
'''
   use cinfogan model, generate three waypoints.
'''

def gen_points(gen, n_tests, n_z, obstacles, q_start, q_goal, n_continuous):
    z = np.random.uniform(-2, 2, (n_tests, n_z+n_continuous))
    z = z.astype(np.float32)
    x = np.random.uniform(0.0, 1.0,(n_tests,12));
    x = x.astype(np.float32)
    xi = gen(z,x)
    xi = xi.data[0]
    
    return xi
gen = Generator(62, 12, 6, 70)
serializers.load_npz("../training/results/models/cinfogan_models_0/50_gen.model",gen)
n_timesteps=20

q_start=np.asarray([0.1,0.16])
q_goal=np.asarray([0.76,0.88])

obstacles=[[0.5,0.78],[0.6,0.5],[0.3,0.3]]
q_start=np.asarray([[q_start[0]],[q_start[1]]])
q_goal=np.asarray([[q_goal[0]],[q_goal[1]]])


# print the three waypoints 
xi = gen_points(gen,50,60,obstacles,q_start,q_goal,2)

    
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


