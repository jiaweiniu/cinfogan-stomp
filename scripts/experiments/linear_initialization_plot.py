import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import copy
from matplotlib.lines import Line2D
'''
   plot the linear initialization as the initial trajectory 
'''
# Initialization (set initial trajectory)
n_timesteps=20

q_start=np.asarray([0.1,0.16])
q_goal=np.asarray([0.76,0.88])

t=np.linspace(0,1,num=n_timesteps)       # divided by timesteps times between 0 to 1
theta_x=[t*q_goal[0]+(1-t)*q_start[0]]
theta_y=[t*q_goal[1]+(1-t)*q_start[1]]
theta=np.concatenate((theta_x,theta_y),axis=0)   # initial trajectory
    
print "Initial trajectory"
print
print theta
print


obstacles=[[0.5,0.78],[0.6,0.5],[0.3,0.3]]
start=np.asarray([[q_start[0]],[q_start[1]]])
goal=np.asarray([[q_goal[0]],[q_goal[1]]])

fig=plt.figure()
ax = fig.add_subplot(111,aspect='equal')


for obs in obstacles:
    ax.add_patch(patches.Circle((obs[0], obs[1]),0.1,color='r',alpha=0.6))

plt.plot([0.1,0.76],[0.16,0.88])

ax.scatter(q_start[0],q_start[1],marker="s",s=130,color="black",zorder=20)
ax.scatter(q_goal[0],q_goal[1],marker="*",s=130,color="black",zorder=21)




plt.axis('equal')

line,=ax.plot([],[])
points,=ax.plot([],[],'ro')

plt.xlim(0.0,1.0)
plt.ylim(0.0,1.0)


plt.show()
