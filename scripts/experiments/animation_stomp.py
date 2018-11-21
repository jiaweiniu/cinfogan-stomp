import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib.lines import Line2D

'''
   plot stomp git using cinfogan only in one condition
'''

def animation_stomp(q_start, q_goal, obstacles, list_ξ):
    fig=plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    for obs in obstacles:
        ax.add_patch(patches.Circle((obs[0], obs[1]),0.1,color='r',alpha=0.6))

    start=np.asarray([[q_start[0]],[q_start[1]]])
    goal=np.asarray([[q_goal[0]],[q_goal[1]]])
    for ξ in list_ξ:
        final_ξ=np.concatenate((start,ξ),axis=1)
        final_ξ=np.concatenate((final_ξ,goal),axis=1)


    line,=ax.plot([],[])
    points,=ax.plot([],[],'ro')

    plt.xlim(0.0,1.0)
    plt.ylim(0.0,1.0)

    def update(i,line):
        concat_x=np.concatenate((np.concatenate((start[0],list_ξ[i*2][0])),goal[0]))
        concat_y=np.concatenate((np.concatenate((start[1],list_ξ[i*2][1])),goal[1]))
        line.set_data(concat_x,concat_y)
        points.set_data(list_ξ[i*2][0],list_ξ[i*2][1])
        return line,

    if len(list_ξ)>1:
        ani = animation.FuncAnimation(fig, update, frames=len(list_ξ)//2,blit=False,fargs=[line])

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    

        ani.save('cinfogan-stomp.gif', writer='imagemagick',fps=4)

        plt.show()
    
