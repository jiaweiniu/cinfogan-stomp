import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib.lines import Line2D
import time
from stomp import stomp
from initial_trajectory import cinfogan_initial_traj

if __name__ == '__main__':
    q_start=np.asarray([0.1,0.16])
    q_goal=np.asarray([0.76,0.88])

    n_timesteps=33
    n_noisy=20

    dt=0.001/n_timesteps
    
    A=np.zeros((n_timesteps+2,n_timesteps))

    A[0][0]=1/dt
    A[1][0]=-2/dt
    A[1][1]=1/dt

    A[n_timesteps+1][n_timesteps-1]=1/dt
    A[n_timesteps][n_timesteps-1]=-2/dt
    A[n_timesteps][n_timesteps-2]=1/dt
    
    for i in range(n_timesteps-2):
        A[i+2][i]=1/dt
        A[i+2][i+1]=-2/dt
        A[i+2][i+2]=1/dt
    
    R_inv=np.linalg.inv(np.dot(A.T,A))
    M=np.zeros((n_timesteps,n_timesteps))
    for i in range(n_timesteps):
        M[:,i]=R_inv[:,i]/(n_timesteps*max(R_inv[:,i]))

    n_iter=30
    
    obstacles=[[0.5,0.78],[0.6,0.5],[0.3,0.3]]

    start_time=time.time()

    ξ_0 = cinfogan_initial_traj(q_start, q_goal, n_timesteps)
    traj_list=stomp(q_start, q_goal, ξ_0, n_timesteps, n_noisy, R_inv, M, n_iter, obstacles, dt)
    print()
    print("--- %s seconds ---" %(time.time()-start_time))

    fig=plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    for obs in obstacles:
        ax.add_patch(patches.Circle((obs[0], obs[1]),0.1,color='r',alpha=0.6))

    start=np.asarray([[q_start[0]],[q_start[1]]])
    goal=np.asarray([[q_goal[0]],[q_goal[1]]])
    for traj in traj_list:
        final_traj=np.concatenate((start,traj),axis=1)
        final_traj=np.concatenate((final_traj,goal),axis=1)


    line,=ax.plot([],[])
    points,=ax.plot([],[],'ro')

    plt.xlim(0.0,1.0)
    plt.ylim(0.0,1.0)

    def update(i,line):
        concat_x=np.concatenate((np.concatenate((start[0],traj_list[i*2][0])),goal[0]))
        concat_y=np.concatenate((np.concatenate((start[1],traj_list[i*2][1])),goal[1]))
        line.set_data(concat_x,concat_y)
        points.set_data(traj_list[i*2][0],traj_list[i*2][1])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(traj_list)//2,blit=False,fargs=[line])

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    

    ani.save('cinfogan-stomp.gif', writer='imagemagick',fps=4)

    plt.show()
