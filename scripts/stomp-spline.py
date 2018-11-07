import numpy as np
from numpy import linalg as LA
import time
from stomp import stomp
from initial_trajectory import cinfogan_initial_traj, linear_initial_traj 
from animation_stomp import animation_stomp
from tqdm import tqdm

if __name__ == '__main__':
    #--- STOMP parameters ---#
    n_timesteps=50
    dt=0.001/n_timesteps
    n_noisy=20
    n_iter=30
    
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

    #--- Problem x definition ---#
    q_start=np.asarray([0.1,0.16])
    q_goal=np.asarray([0.76,0.88])
    obstacles=[[0.5,0.78],[0.6,0.5],[0.3,0.3]]

    record_list_ξ = False
    verbose=False
    for i in tqdm(range(10)):
        start_time=time.time()
        ξ_0 = linear_initial_traj(q_start, q_goal, n_timesteps)

        if record_list_ξ:
            list_ξ = stomp(q_start, q_goal, ξ_0, n_timesteps, n_noisy, R_inv, M,
                           n_iter, obstacles, dt, record_list_ξ)

            end_time = time.time()
            animation_stomp(q_start, q_goal, obstacles, list_ξ)

        else:
            ξ = stomp(q_start, q_goal, ξ_0, n_timesteps, n_noisy, R_inv, M,
                      n_iter, obstacles, dt, record_list_ξ)
            end_time = time.time()

        if verbose:
            print()
            print("--- %s seconds ---" %(end_time-start_time))

