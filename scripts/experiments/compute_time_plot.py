import numpy as np
from numpy import linalg as LA
import time
from stomp import stomp
from initial_trajectory import cinfogan_initial_traj, cgan_initial_traj, linear_initial_traj
from animation_stomp import animation_stomp
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from chainer import serializers
from models import Generator
'''
   plot computing time that linear, cgan, cinfogan as initial trajectory 
'''
#--- STOMP parameters ---#
if __name__ == '__main__':
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

    cinfogan_gen = Generator(62, 12, 6, 70)
    serializers.load_npz("../training/results/models/cinfogan_models_0/40_gen.model",cinfogan_gen)

    cgan_gen = Generator(62, 12, 6, 70)
    serializers.load_npz("../training/results/models/cgan_models_0/20_gen.model",cgan_gen)
        
    record_list_ξ = False
    verbose=False
    data=[]
    n_experiments = 20
    #--- Problem x definition ---#
    for i in tqdm(range(n_experiments)):
        radius=0.1
        obstacles=np.random.random((3,2))
        wrong_setup=True
        while wrong_setup:
            q_start=np.random.random(2)
            q_goal=np.random.random(2)
            collide = False

            for obs in obstacles:
                if np.linalg.norm(q_start-obs)<radius or np.linalg.norm(q_goal-obs)<radius:
                    collide=True
            if not(collide):
                wrong_setup=False
    
        #--- Linear initial trajectory ---#
        start_time=time.time()
        ξ_0 = linear_initial_traj(q_start, q_goal, n_timesteps)
                
        if record_list_ξ:
            list_ξ, iterations = stomp(q_start, q_goal, ξ_0, n_timesteps, n_noisy, R_inv, M, n_iter, obstacles, dt, record_list_ξ)
            
            end_time = time.time()
            animation_stomp(q_start, q_goal, obstacles, list_ξ)
                    
        else:
            ξ, iterations = stomp(q_start, q_goal, ξ_0, n_timesteps, n_noisy, R_inv, M,
                                  n_iter, obstacles, dt, record_list_ξ)
            end_time = time.time()
            
        data.append(["Linear", (end_time-start_time), iterations])

        #--- Cgan initial trajectory ---#
        start_time=time.time()
        ξ_0 = cgan_initial_traj(cgan_gen, q_start, q_goal, n_timesteps)
            
        if record_list_ξ:
            list_ξ, iterations = stomp(q_start, q_goal, ξ_0, n_timesteps, n_noisy, R_inv, M, n_iter, obstacles, dt, record_list_ξ)
                
            end_time = time.time()
            animation_stomp(q_start, q_goal, obstacles, list_ξ)
                
        else:
            ξ, iterations = stomp(q_start, q_goal, ξ_0, n_timesteps, n_noisy, R_inv, M,
                                      n_iter, obstacles, dt, record_list_ξ)
            end_time = time.time()
        data.append(["CGAN", (end_time-start_time), iterations])
        
        #--- Cinfogan initial trajectory ---#
        start_time=time.time()
        ξ_0 = cinfogan_initial_traj(cinfogan_gen, q_start, q_goal, n_timesteps)
            
        if record_list_ξ:
            list_ξ, iterations = stomp(q_start, q_goal, ξ_0, n_timesteps, n_noisy, R_inv, M, n_iter, obstacles, dt, record_list_ξ)
                
            end_time = time.time()
            animation_stomp(q_start, q_goal, obstacles, list_ξ)
                
        else:
            ξ, iterations = stomp(q_start, q_goal, ξ_0, n_timesteps, n_noisy, R_inv, M,
                                      n_iter, obstacles, dt, record_list_ξ)
            end_time = time.time()
        data.append(["CInfoGAN", (end_time-start_time), iterations])
        if verbose:
            print()
            print("--- %s seconds ---" %(end_time-start_time))
    df = pd.DataFrame(data=data,columns=["Initialization","Computing time","Iterations"])
    sns.boxplot(x = "Initialization", y = "Computing time",data=df,width=0.8,showfliers=False)#hue = "Initialization", data=df,showfliers=False)                
    plt.show()
