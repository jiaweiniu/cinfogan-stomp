import numpy as np
import copy
from scipy import interpolate

def cost(trajectory,obstacles,dt):
    final_cost=np.zeros(trajectory.shape[1])    

    for i in range(trajectory.shape[1]):
        for o in obstacles:
            obs=np.asarray([[o[0]],[o[1]]]) 
            final_cost[i]+=0.002*np.exp(-85*np.dot(np.transpose(trajectory[:,i:i+1]-obs),trajectory[:,i:i+1]-obs))  #value is -85
        if (i>0) and (i<trajectory.shape[1]-1) :
            final_cost[i]*=np.linalg.norm(trajectory[:,i+1:i+2]-trajectory[:,i-1:i])/dt
        if abs(trajectory[0,i:i+1]-0.5)>0.5:
            final_cost[i]+=abs(trajectory[0,i:i+1]-0.5)
        if abs(trajectory[1,i:i+1]-0.5)>0.5:
            final_cost[i]+=abs(trajectory[1,i:i+1]-0.5)

    return final_cost


def stomp(q_start, q_goal, n_timesteps, K, R_inv, M, n_iter, obstacles, dt):
    print()
    print("Start point : "+str(q_start) +"   Goal point : "+str(q_goal))
    print("Generation of initial trajectory")

    # Initialization (set initial trajectory)

    t = np.linspace(0,1,num=n_timesteps)       # divided by timesteps times between 0 to 1    
     
    point_1 = np.array([0.3859971,0.6825634])
    point_2 = np.array([0.4785468,0.61927265])
    point_3 = np.array([0.6495005,0.6653705])
    
    ξ_x = np.array([q_start[0],point_1[0],point_2[0],point_3[0],q_goal[0]])
    ξ_y = np.array([q_start[1],point_1[1],point_2[1],point_3[1],q_goal[1]])

    xnew = np.linspace(0.1,0.76,33)   # divide by 33 times
   
    #func = interpolate.interp1d(theta_x,theta_y,kind='quadratic')
    func = interpolate.spline(ξ_x,ξ_y,xnew)

    ynew = func

    ξ = np.array([xnew,ynew])


    list_ξ=[]   # create a []

    list_ξ.append(ξ)   # [[theta]]

    print("Beginning of STOMP")
    m=0
    cost_final = 14
    while(np.sum(cost(ξ,obstacles,dt))>2 and m<n_iter and cost_final >= 12.0):
        zero_mean=np.zeros(n_timesteps)
        noisy_trajectories=np.zeros((K,2,n_timesteps))
        epsilon=np.zeros((K,2,n_timesteps))
        s=np.zeros((K,n_timesteps))
        
        # For each noisy trajectory
        for k in range(K):
            # For each degree of freedom of the robot
            for i in range(2):
                epsilon[k,i]=np.random.multivariate_normal(zero_mean,R_inv)
                noisy_trajectories[k,i]=ξ[i]+epsilon[k,i]
            s[k]=cost(noisy_trajectories[k],obstacles,dt)

        p=np.zeros((K,n_timesteps))
        for j in range(n_timesteps):
            max_s=max(s[:,j])
            min_s=min(s[:,j])
            for k in range(K):
                p[k,j]=np.exp(-10*(s[k,j]-min_s)/(max_s-min_s))
    
        # Compute delta theta
        δξ=np.zeros((n_timesteps,2))
        δξ[:,0]=np.sum(np.dot(np.transpose(epsilon[:,0]),p),axis=0)
        δξ[:,1]=np.sum(np.dot(np.transpose(epsilon[:,1]),p),axis=0)

        # Smoothing
        δξ=np.dot(M,δξ)
    
        # Update trajectory
        ξ+=np.transpose(δξ)
    
        # Compute trajectory cost
        cost_final = np.sum(cost(ξ,obstacles,dt))
        m+=1
        list_ξ.append(copy.deepcopy(ξ))

        print("Iteration : "+str(m)+"  :  "+str(cost_final))
        
    print("Finished")
            
    return list_ξ
