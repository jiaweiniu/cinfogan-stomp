import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

def cost(trajectory,obstacle):
    final_cost=np.zeros(trajectory.shape[1])
    obstacle=np.asarray([[obstacle[0]],[obstacle[1]]])
    
    for i in range(trajectory.shape[1]):
        final_cost[i]=0.2*np.exp(-35*np.dot(np.transpose(trajectory[:,i:i+1]-obstacle),trajectory[:,i:i+1]-obstacle))

    return final_cost


def stomp(q_start,q_goal,n_timesteps,K,n_iter,obstacle):
    print
    print "Start point : "+str(q_start) +"   Goal point : "+str(q_goal)
    print 

    # Initialization
    t=np.linspace(0,1,num=n_timesteps)
    theta_x=[t*q_goal[0]+(1-t)*q_start[0]]
    theta_y=[t*q_goal[1]+(1-t)*q_start[1]]
    theta=np.concatenate((theta_x,theta_y),axis=0)
    print "Initial trajectory"
    print theta

    #for m in range(n_iter):
    while(np.sum(cost(theta,obstacle))>0.01):
        zero_mean=np.zeros(n_timesteps)
        noisy_trajectories=np.zeros((K,2,n_timesteps))
        epsilon=np.zeros((K,2,n_timesteps))
        s=np.zeros((K,n_timesteps))

        # For each noisy trajectory
        for k in range(K):
        # For each degree of freedom of the robot
            for i in range(2):
                epsilon[k,i]=np.random.multivariate_normal(zero_mean,R_inv)
                noisy_trajectories[k,i]=theta[i]+epsilon[k,i]
            s[k]=cost(noisy_trajectories[k],obstacle)

        p=np.zeros((K,n_timesteps))
        for j in range(n_timesteps):
            max_s=max(s[:,j])
            min_s=min(s[:,j])
            for k in range(K):
                p[k,j]=np.exp(-10*(s[k,j]-min_s)/(max_s-min_s))
    
        
        # Compute delta theta
        delta_theta=np.zeros((n_timesteps,2))
        delta_theta[:,0]=np.sum(np.dot(np.transpose(epsilon[:,0]),p),axis=0)
        delta_theta[:,1]=np.sum(np.dot(np.transpose(epsilon[:,1]),p),axis=0)

        # Smoothing
        delta_theta=np.dot(M,delta_theta)
    
        # Update trajectory
        theta+=np.transpose(delta_theta)
    
        # Compute trajectory cost
        print np.sum(cost(theta,obstacle))
    return theta

q_start=np.asarray([0.1,0.16])
q_goal=np.asarray([0.76,0.87])

n_timesteps=10
n_noisy=10

dt=0.001
    
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

n_iter=10


obstacle=[0.4,0.64]

traj=stomp(q_start,q_goal,n_timesteps,n_noisy,n_iter,obstacle)

fig=plt.figure()
ax = fig.add_subplot(111, aspect='equal')

line, = ax.plot([], [], 'o-', lw=2)

ax.add_patch(patches.Circle((obstacle[0], obstacle[1]),0.1,color='r',alpha=0.6))

ax.plot(traj[0,:],traj[1,:])
ax.plot(traj[0,:],traj[1,:],'ro')
plt.xlim(0,1)
plt.ylim(0,1)

plt.axis('equal')

plt.show()
