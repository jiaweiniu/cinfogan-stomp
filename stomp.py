import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import copy
from matplotlib.lines import Line2D
import time

start_time=time.time()
def cost(trajectory,obstacles,dt):
    final_cost=np.zeros(trajectory.shape[1])    

    for i in range(trajectory.shape[1]):
        for o in obstacles:
            obs=np.asarray([[o[0]],[o[1]]]) 
            final_cost[i]+=0.002*np.exp(-85*np.dot(np.transpose(trajectory[:,i:i+1]-obs),trajectory[:,i:i+1]-obs)) #value is -85
        if (i>0) and (i<trajectory.shape[1]-1) :
            final_cost[i]*=np.linalg.norm(trajectory[:,i+1:i+2]-trajectory[:,i-1:i])/dt
        if abs(trajectory[0,i:i+1]-0.5)>0.5:
            final_cost[i]+=abs(trajectory[0,i:i+1]-0.5)
        if abs(trajectory[1,i:i+1]-0.5)>0.5:
            final_cost[i]+=abs(trajectory[1,i:i+1]-0.5)

    return final_cost


def stomp(q_start,q_goal,n_timesteps,K,n_iter,obstacles,dt):
    print
    print "Start point : "+str(q_start) +"   Goal point : "+str(q_goal)
    print 

    # Initialization (set initial trajectory)

    t=np.linspace(0,1,num=n_timesteps)       # divided by timesteps times between 0 to 1
    print t
    #theta_x=[t*q_goal[0]+(1-t)*q_start[0]]
    #theta_y=[t*q_goal[1]+(1-t)*q_start[1]]
    #theta=np.concatenate((theta_x,theta_y),axis=0)   # initial trajectory

    theta_a=[t*0.3859971+(1-t)*q_start[0]]
    theta_b=[t*0.6825634+(1-t)*q_start[1]]
    
    theta_c=[t*0.4785468+(1-t)*0.3859971]
    theta_d=[t*0.61927265+(1-t)*0.6825634]
    
    theta_e=[t*0.6495005+(1-t)*0.4785468]
    theta_f=[t*0.6653705+(1-t)*0.61927265]

    theta_g=[t*q_goal[0]+(1-t)*0.6495005]
    theta_h=[t*q_goal[1]+(1-t)*0.6653705]

    #theta=np.array([[0.1       , 0.17149928, 0.24299855, 0.31449783, 0.3859971,
            #0.40913452, 0.43227195, 0.45540938, 0.4785468,0.52128522,
            #0.56402365, 0.60676208, 0.6495005,0.67712538, 0.70475025,
            #0.73237513, 0.76],
           #[0.16      , 0.29064085, 0.4212817 , 0.55192255, 0.6825634,
            #0.66674071, 0.65091802, 0.63509534, 0.61927265,0.63079711,
            #0.64232157, 0.65384604, 0.6653705,0.71902787, 0.77268525,
            #0.82634262, 0.88]])

   

    theta_1=np.concatenate((theta_a,theta_b),axis=0)
    theta_2=np.concatenate((theta_b,theta_c),axis=0)
    theta_3=np.concatenate((theta_c,theta_d),axis=0)
    theta_4=np.concatenate((theta_d,theta_e),axis=0)

    theta=np.concatenate((theta_1,theta_2,theta_3,theta_4),axis=0)

    #theta=[[0.1 0.265 0.43 0.595 0.76  0.16936556
            #0.20154923 0.2395552  0.28279699 0.33066036 0.38251732 0.43720925
            #0.49305736 0.54790746 0.59942675 0.64570781 0.68556448 0.7187296]]

    print "Initial trajectory"
    print
    #print theta
    print

    list_theta=[]   # create a []

    list_theta.append(theta)   # [[theta]] 
    m=1
    while(np.sum(cost(theta,obstacles,dt))>2 and m<n_iter):
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
            s[k]=cost(noisy_trajectories[k],obstacles,dt)
            #print noisy_trajectories[1]

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
        print "Iteration : "+str(m)+"  :  "+str(np.sum(cost(theta,obstacles,dt)))
        m+=1
        
        list_theta.append(copy.deepcopy(theta))
    return list_theta

q_start=np.asarray([0.1,0.16])
q_goal=np.asarray([0.76,0.88])

n_timesteps=17
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

n_iter=50


obstacles=[[0.5,0.78],[0.6,0.5],[0.3,0.3]]
#obstacles=[[0.5,0.78],[0.6,0.5],[0.3,0.3],[0.2,0.5]]

traj_list=stomp(q_start,q_goal,n_timesteps,n_noisy,n_iter,obstacles,dt)
print traj_list[-1]

fig=plt.figure()
ax = fig.add_subplot(111, aspect='equal')

for obs in obstacles:
    ax.add_patch(patches.Circle((obs[0], obs[1]),0.1,color='r',alpha=0.6))

start=np.asarray([[q_start[0]],[q_start[1]]])
goal=np.asarray([[q_goal[0]],[q_goal[1]]])
for traj in traj_list:
    final_traj=np.concatenate((start,traj),axis=1)
    final_traj=np.concatenate((final_traj,goal),axis=1)

plt.axis('equal')

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

ani = animation.FuncAnimation(fig, update, frames=len(traj_list)/2,blit=False,fargs=[line])

print ("--- %s seconds ---" %(time.time()-start_time))

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


ani.save('cinfogan-stomp.gif', writer='imagemagick',fps=4)

plt.show()
