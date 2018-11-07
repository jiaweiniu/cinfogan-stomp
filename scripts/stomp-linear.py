import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import copy
from matplotlib.lines import Line2D
import time

start_time = time.time()

def cost(trajectory,obstacles,dt):
    final_cost=np.zeros(trajectory.shape[1])
    
    for i in range(trajectory.shape[1]):
        #print i
        #print "i"
        for o in obstacles:
            obs=np.asarray([[o[0]],[o[1]]])
            final_cost[i]+=0.002*np.exp(-85*np.dot(np.transpose(trajectory[:,i:i+1]-obs),trajectory[:,i:i+1]-obs))  #value is -85
            #print "trajectory"
            #print
            #print trajectory[:,i:i+1]
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
    theta_x=[t*q_goal[0]+(1-t)*q_start[0]]   # x coordinate
    theta_y=[t*q_goal[1]+(1-t)*q_start[1]]   # y coordinate
    theta=np.concatenate((theta_x,theta_y),axis=0)   # initial trajectory
    
    print "Initial trajectory"
    print
    print theta
    print

    plt.show()
    
    list_theta=[]   # create a []

    list_theta.append(theta)   # [[theta]] 
    m=0
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
            #print "noise_traj"
            #print noisy_trajectories[0][1]

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

        cost_final = str(np.sum(cost(theta,obstacles,dt)))
        m+=1
        
        #print "theta_x"
        #print theta_x[0]
        #print
        #print "theta_y"
        #print theta_y[0]

        
        

        #print "Iteration : "+str(m)+"  :  "+str(np.sum(cost(theta,obstacles,dt)))
        #print "noisy0" 
        #print noisy_trajectories[0]
        #print
        #print "noisy1"
        #print noisy_trajectories[1]
        #print
        #print "noisy2"
        #print [noisy_trajectories[2][0][1],noisy_trajectories[2][1][1]]
        #print noisy_trajectories[2][1][1]

        list_theta.append(copy.deepcopy(theta))

        #for o in obstacles:
            #for n in noisy_trajectories:
                #for i in n[0]:
                    #for j in n[1]:
                        #dist = np.sqrt(np.sum(np.square(np.array(i,j)-o)))
                        #print "dist"
                        #print dist

        #if (str(dist) < str(0.1)):
            #print "Iteration : "+str(m)+"  :  "+cost_final
            #print "iteratie"
            #continue
            
        #else:
            #print "finished"
            #break

            
        
        if (cost_final >= str(12.0)):
           
            print "Iteration : "+str(m)+"  :  "+cost_final
            continue
            
        else:
            print "finished"
            break

        
              
        
    return list_theta


q_start=np.asarray([0.1,0.16])
q_goal=np.asarray([0.76,0.88])

n_timesteps=20
n_noisy=20

n_iter=50

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




obstacles=[[0.5,0.78],[0.6,0.5],[0.3,0.3]]
#obstacles=[[0.6,0.5],[0.3,0.3],[0.2,0.6]]

traj_list=stomp(q_start,q_goal,n_timesteps,n_noisy,n_iter,obstacles,dt)
print
print "final_trajectory"
print
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

print
print ("=== %s seconds ---" %(time.time() -start_time))

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ani.save('usualstomp.gif',writer='imagemagick',fps=4)

#ani.save('usualstomp.gif',writer='imagemagick')
plt.show()
