import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.patches as patches
from matplotlib import animation
import copy
from scipy import interpolate
from matplotlib.lines import Line2D
import time

start_time=time.time()
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


def stomp(q_start,q_goal,n_timesteps,K,n_iter,obstacles,dt):
    print
    print "Start point : "+str(q_start) +"   Goal point : "+str(q_goal)
    print 

    # Initialization (set initial trajectory)

    t=np.linspace(0,1,num=n_timesteps)       # divided by timesteps times between 0 to 1
    
    t1=np.linspace(0,1,num=n_timesteps1)       # divided by timesteps times between 0 to 1
   

    #theta=np.array([[0.1       , 0.17149928, 0.24299855, 0.31449783, 0.3859971,
            #0.40913452, 0.43227195, 0.45540938, 0.4785468,0.52128522,
            #0.56402365, 0.60676208, 0.6495005,0.67712538, 0.70475025,
            #0.73237513, 0.76],
           #[0.16      , 0.29064085, 0.4212817 , 0.55192255, 0.6825634,
            #0.66674071, 0.65091802, 0.63509534, 0.61927265,0.63079711,
            #0.64232157, 0.65384604, 0.6653705,0.71902787, 0.77268525,
            #0.82634262, 0.88]])
     


    point_1 = np.array([0.3859971,0.6825634])
    point_2 = np.array([0.4785468,0.61927265])
    point_3 = np.array([0.6495005,0.6653705])

    """
    theta_abx = t1*point_1[0]+(1-t1)*q_start[0]
    theta_aby = t1*point_1[1]+(1-t1)*q_start[1]
    theta_bcx = t1*point_2[0]+(1-t1)*point_1[0]
    theta_bcy = t1*point_2[1]+(1-t1)*point_1[1]
    theta_cdx = t1*point_3[0]+(1-t1)*point_2[0]
    theta_cdy = t1*point_3[1]+(1-t1)*point_2[1]
    theta_dex = t1*q_goal[0]+(1-t1)*point_3[0]
    theta_dey = t1*q_goal[1]+(1-t1)*point_3[1]

    print type(theta_abx)
    print type(theta_bcx[1:5])
    print "1"
    print theta_abx[1:5] + theta_bcx[1:5]
    theta_x =[theta_abx, theta_bcx[1:5], theta_cdx[1:5], theta_dex[1:5]]
    print theta_x
    

    theta_y =list(theta_aby)+list(theta_bcy[1:5]) + list(theta_cdy[1:5]) + list(theta_dey[1:5])
    
   
  

    #theta_x = [0.1       , 0.17149928, 0.24299855, 0.31449783, 0.3859971,
            #0.40913452, 0.43227195, 0.45540938, 0.4785468,0.52128522,
            #0.56402365, 0.60676208, 0.6495005,0.67712538, 0.70475025,
            #0.73237513, 0.76]
    #theta_y = [0.16      , 0.29064085, 0.4212817 , 0.55192255, 0.6825634,
            #0.66674071, 0.65091802, 0.63509534, 0.61927265,0.63079711,
            #0.64232157, 0.65384604, 0.6653705,0.71902787, 0.77268525,
            #0.82634262, 0.88]

    """
    theta_x = np.array([q_start[0],point_1[0],point_2[0],point_3[0],q_goal[0]])

    theta_y = np.array([q_start[1],point_1[1],point_2[1],point_3[1],q_goal[1]])

    xnew = np.arange(0.1,0.76,0.02)   # divide by 33 times
    #xnew = np.linespace(0,1,0.02)
   
    #func = interpolate.interp1d(theta_x,theta_y,kind='quadratic')
    func = interpolate.spline(theta_x,theta_y,xnew)
    print func

    ynew = func

    theta = np.array([xnew,ynew])

    print "Initial trajectory"
    print
    print theta
    print

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
        cost_final = str(np.sum(cost(theta,obstacles,dt)))
        m+=1
        list_theta.append(copy.deepcopy(theta))

        if (cost_final >= str(12.0)):
           
            print "Iteration : "+str(m)+"  :  "+cost_final
            continue
            
        else:
            print "finished"
            break

        #for o in obstacles:
           #for i in xnew:
               #for j in ynew:
                   #dist = np.sqrt(np.sum(np.square(np.array(i,j)-o)))
                   #print "dist"
                   #print dist
        
        #if (dist < 0.1):
           
            #print "Iteration : "+str(m)+"  :  "+cost_final
            #continue
            
        #else:
            #print "finished"
            #break

            
    return list_theta

q_start=np.asarray([0.1,0.16])
q_goal=np.asarray([0.76,0.88])

n_timesteps=33
n_timesteps1=2
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
#print "obstacle1"
#print obstacles[0][1]
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
