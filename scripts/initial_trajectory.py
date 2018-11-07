from scipy import interpolate
import numpy as np

def cinfogan_initial_traj(q_start, q_goal, n_timesteps):
    t = np.linspace(0,1,n_timesteps)       # divided by timesteps times between 0 to 1    
     
    point_1 = np.array([0.3859971,0.6825634])
    point_2 = np.array([0.4785468,0.61927265])
    point_3 = np.array([0.6495005,0.6653705])
    
    ξ_x = np.array([q_start[0],point_1[0],point_2[0],point_3[0],q_goal[0]])
    ξ_y = np.array([q_start[1],point_1[1],point_2[1],point_3[1],q_goal[1]])

    xnew = np.linspace(0.1,0.76,33)   # divide by 33 times
   
    #func = interpolate.interp1d(theta_x,theta_y,kind='quadratic')
    func = interpolate.spline(ξ_x,ξ_y,xnew)

    ynew = func

    ξ_0 = np.array([xnew,ynew])
    return ξ_0
