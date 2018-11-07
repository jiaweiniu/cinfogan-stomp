import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import interpolate
from models import Generator, Discriminator, Critic

from chainer import serializers, Variable

import import_dataset
import rustmetric

# Measure the percentage of collision of the trajectory and compare it to the linear initialization

def collisions_measure(x_list,xi_pred_list):
    measure_lin_col=[]
    ratio_lin_col=[]
    measure_infogan_col=[]
    ratio_infogan_col=[]

    list_xi_lin=[]
    list_xi_infogan=[]
    for x,xi_pred in zip(x_list,xi_pred_list):
        t=np.linspace(0,1,100)
        t=np.asarray([t])
        
        x=import_dataset.denormalize_data_random_left_right(x)  # import dataset from random left right
        start=np.asarray([x[0:2]]) # give the position of start 
        goal=np.asarray([x[2:4]])  # give the position of goal
       
        # linear initialization

        lin=np.dot(1-np.transpose(t),start)+np.dot(np.transpose(t),goal)  # zhuan(3)zhi(4)

        # infogan initialization

        t_xi=[0,0.333,0.5,0.644,1]
        before_spline=np.array([[start[0][0],start[0][1]],
                                [xi_pred[0],xi_pred[1]],
                                [xi_pred[2],xi_pred[3]],
                                [xi_pred[4],xi_pred[5]],
                                [goal[0][0],goal[0][1]]])     # become an array
        
        spline,u=interpolate.splprep([before_spline[:,0],before_spline[:,1]],u=t_xi)  # cha(1)zhi(2)
    
        infogan=interpolate.splev(t,spline)
        infogan=np.concatenate((np.transpose(infogan[0]),np.transpose(infogan[1])),axis=1)

        list_xi_lin.append(lin.tolist())
        list_xi_infogan.append(infogan.tolist())
        
    n_dim=2
    n_obs=4
    collision_limit=0.1

    f1_lin  = rustmetric.f1(x_list.tolist(),list_xi_lin, n_dim,n_obs,collision_limit)
    f1_infogan = rustmetric.f1(x_list.tolist(),list_xi_infogan,n_dim,n_obs,collision_limit)

    f1_lin = 100.0*f1_lin/len(x_list)
    f1_infogan = 100.0*f1_infogan/len(x_list)
    
    return [f1_lin, f1_infogan, f1_lin-f1_infogan]
            
def test(gen,n_tests,n_z, n_continuous):
    z = np.random.uniform(-2, 2, (n_tests, n_z+n_continuous))
   
    z = Variable(np.array(z, dtype=np.float32))
    
    x = np.random.uniform(0.0, 1.0, (n_tests,12));
    x = Variable(x.astype(np.float32))
    xi= gen(z,x)
   

    return collisions_measure(x.data,xi.data)
    

if __name__ == '__main__':
    gen=Generator(60, 12, 6, 2, 100)
    serializers.load_npz("../results/models/tmp/10_gen.model",gen)
    print(gen)
    print(test(gen,50,20,2))
