import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import interpolate

from chainer import serializers, Variable

import import_dataset
import rustmetric

# Measure the percentage of collision of the trajectory and compare it to the linear initialization
def collisions_measure(x_list,xi_pred_list):
    measure_lin_col=[]
    ratio_lin_col=[]
    measure_cgan_col=[]
    ratio_cgan_col=[]

    list_xi_lin=[]
    list_xi_cgan=[]
    for x,xi_pred in zip(x_list,xi_pred_list):
        t=np.linspace(0,1,100)
        t=np.asarray([t])

        x=import_dataset.denormalize_data_random_left_right(x)
        start=np.asarray([x[0:2]])
        goal=np.asarray([x[2:4]])
        
        # linear initialization
        lin=np.dot(1-np.transpose(t),start)+np.dot(np.transpose(t),goal)

        # cgan initialization
        t_xi=[0,0.333,0.5,0.644,1]
        before_spline=np.array([[start[0][0],start[0][1]],
                                [xi_pred[0],xi_pred[1]],
                                [xi_pred[2],xi_pred[3]],
                                [xi_pred[4],xi_pred[5]],
                                [goal[0][0],goal[0][1]]])
        
        spline,u=interpolate.splprep([before_spline[:,0],before_spline[:,1]],u=t_xi)
    
        cgan=interpolate.splev(t,spline)
        cgan=np.concatenate((np.transpose(cgan[0]),np.transpose(cgan[1])),axis=1)

        list_xi_lin.append(lin.tolist())
        list_xi_cgan.append(cgan.tolist())
        
    n_dim=2
    n_obs=4
    collision_limit=0.1

    f1_lin  = rustmetric.f1(x_list.tolist(),list_xi_lin, n_dim,n_obs,collision_limit)
    f1_cgan = rustmetric.f1(x_list.tolist(),list_xi_cgan,n_dim,n_obs,collision_limit)

    f1_lin = 100.0*f1_lin/len(x_list)
    f1_cgan = 100.0*f1_cgan/len(x_list)
    
    return [f1_lin, f1_cgan, f1_lin-f1_cgan]
            
def test(gen,n_tests,noise_dim):
    z = np.random.uniform(0, 1, (n_tests, noise_dim))
    z = Variable(np.array(z, dtype=np.float32))
    
    x = np.random.uniform(0.0, 1.0, (n_tests,12));
    x = Variable(x.astype(np.float32))
    xi= gen(z,x)

    return collisions_measure(x.data,xi.data)
    
