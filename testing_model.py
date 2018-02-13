import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import interpolate

from chainer import serializers, Variable

import import_dataset

# Measure the percentage of collision of the trajectory and compare it to the linear initialization
def collisions_measure(x_list,xi_list):
    measure_lin_col=[]
    ratio_lin_col=[]
    measure_cgan_col=[]
    ratio_cgan_col=[]
    
    for x,xi in zip(x_list,xi_list):
        t=np.linspace(0,1,100)
        t=np.asarray([t])

        x=import_dataset.denormalize_data_random_left_right(x)
        start=np.asarray([x[0:2]])
        goal=np.asarray([x[2:4]])
        
        # linear initialization
        lin=np.dot(1-np.transpose(t),start)+np.dot(np.transpose(t),goal)

        # cgan initialization
        t_xi=[0,0.333,0.5,0.644,1]
        before_spline=np.array([[start[0][0],start[0][1]],[xi[0],xi[1]],[xi[2],xi[3]],[xi[4],xi[5]],[goal[0][0],goal[0][1]]])
        
        spline,u=interpolate.splprep([before_spline[:,0],before_spline[:,1]],u=t_xi)
    
        cgan=interpolate.splev(t,spline)
        cgan=np.concatenate((np.transpose(cgan[0]),np.transpose(cgan[1])),axis=1)

        obstacles=[ [x[4],x[5]], [x[6],x[7]], [x[8],x[9]], [x[10],x[11]] ]
    
        measure_lin=0
        lin_col_bool=False
        for p in lin:
            point_in_obs=False
            for o in obstacles:
                if np.linalg.norm(o-p)<0.1:
                    point_in_obs=True
            if(point_in_obs):
                measure_lin+=1
                lin_col_bool=True

        if(lin_col_bool):
            ratio_lin_col.append(1)
        else:
            ratio_lin_col.append(0)
        measure_lin_col.append(measure_lin)

        measure_cgan=0
        cgan_col_bool=False
        for p in cgan:
            point_in_obs=False
            for o in obstacles:
                if np.linalg.norm(o-p)<0.1:
                    point_in_obs=True
            if(point_in_obs):
                measure_cgan+=1
                cgan_col_bool=True

        if(cgan_col_bool):
            ratio_cgan_col.append(1)
        else:
            ratio_cgan_col.append(0)
        measure_cgan_col.append(measure_cgan)

    return [100*np.mean(ratio_lin_col),np.mean(measure_lin_col),
            100*np.mean(ratio_cgan_col),np.mean(measure_cgan_col),
            100*(np.mean(ratio_lin_col)-np.mean(ratio_cgan_col))]
            
def test(gen,n_tests,noise_dim):
    z = np.random.uniform(0, 1, (n_tests, noise_dim))
    z = Variable(np.array(z, dtype=np.float32))
    
    x = np.random.uniform(0.0, 1.0, (n_tests,12));
    x = Variable(x.astype(np.float32))
    xi= gen(z,x)

    return collisions_measure(x.data,xi.data)
    
