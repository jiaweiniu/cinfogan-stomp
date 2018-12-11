import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import interpolate
from cgan_models import Cgan_Generator   # cgan model
from models import Generator             # infogan model
from chainer import serializers, Variable, reporter

import import_dataset
import rustmetric

# Measure the percentage of collision of the trajectory and compare it to the linear initialization

def collisions_measure(x_list,cgan_xi_pred_list,infogan_xi_pred_list):
    measure_lin_col=[]
    ratio_lin_col=[]

    measure_infogan_col=[]
    ratio_infogan_col=[]

    measure_cgan_col=[]
    ratio_cgan_col=[]

    list_xi_lin=[]
    list_xi_infogan=[]
    list_xi_cgan=[]

    for x,cgan_xi_pred,infogan_xi_pred in zip(x_list,cgan_xi_pred_list,infogan_xi_pred_list):
        t=np.linspace(0,1,100)
        t=np.asarray([t])
        # import dataset from random left right
        x=import_dataset.denormalize_data_random_left_right(x) 

        start=np.asarray([x[0:2]]) # give the position of start 
        goal=np.asarray([x[2:4]])  # give the position of goal
       
        # linear initialization
        lin=np.dot(1-np.transpose(t),start)+np.dot(np.transpose(t),goal)  # zhuan(3)zhi(4)

        # infogan initialization
        t_xi=[0,0.333,0.5,0.644,1]
        before_spline=np.array([[start[0][0],start[0][1]],
                                [infogan_xi_pred[0],infogan_xi_pred[1]],
                                [infogan_xi_pred[2],infogan_xi_pred[3]],
                                [infogan_xi_pred[4],infogan_xi_pred[5]],
                                [goal[0][0],goal[0][1]]])     # become an array
        
        spline,u=interpolate.splprep([before_spline[:,0],before_spline[:,1]],u=t_xi)  # cha(1)zhi(2)
    
        infogan=interpolate.splev(t,spline)
        infogan=np.concatenate((np.transpose(infogan[0]),np.transpose(infogan[1])),axis=1)

        list_xi_lin.append(lin.tolist())
        list_xi_infogan.append(infogan.tolist())
       

        # cgan initialization

        t_xi=[0,0.333,0.5,0.644,1]
        before_spline=np.array([[start[0][0],start[0][1]],
                                [cgan_xi_pred[0],cgan_xi_pred[1]],
                                [cgan_xi_pred[2],cgan_xi_pred[3]],
                                [cgan_xi_pred[4],cgan_xi_pred[5]],
                                [goal[0][0],goal[0][1]]])     # become an array
        
        spline,u=interpolate.splprep([before_spline[:,0],before_spline[:,1]],u=t_xi)  # cha(1)zhi(2)
    
        cgan=interpolate.splev(t,spline)
        cgan=np.concatenate((np.transpose(cgan[0]),np.transpose(cgan[1])),axis=1)

        list_xi_cgan.append(cgan.tolist())
        
    n_dim=2
    n_obs=4
    collision_limit=0.1

    f1_lin  = rustmetric.f1(x_list.tolist(),list_xi_lin, n_dim,n_obs,collision_limit)
    f1_infogan = rustmetric.f1(x_list.tolist(),list_xi_infogan,n_dim,n_obs,collision_limit)
    f1_cgan = rustmetric.f1(x_list.tolist(),list_xi_cgan,n_dim,n_obs,collision_limit)

    f1_lin = 100.0*f1_lin/len(x_list)
    f1_infogan = 100.0*f1_infogan/len(x_list)
    f1_cgan = 100.0*f1_cgan/len(x_list)

    return [f1_lin, f1_cgan, f1_infogan]
            
def test(gen_infogan,gen_cgan,n_tests,n_z_infogan, n_z_cgan, n_continuous):
    # infogan
    z_infogan = np.random.uniform(-2, 2, (n_tests, n_z_infogan+n_continuous))
    z_infogan = Variable(np.array(z_infogan, dtype=np.float32))

    # cgan
    z_cgan = np.random.uniform(-2, 2, (n_tests, n_z_cgan))
    z_cgan = Variable(np.array(z_cgan, dtype=np.float32))
    
    x = np.random.uniform(0.0, 1.0, (n_tests,12));
    x = Variable(x.astype(np.float32))
    infogan_xi= gen_infogan(z_infogan,x)
    cgan_xi = gen_cgan(z_cgan,x)
    return collisions_measure(x.data,cgan_xi.data,infogan_xi.data)

    

if __name__ == '__main__':
    for i in range(10):
        for n in range(200):
            gen_infogan=Generator(60, 12, 6, 2, 70)
            gen_cgan=Cgan_Generator(62, 12, 6, 70)
            serializers.load_npz("cinfogan_results/models_"+str(i)+"/"+str(n)+"_gen.model", gen_infogan)
            serializers.load_npz("cgan_results/models_"+str(i)+"/"+str(n)+"_gen.model", gen_cgan)
            result = (test(gen_infogan,gen_cgan,20000,60,62,2))
            print(result)
            f=open('results/f_metric.dat','a')
            f.write(str(result[0])+" "+str(result[1])+" "+str(result[2])+"\n")
            f.close()
                    
