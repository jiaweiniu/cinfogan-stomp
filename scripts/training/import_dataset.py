import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the data from the dataset and apply a normalization
def import_data(configuration, x_dim, xi_dim):
    n_data = configuration["n_data"]
    number_of_parameters=x_dim+xi_dim
    dataset=np.zeros((n_data,number_of_parameters))

    for m in tqdm(range(n_data)): 
        f=open(configuration["dataset_path"]+"/path_"+str(m+1)+".dat",'r')

        # x : problem
        x=np.zeros(x_dim)
        str_tmp=str.split(f.readline().strip())        
        for i in range(x_dim):
            x[i]=float(str_tmp[i])
        x=normalize_data(configuration["experiment"],x)
        dataset[m,:x_dim]=x
        
        # xi : trajectory
        str_tmp=str.split(f.readline().strip())
        for i in range(xi_dim):
            dataset[m,i+x_dim]=float(str_tmp[i])

    dataset=np.asarray(dataset).astype(np.float32)
    return dataset



def normalize_data(experiment,x):
    choice=dict({"random_left_right": normalize_data_random_left_right,
                 "table": normalize_data_table,
                 "window": normalize_data_window})
    return x

def denormalize_data(experiment,x):
    choice=dict({"random_left_right": denormalize_data_random_left_right,
                 "table":  denormalize_data_table,
                 "window": denormalize_data_window})
    return choice[experiment](x)


def normalize_data_random_left_right(x):
    x[0]=(x[0]-0.02)/(0.15-0.02)
    x[1]=(x[1]-0.02)/(0.98-0.02)
    x[2]=(x[2]-0.85)/(0.98-0.85)
    x[3]=(x[3]-0.02)/(0.98-0.02)
    x[4]=(x[4]-0.4)/(0.6-0.4)
    x[5]=(x[5]-0.02)/(0.98-0.02)
    x[6]=(x[6]-0.4)/(0.6-0.4)
    x[7]=(x[7]-0.02)/(0.98-0.02)
    x[8]=(x[8]-0.4)/(0.6-0.4)
    x[9]=(x[9]-0.02)/(0.98-0.02)
    x[10]=(x[10]-0.4)/(0.6-0.4)
    x[11]=(x[11]-0.02)/(0.98-0.02)

    return x
'''
def denormalize_data_random_left_right(x):
    x[0]=x[0]*(0.15-0.02)+0.02
    x[1]=x[1]*(0.98-0.02)+0.02
    x[2]=x[2]*(0.98-0.85)+0.85
    x[3]=x[3]*(0.98-0.02)+0.02

    for i in range(4):
        x[2*i+4]=x[2*i+4]*(0.6-0.4)+0.4
        x[2*i+5]=x[2*i+5]*(0.98-0.02)+0.02
        
    return x
'''
def denormalize_data_random_left_right(x):
    x[0]=x[0]*0.09+0.11
    x[1]=x[1]*0.09+0.11
    x[2]=x[2]*0.09+0.8
    x[3]=x[3]*0.09+0.8

    for i in range(4):
        x[2*i+4]=x[2*i+4]*0.6+0.2
        x[2*i+5]=x[2*i+5]*0.8+0.1
        
    return x


def normalize_data_table(x):
    return x

def denormalize_data_table(x):        
    return x

def normalize_data_window(x):
    return x

def denormalize_data_window(x):        
    return x

