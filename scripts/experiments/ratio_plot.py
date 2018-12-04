import numpy as np
import matplotlib.pyplot as plt
'''
    plot avoiding of obstacles in linear and cinfogan
'''
fig = plt.figure()

axes = plt.gca()
axes.set_xlim([0,200])
#axes.set_ylim([0.100])

ratio_lin=[]
ratio_infogan=[]
measure_lin=[]
measure_infogan=[]
n_lines=0

with open("../training/results/f_metric.dat",'r') as f:
    for i, l in enumerate(f):
        str_tmp=str.split(l.strip())
        ratio_lin.append(float(str_tmp[0]))
        ratio_infogan.append(float(str_tmp[1]))
    n_lines=i+1

    
t=np.linspace(2,n_lines+1,n_lines)

plt.plot(t,ratio_lin,label='linear')
plt.plot(t,ratio_infogan,label='CInfoGAN')

plt.xlabel("Epoch")
plt.ylabel("F1 measure(%)")

plt.legend(loc='lower right')
plt.show()
