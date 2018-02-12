import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()

axes = plt.gca()
axes.set_ylim([0,100])

ratio_lin=[]
ratio_cgan=[]
measure_lin=[]
measure_cgan=[]
n_lines=0

with open("results/f1_metric.dat",'r') as f:
    for i, l in enumerate(f):
        str_tmp=str.split(l.strip())
        ratio_lin.append(float(str_tmp[0]))
        measure_lin.append(float(str_tmp[1]))
        ratio_cgan.append(float(str_tmp[2]))
        measure_cgan.append(float(str_tmp[3]))
    n_lines=i+1

    
t=np.linspace(2,n_lines+1,n_lines)

plt.plot(t,ratio_lin,label='Linear')
plt.plot(t,ratio_cgan,label='CGAN')

plt.xlabel("Epoch")
plt.ylabel("F1 measure")

plt.legend(loc='lower right')
plt.show()
