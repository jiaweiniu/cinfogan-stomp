import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()

axes = plt.gca()
axes.set_ylim([0,100])

ratio_lin=[]
ratio_cgan=[]
measure_lin=[]
measure_cgan=[]
t=np.linspace(2,200,199)


f=open("results.dat",'r')
for i in range(199):
    str_tmp=str.split(f.readline().strip())
    ratio_lin.append(float(str_tmp[0]))
    measure_lin.append(float(str_tmp[1]))
    ratio_cgan.append(float(str_tmp[2]))
    measure_cgan.append(float(str_tmp[3]))

    
plt.plot(t,ratio_lin,label='Linear')
plt.plot(t,ratio_cgan,label='CGAN')

plt.xlabel("Epoch")
plt.ylabel("F1 measure")

"""
plt.plot(t,measure_lin,label='Linear')
plt.plot(t,measure_cgan,label='CGAN')

plt.xlabel("Epoch")
plt.ylabel("F2 measure")
"""

plt.legend(loc='lower right')
plt.show()
