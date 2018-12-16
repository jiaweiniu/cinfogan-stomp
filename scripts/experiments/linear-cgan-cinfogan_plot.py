import pandas as pd
import sys
import numpy as np
import csv
from pandas.core.frame import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

f = open('../training/results/f_metric.dat',"r")
list_of_lines = []
line = f.readline()
n_epochs = 200

while(line):
    list_of_lines.append(line)
    line = f.readline()
f.close


list_of_lines = [i.split() for i in list_of_lines]

v=[]
for i in range(len(list_of_lines)):
    v.append([i-n_epochs*(i//n_epochs), 'Linear',   float(list_of_lines[i][0])])
    v.append([i-n_epochs*(i//n_epochs), 'CGAN',     float(list_of_lines[i][1])])
    v.append([i-n_epochs*(i//n_epochs), 'CInfoGAN', float(list_of_lines[i][2])])


data = DataFrame(v)
data.columns = ['epoch','method','F measure(%)']

sns.lineplot(x="epoch",y="F measure(%)",
             hue="method",data=data)
plt.show()
