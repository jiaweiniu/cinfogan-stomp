import pandas as pd
import sys
import numpy as np
import csv
from pandas.core.frame import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

f = open('../training/results/1.dat',"r")
lists = []
line = f.readline()

while(line):
    lists.append(line)
    line = f.readline()
f.close

lists = [i.split() for i in lists]
v = ["epoch","method","F measure(%)"]

for i in range (len(lists)):
    v.append([i,'Linear',lists[i][0]])
    v.append([i,'CGAN',lists[i][1]])
    v.append([i,'CInfoGAN',lists[i][2]])

newl = []
for i in range (3,len(v)):
    newl.append([int(v[i][0]),str(v[i][1]),float(v[i][2])])

for i in range (600,1200):
    newl[i][0] = newl[i][0] - 200

for i in range (1200,1800):
    newl[i][0] = newl[i][0] - 400

for i in range (1800,2400):
    newl[i][0] = newl[i][0] - 600

for i in range (2400,3000):
    newl[i][0] = newl[i][0] - 800

for i in range (3000,3600):
    newl[i][0] = newl[i][0] - 1000

#name = ["epoch","method","acc"]
name = ["epoch"," ","acc"]

data = DataFrame(newl)
data.columns = ['epoch','method','F measure(%)']


print (data)

sns.lineplot(x="epoch",y="F measure(%)",
             hue="method",data=data)
plt.show()
