import os
from chainer import training, cuda, Variable, serializers
from chainer.training import extension
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import import_dataset
import matplotlib.patches as patches

fig,ax = plt.subplots(1)

start=[0.03,0.05]
goal=[0.92,0.96]

circle = patches.Circle((0.7,0.6), radius=0.1, color='red',alpha=0.7)
ax.add_patch(circle)

circle = patches.Circle((0.6,0.2), radius=0.1, color='red',alpha=0.7)
ax.add_patch(circle)

circle = patches.Circle((0.4,0.4), radius=0.1, color='red',alpha=0.7)
ax.add_patch(circle)

circle = patches.Circle((0.3,0.8), radius=0.1, color='red',alpha=0.7)
ax.add_patch(circle)

rect = patches.Rectangle((0.01,0.01),0.19,0.19,edgecolor='green',facecolor='none')
ax.add_patch(rect)

rect = patches.Rectangle((0.8,0.8),0.19,0.19,edgecolor='green',facecolor='none')
ax.add_patch(rect)

rect = patches.Rectangle((0.2,0.1),0.6,0.8,edgecolor='red',facecolor='none')
ax.add_patch(rect)


ax.scatter(start[0],start[1],marker="s",s=30,color="black",zorder=20)
ax.scatter(goal[0],goal[1],marker="*",s=30,color="black",zorder=21)
ax.set_xlim((0,1))
ax.set_ylim((0,1))
ax.set_aspect('equal')
plt.show()
