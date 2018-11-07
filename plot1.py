import os
from chainer import training, cuda, Variable, serializers
from chainer.training import extension
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import import_dataset
import matplotlib.patches as patches

def __call__(self, trainer):
        dirname = os.path.join(trainer.out, self._dirname)
        if (trainer.updater.epoch%1==0):            
            n_gen=1
            #z=Variable(np.random.uniform(-1,1,(n_gen,self.noise_dim)).astype(np.float32))
            z=Variable(np.random.uniform(-2,2,(n_gen,self.noise_dim+self.n_continuous)).astype(np.float32))

            dataset=import_dataset.import_data(self.configuration,self.x_dim,self.xi_dim)
            x=Variable(dataset[1:2,:self.x_dim])

            fig,ax = plt.subplots(1)
            
            x=x.data[0]
            x=import_dataset.denormalize_data_random_left_right(x)
            
            start=[x[0],x[1]]
            goal=[x[2],x[3]]
            
            circle = patches.Circle((x[4],x[5]), radius=0.1, color='red',alpha=0.7)
            ax.add_patch(circle)
            circle = patches.Circle((x[6],x[7]), radius=0.1, color='red',alpha=0.7)
            ax.add_patch(circle)
            circle = patches.Circle((x[8],x[9]), radius=0.1, color='red',alpha=0.7)
            ax.add_patch(circle)
            circle = patches.Circle((x[11],x[10]), radius=0.1, color='red',alpha=0.7)
            ax.add_patch(circle)

            #ax.plot([start[0],xi_gen[+0],xi_gen[2],xi_gen[4],goal[0]],[start[1],xi_gen[1],xi_gen[3],xi_gen[5],goal[1]],marker='*')
            
            ax.scatter(start[0],start[1],marker="s",s=130,color="black",zorder=20)

            ax.scatter(goal[0],goal[1],marker="*",s=130,color="black",zorder=21)

            ax.set_xlim((0,1))
            ax.set_ylim((0,1))
            ax.set_aspect('equal')
            plt.show()

