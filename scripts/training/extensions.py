import os
from chainer import training, cuda, Variable, serializers
from chainer.training import extension
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import import_dataset
import matplotlib.patches as patches


class GeneratorSample(extension.Extension):
    def __init__(self, configuration, x_dim, xi_dim, n_continuous,
                 noise_dim, dirname='samples', sample_format='png'):
        self._dirname = dirname
        self._sample_format = sample_format
        self.noise_dim=noise_dim
        self.n_continuous=n_continuous
        self.x_dim=x_dim
        self.xi_dim=xi_dim
        self.configuration=configuration
    def __call__(self, trainer):
        dirname = os.path.join(trainer.out, self._dirname)
        if (trainer.updater.epoch%1==0):            
            n_gen=1
            z=Variable(np.random.uniform(-1,1,(n_gen,self.noise_dim+self.n_continuous)).astype(np.float32))
            dataset=import_dataset.import_data(self.configuration,self.x_dim,self.xi_dim)
            x=Variable(dataset[1:2,:self.x_dim])

            xi_gen = trainer.updater.generator(z,x).data[0]
            fig,ax = plt.subplots(1)
            x=x.data[0]
            x=import_dataset.denormalize_data_random_left_right(x)
            start=[0.1,0.16]
            goal=[0.76,0.88]
            
            circle = patches.Circle((0.5,0.78), radius=0.1, color='red',alpha=0.6)
            ax.add_patch(circle)
            circle = patches.Circle((0.6,0.5), radius=0.1, color='red',alpha=0.6)
            ax.add_patch(circle)
            circle = patches.Circle((0.3,0.3), radius=0.1, color='red',alpha=0.6)
            ax.add_patch(circle)

            ''' # when do not have precise number                        
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
            '''
            ax.plot([start[0],xi_gen[0],xi_gen[2],xi_gen[4],goal[0]],[start[1],xi_gen[1],xi_gen[3],xi_gen[5],goal[1]],marker='*')
            
            ax.scatter(start[0],start[1],marker="s",s=130,color="black",zorder=20)

            ax.scatter(goal[0],goal[1],marker="*",s=130,color="black",zorder=21)

            ax.set_xlim((0,1))
            ax.set_ylim((0,1))
            ax.set_aspect('equal')

            # TODO add verbose option
            """
            if (trainer.updater.epoch == 2):    
                print("Initial trajectory point")
                print("generated point1")
                print(xi_gen[0],xi_gen[1])
                print("generated point2")
                print(xi_gen[2],xi_gen[3])
                print("generated point3")
                print(xi_gen[4],xi_gen[5])
            """
            '''# plot 5 generated trajectories
            for _ in range(1,5):
                x=Variable(dataset[136:137,:self.x_dim])
                z=Variable(np.random.uniform(-1,1,(n_gen,self.noise_dim+self.n_continuous)).astype(np.float32))
                xi_gen = trainer.updater.generator(z,x).data[0]
                ax.plot([start[0],xi_gen[0],xi_gen[2],xi_gen[4],goal[0]],[start[1],xi_gen[1],xi_gen[3],xi_gen[5],goal[1]],marker='*')
            '''  

            filename = '{}.{}'.format(trainer.updater.epoch,
                                      self._sample_format)
            filename = os.path.join(dirname, filename)
        
            plt.savefig(filename)
            plt.close()
            serializers.save_npz("results/models/"+str(trainer.updater.epoch)+".model",trainer.updater.generator)
            
      
    def sample(self, trainer):
        x = trainer.updater.forward(test=True)
        x = x.data
        return x


    
