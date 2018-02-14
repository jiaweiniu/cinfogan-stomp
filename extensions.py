import os
from chainer import training, cuda, Variable, serializers
from chainer.training import extension
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import import_dataset
import matplotlib.patches as patches

class GeneratorSample(extension.Extension):
    def __init__(self, dataset_path, x_dim, xi_dim,
                 noise_dim, dirname='samples', sample_format='png'):
        self._dirname = dirname
        self._sample_format = sample_format
        self.noise_dim=noise_dim
        self.x_dim=x_dim
        self.xi_dim=xi_dim
        self.dataset_path=dataset_path
        
    def __call__(self, trainer):
        dirname = os.path.join(trainer.out, self._dirname)
        if (trainer.updater.epoch%20==0):            
            n_gen=1
            z=Variable(np.random.uniform(-1,1,(n_gen,self.noise_dim)).astype(np.float32))

            dataset=import_dataset.import_data(self.dataset_path,3,self.x_dim,self.xi_dim)
            x=Variable(dataset[1:2,:self.x_dim])

            xi_gen = trainer.updater.generator(z,x).data[0]
            
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

            ax.plot([start[0],xi_gen[+0],xi_gen[2],xi_gen[4],goal[0]],[start[1],xi_gen[1],xi_gen[3],xi_gen[5],goal[1]],marker='*')
            
            ax.scatter(start[0],start[1],marker="s",s=130,color="black",zorder=20)

            ax.scatter(goal[0],goal[1],marker="*",s=130,color="black",zorder=21)

            ax.set_xlim((0,1))
            ax.set_ylim((0,1))
            ax.set_aspect('equal')


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

"""
@training.make_extension(trigger=(1, 'epoch'))
def sample_ims(trainer):
    x = trainer.updater.forward(test)
    x = x.data
    print(x)
    filename = 'result/sample/{}.png'.format(trainer.updater.epoch)
    plot.save_ims(filename, x)
"""
