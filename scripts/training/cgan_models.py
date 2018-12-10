import chainer
from chainer import Chain
from chainer import functions as F
from chainer import links as L

class Cgan_Generator(Chain):
    def __init__(self, n_z, x_dim, xi_dim, n_neurons_gen):
        super(Cgan_Generator, self).__init__(
            l1=L.Linear(x_dim+n_z, n_neurons_gen),
            l2=L.Linear(None, n_neurons_gen),
            l3=L.Linear(None, xi_dim),
        )
        
    def __call__(self, z, y):
        h = F.concat([z,y])
        ''' I deleted the batch normalization because if we normalize how can
         the generator understand the exact problem ?
        print("dim z: "+str(z.shape)) '''

        h = F.relu(self.l1(h))
        
        h = F.relu(self.l2(z))
        h = F.sigmoid(self.l3(h))
        return h


class Cgan_Discriminator(Chain):
    def __init__(self, x_dim, xi_dim, n_neurons_dis):
        super(Cgan_Discriminator, self).__init__(
            l1=L.Linear(x_dim+xi_dim, n_neurons_dis),
            l2=L.Linear(None, n_neurons_dis),
            fc_d=L.Linear(None, 2),           
        )

    def __call__(self, x, y):
        h = F.concat([x,y])
        h = F.relu(self.l1(h))
        h = self.l2(h)
        # Real/Fake prediction
        d = self.fc_d(h)

        return d

class Cgan_Critic(Chain):
    def __init__(self, x_dim, xi_dim, n_neurons_cri):
        super(Cgan_Critic, self).__init__(
            l1=L.Linear(x_dim+xi_dim, n_neurons_cri),
            l2=L.Linear(None, n_neurons_cri),
            l3=L.Linear(None, 1),
            bn_l1=L.BatchNormalization(n_neurons_cri),
            bn_l2=L.BatchNormalization(n_neurons_cri),
            )
        
    def clamp(self, lower=-0.01, upper=0.01):
        for params in self.params():
            params_clipped = F.clip(params, lower, upper)
            params.data = params_clipped.data
            
    def __call__(self, xi, x):
        h = F.concat([xi,x])
        h = F.leaky_relu(self.l1(h))
        h = F.leaky_relu(self.l2(h))
        h = self.l3(h)
        return h
       
    
