import chainer
from chainer import Chain
from chainer import functions as F
from chainer import links as L


class Generator(Chain):
    def __init__(self, x_dim, xi_dim, noise_dim):
        super(Generator, self).__init__(
            l1=L.Linear(x_dim+noise_dim, 100),
            l2=L.Linear(None, 100),
            l3=L.Linear(None, xi_dim),
            bn_l1=L.BatchNormalization(100),
            bn_l2=L.BatchNormalization(100),
        )
        
    def __call__(self, z, y):
        h = F.concat([z,y])
        # I deleted the batch normalization because if we normalize how can
        # the generator understand the exact problem ?
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(z))
        h = F.sigmoid(self.l3(h))
        return h


class Discriminator(Chain):
    def __init__(self, x_dim, xi_dim):
        super(Discriminator, self).__init__(
            l1=L.Linear(x_dim+xi_dim, 512),
            l2=L.Linear(None, 2),
            )

    def __call__(self, x,y):
        h = F.concat([x,y])
        h = F.relu(self.l1(h))
        h = self.l2(h)
        return h

class Critic(Chain):
    def __init__(self, x_dim, xi_dim):
        super(Critic, self).__init__(
            l1=L.Linear(x_dim+xi_dim, 100), # we take the two axis and repeat the information
            l2=L.Linear(None, 100),
            l3=L.Linear(None, 1),
            bn_l1=L.BatchNormalization(100),
            bn_l2=L.BatchNormalization(100),
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
