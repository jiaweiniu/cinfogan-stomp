import numpy as np
import chainer
from chainer import training, reporter
from chainer import functions as F
from chainer import Variable
import testing_model

class GANUpdater(training.StandardUpdater):
    def __init__(self, iterator, noise_iterator, noise_dim,
                 x_dim, xi_dim,
                 optimizer_generator,
                 optimizer_discriminator, device=-1):

        iterators = {'main': iterator, 'z': noise_iterator}
        optimizers = {'gen': optimizer_generator,
                      'dis': optimizer_discriminator}
            
        super(GANUpdater,self).__init__(iterators, optimizers, device=device)
        self.epoch_counter=0
        self.noise_dim=noise_dim
        self.x_dim = x_dim
        self.xi_dim = xi_dim

    @property
    def generator(self):
        return self._optimizers['gen'].target

    @property
    def discriminator(self):
        return self._optimizers['dis'].target

    def forward(self, test=False):
        x_real_it = self._iterators['main'].next()
        x_real = self.converter(x_real_it, self.device)

        x_real_x  = x_real[:,:self.x_dim]
        x_real_xi = x_real[:,self.x_dim:]
        y_real = self.discriminator(Variable(x_real_xi),Variable(x_real_x))
        
        z_it = self._iterators['z'].next()
        z = self.converter(z_it, self.device)

        x_fake = self.generator(Variable(z),Variable(x_real_x))
        y_fake = self.discriminator(x_fake,Variable(x_real_x))

        if test:
            return x_fake
        else:
            return y_fake, y_real

    def backward(self, y):
        y_fake, y_real = y

        generator_loss = F.softmax_cross_entropy(y_fake,Variable(np.ones(y_fake.shape[0], dtype=np.int32)))
        discriminator_loss = F.softmax_cross_entropy(y_fake,Variable(np.zeros(y_fake.shape[0], dtype=np.int32)))
        discriminator_loss += F.softmax_cross_entropy(y_real,Variable(np.ones(y_real.shape[0], dtype=np.int32)))
        discriminator_loss /= 2

        return {'gen': generator_loss, 'dis': discriminator_loss}

    def update_params(self, losses, report=True):
        for name, loss in losses.items():
            if report:
                reporter.report({'{}/loss'.format(name): loss})

            self._optimizers[name].target.cleargrads()
            loss.backward()
            self._optimizers[name].update()

    def update_core(self):
        if self.is_new_epoch:
            self.epoch_counter+=1
            if self.epoch_counter%1==0:
                result=testing_model.test(self.generator,1000,self.noise_dim)
                reporter.report({'lin_ratio': result[0]})
                reporter.report({'cgan_ratio': result[2]})
                reporter.report({'diff_ratio': result[4]})
                f=open('results/f1_metric.dat','a')
                f.write(str(result[0])+" "+str(result[1])+" "+str(result[2])+" "+str(result[3])+str(result[4])+"\n")
                f.close()
                
                self.epoch_counter=0
            pass

        losses = self.backward(self.forward())
        self.update_params(losses, report=True)


class WassersteinGANUpdater(training.StandardUpdater):
    def __init__(self, iterator, noise_iterator, noise_dim, optimizer_generator,
                 optimizer_critic, device=-1):

        iterators = {'main': iterator, 'z': noise_iterator}
        optimizers = {'gen': optimizer_generator,
                      'cri': optimizer_critic}
            
        super(WassersteinGANUpdater,self).__init__(iterators, optimizers, device=device)
        self.noise_dim=noise_dim
        self.epoch_counter=0
        
    @property
    def generator(self):
        return self._optimizers['gen'].target

    @property
    def critic(self):
        return self._optimizers['cri'].target

    def forward(self,test=False):
        z_it = self._iterators['z'].next()
        z = self.converter(z_it, self.device)

        x_fake = self.generator(Variable(z))
        
        return x_fake

    def backward(self, y):
        y_fake, y_real = y

        generator_loss = F.softmax_cross_entropy(y_fake,Variable(np.ones(y_fake.shape[0], dtype=np.int32)))
        discriminator_loss = F.softmax_cross_entropy(y_fake,Variable(np.zeros(y_fake.shape[0], dtype=np.int32)))
        discriminator_loss += F.softmax_cross_entropy(y_real,Variable(np.ones(y_real.shape[0], dtype=np.int32)))
        discriminator_loss /= 2

        return {'gen': generator_loss, 'cri': discriminator_loss}

    def update_params(self, losses, report=True):
        for name, loss in losses.items():
            if report:
                reporter.report({'{}/loss'.format(name): loss})

            self._optimizers[name].target.cleargrads()
            loss.backward()
            self._optimizers[name].update()

    def update_core(self):
        if self.is_new_epoch:
            pass

        def _update(optimizer, loss):
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

        # Update critic 5 times
        for _ in range(5):
            # Clamp critic parameters
            self.critic.clamp()
            
            # Real images
            x_real_it = self._iterators['main'].next()
            x_real = self.converter(x_real_it, self.device)

            
            x_real_x=x_real[:,0:12]
            x_real_xi=x_real[:,12:18]
            y_real = self.critic(Variable(x_real_x),Variable(x_real_xi))
        
            y_real.grad = np.ones_like(y_real.data)
            _update(self._optimizers['cri'], y_real)

            
            # Fake images
            z_it = self._iterators['z'].next()
            z = self.converter(z_it, self.device)

            c=Variable(x_real_x)
            x_fake = self.generator(Variable(z),c)
            y_fake = self.critic(x_fake,c)
            y_fake.grad = -1 * np.ones_like(y_fake.data)
            _update(self._optimizers['cri'], y_fake)
            
            reporter.report({
                'cri/loss/real': F.sum(y_real)/y_real.shape[0],
                'cri/loss/fake': F.sum(y_fake)/y_fake.shape[0],
                'cri/loss':      F.sum(y_real - y_fake)/y_real.shape[0]
            })

        # Update generator 1 time
        z_it = self._iterators['z'].next()
        z = self.converter(z_it, self.device)

        c=Variable(x_real_x)
        x_fake = self.generator(z,c)
        y_fake = self.critic(x_fake,c)
        y_fake.grad = np.ones_like(y_fake.data)
        _update(self._optimizers['gen'], y_fake)
        
        reporter.report({'gen/loss': F.sum(y_fake)/y_fake.shape[0]})

        self.epoch_counter+=1
        if self.epoch_counter%66==0:
            result=testing_model.test(self.generator,1000,self.noise_dim)
            reporter.report({'cgan_improvement': result[0]})

            self.epoch_counter=0
