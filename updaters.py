import numpy as np
import chainer
from chainer import training, reporter, serializers
from chainer import functions as F
from chainer import Variable
import testing_model

#setting continous
def rnd_continuous(n, n_continuous, mu=0, std=1):
    return np.random.normal(mu, std, size=(n, n_continuous))


class GANUpdater(training.StandardUpdater):
    def __init__(self, iterator, noise_iterator, noise_dim,
                 continuous_dim,
                 x_dim, xi_dim, experiment,
                 optimizer_generator,
                 optimizer_discriminator, device=-1):

        iterators = {'main': iterator, 'z': noise_iterator}
        optimizers = {'gen': optimizer_generator,
                      'dis': optimizer_discriminator}
            
        super(GANUpdater,self).__init__(iterators, optimizers, device=device)
        self.epoch_counter=0
        self.noise_dim=noise_dim
        self.continuous_dim=continuous_dim
        self.x_dim = x_dim
        self.xi_dim = xi_dim
        self.experiment = experiment
        
    @property
    def generator(self):
        return self._optimizers['gen'].target

    @property
    def discriminator(self):
        return self._optimizers['dis'].target

    def forward(self, test=False):
        x_real_it = self._iterators['main'].next()
        x_real = self.converter(x_real_it, self.device)

        x_real_x = x_real[:,:self.x_dim]

        x_real_xi = x_real[:,self.x_dim:]
        y_real, _ = self.discriminator(Variable(x_real_xi),Variable(x_real_x))
        z_it = self._iterators['z'].next()
        z = self.converter(z_it, self.device)

        x_fake = self.generator(Variable(z),Variable(x_real_x))
        y_fake, mi = self.discriminator(x_fake,Variable(x_real_x))

        #print("y fake")
        #print(y_fake)

        if test:
            return x_fake
        else:
            return y_fake, mi, y_real

    def backward(self, y_fake, mi, y_real):
        
        generator_loss = F.softmax_cross_entropy(y_fake,Variable(np.ones(y_fake.shape[0], dtype=np.int32)))
        discriminator_loss = F.softmax_cross_entropy(y_fake,Variable(np.zeros(y_fake.shape[0], dtype=np.int32)))
        discriminator_loss += F.softmax_cross_entropy(y_real,Variable(np.ones(y_real.shape[0], dtype=np.int32)))
        discriminator_loss /= 2                

        mi = mi.data
        #print("mi:")
        #print(mi.shape)

        # Mutual Information loss
        # Sample continuous codes to learn rotation, thickness, etc.
        c_continuous = np.asarray(rnd_continuous(mi.shape[0], mi.shape[1]), dtype=np.float32)

        # Continuous loss - Fix standard deviation to 1, i.e. log variance is 0
        mi_continuous_ln_var = np.empty_like(mi, dtype=np.float32)
        mi_continuous_ln_var.fill(1)
        # mi_continuous_ln_var.fill(1e-6)
        continuous_loss = F.gaussian_nll(mi, Variable(c_continuous), Variable(mi_continuous_ln_var))
        continuous_loss /= mi.shape[0]

        #print("continuous_loss:")
        #print(continuous_loss)

        #generator_loss += categorical_loss
        generator_loss += continuous_loss

        return {'gen': generator_loss, 'dis': discriminator_loss}

    def update_params(self, losses, report=True):
        for name, loss in losses.items():
            if report:
                reporter.report({'{}/loss'.format(name): loss})

            self._optimizers[name].target.cleargrads()
            loss.backward()
            self._optimizers[name].update()

    def update_core(self):
        if self.epoch==self.epoch_counter:
            self.epoch_counter+=1
            if self.experiment=="random_left_right":
                #print("update core!!")
                result=testing_model.test(self.generator,50000,self.noise_dim, self.continuous_dim)
                serializers.save_npz("results/models/tmp/"+str(self.epoch_counter-1)+"_gen.model",self.generator)
                reporter.report({'lin_ratio': result[0]})
                reporter.report({'infogan_ratio': result[1]})
                reporter.report({'diff_ratio': result[2]})
                f=open('results/f1_metric.dat','a')
                f.write(str(result[0])+" "+str(result[1])+" "+str(result[2])+"\n")
                f.close()
                
            pass

        y_fake, mi, y_real = self.forward()
        losses = self.backward(y_fake, mi, y_real)
        self.update_params(losses, report=True)


class WassersteinGANUpdater(training.StandardUpdater):
    def __init__(self, iterator, noise_iterator, noise_dim,
                 x_dim, xi_dim, experiment,
                 optimizer_generator,
                 optimizer_critic, device=-1):
        
        iterators = {'main': iterator, 'z': noise_iterator}
        optimizers = {'gen': optimizer_generator,
                      'cri': optimizer_critic}
            
        super(WassersteinGANUpdater,self).__init__(iterators, optimizers, device=device)
        self.noise_dim=noise_dim
        self.epoch_counter=0
        self.x_dim = x_dim
        self.xi_dim = xi_dim
        self.experiment = experiment        

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
        if self.epoch==self.epoch_counter:
            self.epoch_counter+=1
            if self.experiment=="random_left_right":
                result=testing_model.test(self.generator,50,self.noise_dim)
                serializers.save_npz("results/models/tmp/"+str(self.epoch_counter-1)+"_gen.model",self.generator)
                reporter.report({'lin_ratio': result[0]})
                reporter.report({'infogan_ratio': result[1]})
                reporter.report({'diff_ratio': result[2]})
                f=open('results/f1_metric.dat','a')
                f.write(str(result[0])+" "+str(result[1])+" "+str(result[2])+"\n")
                f.close()

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

            
            x_real_x=x_real[:,:self.x_dim]
            x_real_xi=x_real[:,self.x_dim:]
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

