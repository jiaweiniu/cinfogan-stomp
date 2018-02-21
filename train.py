import os
import json
import ast
import argparse
from chainer import Variable,datasets, training, iterators, optimizers, serializers
from chainer.training import updater, extensions
from iterators import RandomNoiseIterator, UniformNoiseGenerator
from models import Generator, Discriminator, Critic
from updaters import GANUpdater, WassersteinGANUpdater
from extensions import GeneratorSample
import numpy as np
import import_dataset

iterators.RandomNoiseIterator = RandomNoiseIterator
updater.GANUpdater = GANUpdater
extensions.GeneratorSample = GeneratorSample

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=15)
    parser.add_argument('--noisedim', type=int, default=20)
    parser.add_argument("--wasserstein", action="store_true")
    
    return parser.parse_args()


if __name__ == '__main__':
    with open(os.path.join("conf.json")) as fd:
        json_data = json.load(fd)
    configuration=json_data
    
    batch_size = configuration["n_batchsize"]
    epochs     = configuration["n_epochs"]
    noise_dim  = configuration["n_noisedim"]
    experiment = configuration["experiment"]
    n_neurons_gen  = configuration["n_neurons_gen"]
    n_neurons_dis  = configuration["n_neurons_dis"]
    
    args = parse_args()
    gpu = args.gpu

    if experiment == "random_left_right":
        x_dim=12
        xi_dim=6
    else:
        x_dim=14
        xi_dim=7
   
    train = import_dataset.import_data(configuration, x_dim, xi_dim)    
    train_iter = iterators.SerialIterator(train, batch_size)
    z_iter = iterators.RandomNoiseIterator(UniformNoiseGenerator(-1, 1, noise_dim), batch_size)

    # Creating the Neural Networks models
    gen=Generator(x_dim, xi_dim, noise_dim,n_neurons_gen)
    dis=Discriminator(x_dim, xi_dim,n_neurons_dis)
    critic=Critic(x_dim, xi_dim)
    
    if configuration["wasserstein"]:
        print("Using Wasserstein")
        optimizer_generator = optimizers.RMSprop(lr=0.00005)
        optimizer_critic = optimizers.RMSprop(lr=0.00005)

        a=np.zeros((1,xi_dim)).astype(np.float32)
        b=np.zeros((1,x_dim)).astype(np.float32)
        critic(Variable(a),Variable(b))
        
        optimizer_generator.setup(gen)
        optimizer_critic.setup(critic)

        updater = WassersteinGANUpdater(
            iterator=train_iter,
            noise_iterator=z_iter,
            noise_dim=noise_dim,
            optimizer_generator=optimizer_generator,
            optimizer_critic=optimizer_critic,
            device=gpu,
        )
    else:
        print("Not using Wasserstein")
        optimizer_generator = optimizers.Adam()
        optimizer_discriminator = optimizers.SGD()
        
        optimizer_generator.setup(gen)
        optimizer_discriminator.setup(dis)

        updater = GANUpdater(
            iterator=train_iter,
            noise_iterator=z_iter,
            noise_dim=noise_dim,
            x_dim=x_dim,
            xi_dim=xi_dim,
            experiment=configuration["experiment"],
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            device=gpu,
        )

    trainer = training.Trainer(updater, stop_trigger=(epochs, 'epoch'))
    trainer.out="results" # changing the name because we do multiple experiments
    trainer.extend(extensions.LogReport())

    
    if configuration["wasserstein"]:        
        print_report_args = ['epoch', 'gen/loss', 'cri/loss',
                             'lin_ratio','cgan_ratio','diff_ratio']
    else:
        print_report_args = ['epoch', 'gen/loss', 'dis/loss',
                             'lin_ratio','cgan_ratio','diff_ratio']

    trainer.extend(extensions.PrintReport(print_report_args))
    trainer.extend(extensions.ProgressBar())
    if configuration["experiment"] == "random_left_right":
        trainer.extend(extensions.GeneratorSample(configuration, x_dim,
                                                  xi_dim,noise_dim), trigger=(1, 'epoch'))


    # We delete the f1_metric.dat file to be sure we do not mixed multiple experiment data.
    cmd = "touch results/f1_metric.dat && rm results/f1_metric.dat"
    os.system(cmd)


    trainer.run()

    if configuration["output_name"] != "":
        output_name=configuration["output_name"]
    else:
        output_name=str(configuration["experiment"])

    # Saving the models
    serializers.save_npz("results/models/"+output_name+"_gen.model",gen)
    if configuration["wasserstein"]:
        serializers.save_npz("results/models/"+output_name+"_cri.model",cri)
    else:
        serializers.save_npz("results/models/"+output_name+"_dis.model",dis)
    
