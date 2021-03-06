import os
import json
import argparse
from chainer import Variable,datasets, training, iterators, optimizers, serializers
from chainer.training import updater, extensions, Trainer
from iterators import RandomNoiseIterator, UniformNoiseGenerator
# Cinfogan
from models import Generator, Discriminator, Critic
from updaters import GANUpdater, WassersteinGANUpdater
# Cgan
from cgan_updaters import Cgan_GANUpdater, Cgan_WassersteinGANUpdater

from extensions import GeneratorSample, saving_model
import numpy as np
import import_dataset

iterators.RandomNoiseIterator = RandomNoiseIterator
updater.GANUpdater = Cgan_GANUpdater
extensions.GeneratorSample = GeneratorSample


def training(configuration, i, verbose=False):
    #setting parameters
    batch_size = configuration["n_batchsize"]
    epochs     = configuration["n_epochs"]
    n_z  = configuration["n_z"]
    n_categorical = configuration["n_categorical"]
    n_continuous = configuration["n_continuous"]
    experiment = configuration["experiment"]
    n_neurons_gen  = configuration["n_neurons_gen"]
    n_neurons_dis  = configuration["n_neurons_dis"]
    n_neurons_cri  = configuration["n_neurons_cri"]
    gpu = configuration["gpu"]
    output_name = configuration["output_name"]

    #import the training data 

    if experiment == "random_left_right":
        x_dim=12         #problem
        xi_dim=6         #trajectory
    else:
        x_dim=14
        xi_dim=7

    if verbose:
        print("Loading data")
    train = import_dataset.import_data(configuration, x_dim, xi_dim)
    train_iter = iterators.SerialIterator(train, batch_size)

    z_iter = iterators.RandomNoiseIterator(UniformNoiseGenerator(-1, 1, n_z+n_continuous), batch_size)

    # Creating the Neural Networks models
    gen = Generator(n_z+n_continuous, x_dim, xi_dim, n_neurons_gen)
    dis = Discriminator(x_dim, xi_dim, n_neurons_dis,
                        cinfogan=configuration["cinfogan"], n_continuous=n_continuous)
    critic=Critic(x_dim, xi_dim, n_neurons_cri)

    if(configuration["cinfogan"]):
        saving_directory="results/models/cinfogan_models_"+str(i)
    else:
        saving_directory="results/models/cgan_models_"+str(i)
        
    if configuration["wasserstein"]:
        if verbose:
            print("Using Wasserstein")
        optimizer_generator = optimizers.RMSprop(lr=0.00005)
        optimizer_critic = optimizers.RMSprop(lr=0.00005)

        a=xp.zeros((1,xi_dim)).astype(xp.float32)
        b=xp.zeros((1,x_dim)).astype(xp.float32)
        critic(Variable(a),Variable(b))
        
        optimizer_generator.setup(gen)
        optimizer_critic.setup(critic)

        updater = WassersteinGANUpdater(
            iterator=train_iter,
            noise_iterator=z_iter,
            noise_dim=n_z,
            x_dim=x_dim,
            xi_dim=xi_dim,
            experiment=configuration["experiment"],
            optimizer_generator=optimizer_generator,
            optimizer_critic=optimizer_critic,
            device=gpu,
        )
    else:
        if verbose:
            print("Not using Wasserstein")
        optimizer_generator = optimizers.Adam()
        optimizer_discriminator = optimizers.SGD()
        
        optimizer_generator.setup(gen)
        optimizer_discriminator.setup(dis)

        if configuration["cinfogan"]:
            updater = GANUpdater(
                iterator=train_iter,
                noise_iterator=z_iter,
                noise_dim=n_z,
                continuous_dim = n_continuous,
                x_dim=x_dim,
                xi_dim=xi_dim,
                experiment=configuration["experiment"],
                optimizer_generator=optimizer_generator,
                optimizer_discriminator=optimizer_discriminator,
                collision_measure=configuration["collision_measure"],
                saving_directory=saving_directory,
                device=gpu
            )
        else:
            updater = Cgan_GANUpdater(
                iterator=train_iter,
                noise_iterator=z_iter,
                noise_dim=n_z,
                x_dim=x_dim,
                xi_dim=xi_dim,
                experiment=configuration["experiment"],
                optimizer_generator=optimizer_generator,
                optimizer_discriminator=optimizer_discriminator,
                collision_measure=configuration["collision_measure"],
                saving_directory=saving_directory,
                device=gpu
            )

    if verbose:
        print("setup trainer...")
    trainer = Trainer(updater, stop_trigger=(epochs, 'epoch'))

    # changing the name because we do multiple experiments
    if configuration["cinfogan"]:
        trainer.out="results/models/cinfogan_models_"+str(i)
    else:
        trainer.out="results/models/cgan_models_"+str(i)

        
    trainer.extend(extensions.LogReport())

    
    if configuration["wasserstein"]:        
        print_report_args = ['epoch', 'gen/loss', 'cri/loss',
                             'lin_ratio','cgan_ratio','diff_ratio']
    else:
        print_report_args = ['epoch', 'gen/loss', 'dis/loss',
                             'lin_ratio','cgan_ratio','diff_ratio']


    if verbose:
        trainer.extend(extensions.PrintReport(print_report_args))
        trainer.extend(extensions.ProgressBar())
    if configuration["collision_measure"] != 0:
        trainer.extend(extensions.GeneratorSample(configuration, x_dim,                                                xi_dim, n_continuous, n_z, train), trigger=(1, 'epoch'))


    # We delete the f_metric.dat file to be sure we do not mixed multiple experiment data.
    cmd = "touch results/f_metric.dat && rm results/f_metric.dat"
    os.system(cmd)

    if verbose:
        print("START TRAINING!!")
    trainer.run()

    if configuration["output_name"] != "":
        output_name=configuration["output_name"]
    else:
        output_name=str(configuration["experiment"])   


if __name__ == '__main__':
    with open(os.path.join("conf.json")) as fd:
        json_data = json.load(fd)
    configuration=json_data
    training(configuration,0)
    
