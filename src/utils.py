#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import numpy as np
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def generate_random_rotation_angle():
    return np.random.randint(0, 180, dtype=np.uint8)


def rotate_mnist(images):
    random_rotation_angle = generate_random_rotation_angle()
    apply_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation((random_rotation_angle-10, random_rotation_angle+10), fill=0)])
    rotated_images = list(map(apply_transform, images))
    return rotated_images


def get_random_color():
    return np.random.randint(0, 255, dtype=np.uint8)


COLOR_FOR_CLASS = {i: (get_random_color(), get_random_color(),
                       get_random_color()) for i in range(10)}


def generate_random_colors(number_of_colors=2):
    random_colors = []
    for i in range(number_of_colors):
        random_color = (get_random_color(),
                        get_random_color(), get_random_color())
        random_colors.append(random_color)
    return random_colors


def pick_at_random(a_list):
    if len(a_list) == 1:
        return a_list[0]
    else:
        return a_list[np.random.randint(0, len(a_list) - 1)]


def expand_to_3d(data):
    data_3channel = []
    for image in data:
        image = torch.reshape(image, [28, 28, 1])
        data_3channel.append(torch.cat((image, image, image), axis=-1))
    return data_3channel


def color_mnist(list_of_images, number_of_colors=2):
    colors = generate_random_colors(number_of_colors)
    colored_images = []
    for image in list_of_images:
        color = torch.tensor(pick_at_random(colors))
        colored_images.append(torch.where(
            image != torch.tensor([0, 0, 0]), color, image))
    return colored_images


def classwise_color_mnist(images, labels, number_of_colors_per_class=1):
    colored_images = []
    for (image, label) in zip(images, labels):
        color = torch.tensor(COLOR_FOR_CLASS[int(label)])
        colored_images.append(torch.where(
            image != torch.tensor([0, 0, 0]), color, image))
    return colored_images


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    data_dir = '../data/{}'.format(args.dataset)
    if args.dataset == 'cifar':
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose unequal splits for every user
                raise NotImplementedError()
            else:
                # Chose equal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)
    elif args.dataset == 'mnist' or 'fmnist' or 'cmnist' or 'rotmnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(
                    train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    if args.dataset == 'cmnist':
        train_dataset.data = expand_to_3d(train_dataset.data)
        print("Coloring the MNIST")
        print("shapes", train_dataset.data[0].shape)
        for client in user_groups:
            colored_mnist_for_client = color_mnist(list(
                map(train_dataset.data.__getitem__, user_groups[client])), number_of_colors=5)
            for idx_client, idx in enumerate(user_groups[client]):
                train_dataset.data[idx] = colored_mnist_for_client[idx_client]

        for idx in range(len(train_dataset.data)):
            train_dataset.data[idx] = torch.moveaxis(
                train_dataset.data[idx], -1, 0)
            train_dataset.data[idx] = torch.moveaxis(
                train_dataset.data[idx], 1, -1)/255

        test_dataset.data = expand_to_3d(test_dataset.data)
        test_dataset.data = color_mnist(test_dataset.data, number_of_colors=5)
        for idx in range(len(test_dataset.data)):
            test_dataset.data[idx] = torch.moveaxis(
                test_dataset.data[idx], -1, 0)
            test_dataset.data[idx] = torch.moveaxis(
                test_dataset.data[idx], 1, -1)/255

    if args.dataset == 'rotmnist':
        for client in user_groups:
            rotated_mnist_for_client = rotate_mnist(list(
                map(train_dataset.data.__getitem__, user_groups[client])))
            for idx_client, idx in enumerate(user_groups[client]):
                train_dataset.data[idx] = rotated_mnist_for_client[idx_client]/255

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}')
    print(
        f'    Device   : {"cuda:{}".format(args.gpu) if args.gpu != None else "cpu"}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
