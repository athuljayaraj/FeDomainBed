#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import numpy as np
from domainbed.algorithms import get_algorithm_class
from domainbed import datasets
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details

from domainbed.lib import hparams_registry
import wandb
import os

os.environ["WANDB_API_KEY"] = "69b6150e50e6f283a5d48476cbc43830f0e35d3c"

if __name__ == '__main__':
    wandb.init(project="fedomainbed", entity="athul")
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = "cpu"
    if args.gpu != None:
        device = 'cuda'
        torch.cuda.device(args.gpu)

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # choose algorithm
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    batch_size = hparams['batch_size']

    algorithm_class = get_algorithm_class(args.algorithm)
    input_shape = train_dataset.data[0].shape
    print(f'input shape: {input_shape}')
    algorithm = algorithm_class(input_shape, 10, len(
        train_dataset.data) - 10000, hparams)

    global_model = algorithm.get_network()

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()

    for epoch in tqdm(range(args.epochs)):
        local_weights = []
        local_accuracies, local_losses = [], []

        print(f'\n | Global Training Round : {epoch+1} |\n')

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, algorithm=algorithm)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))

            local_model_for_client = algorithm.get_network()
            local_model_for_client.load_state_dict(w)
            train_accuracy_for_client, train_loss_for_client = local_model.train_inference(
                local_model_for_client)

            local_accuracies.append(train_accuracy_for_client)
            local_losses.append(train_loss_for_client)

        train_accuracy += [sum(local_accuracies)/len(local_accuracies)]
        train_loss += [sum(local_losses) / len(local_losses)]

        # update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        train_acc_this_epoch = train_accuracy[-1]
        train_loss = np.mean(np.array(train_loss))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f'\nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {train_loss}')
            print('Train Accuracy: {:.2f}% \n'.format(
                100*train_acc_this_epoch))

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        wandb.log({'test accuracy': test_acc, 'test loss': test_loss,
                  'train accuracy': train_acc_this_epoch, 'train loss': train_loss})

        print(f' \n Results after {args.epochs} global rounds of training:')
        print(
            "|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
