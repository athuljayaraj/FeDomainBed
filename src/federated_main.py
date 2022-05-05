#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import pickle
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
os.environ["WANDB_API_KEY"] = "e850cb010e84db3ef3f51b131087c4615181d915"

if __name__ == '__main__':
    wandb.init(project="FeDomainBed")

    if __name__ == '__main__':
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
    
    # BUILD MODEL
    # if args.model == 'cnn':
    #     # Convolutional neural netork
    #     if args.dataset == 'mnist':
    #         global_model = CNNMnist(args=args)
    #     elif args.dataset == 'fmnist':
    #         global_model = CNNFashion_Mnist(args=args)
    #     elif args.dataset == 'cifar':
    #         global_model = CNNCifar(args=args)

    # elif args.model == 'mlp':
    #     # Multi-layer preceptron
    #     img_size = train_dataset[0][0].shape
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #         global_model = MLP(dim_in=len_in, dim_hidden=64,
    #                            dim_out=args.num_classes)
    # else:
    #     exit('Error: unrecognized model')

    # Training
        train_loss, train_accuracy = [], []
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 2
        val_loss_pre, counter = 0, 0

    # choose algorithm
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
        batch_size=hparams['batch_size']
        dataset = vars(datasets)["ColoredMNIST"]("data/", 100000, hparams)

        algorithm_class = get_algorithm_class(args.algorithm)
    
        input_shape = train_dataset.data[0].shape
        print('input shape: ')
        print(input_shape)
        algorithm = algorithm_class(input_shape, 10, len(train_dataset.data) - 10000, hparams)
    
        global_model = algorithm.get_network()

    # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()

    # copy weights
        global_weights = global_model.state_dict()


        for epoch in tqdm(range(args.epochs)):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')

            global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, algorithm=algorithm)
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

        # update global weights
            global_weights = average_weights(local_weights)

        # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss += [loss_avg]

        # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            global_model.eval()
            for c in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, algorithm=algorithm)
                acc, loss = local_model.inference(model=global_model)
                list_acc += [acc]
                list_loss += [loss]
            train_accuracy += [sum(list_acc)/len(list_acc)]

        # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            train_acc = train_accuracy[-1]; train_loss = np.mean(np.array(train_loss))
    # Test inference after completion of training
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            wandb.log({'test accuracy': test_acc, 'test loss': test_loss, 'train accuracy': train_acc, 'train loss': train_loss})
        
        print(f' \n Results after {args.epochs} global rounds of training:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
        file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
            format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
