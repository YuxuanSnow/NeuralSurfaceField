from libs.global_variable import ROOT_DIR

import torch
import torch.nn as nn
import torch.optim as optim

from pytorch3d.structures.meshes import Meshes
from pytorch3d.io import load_ply

import os
from glob import glob
import numpy as np
from os.path import join
from tqdm import tqdm
from collections import Counter

from libs.individual_checkpoint_loader import individual_conditional_sdf_checkpoint_

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                                                     #
#                                                             conditional sdf to learn coarse template                                                                #
#                                                                                                                                                                     #
#                                                                                                                                                                     #
class Basic_Trainer_sdf(object):

    def __init__(self,  module_dict, pretrained_module_name_dict, device, train_dataset=None, val_dataset=None, exp_name=None, dataset=None, optimizer='Adam'):
        self.device = device

        optimization_param_list = []
        feat_param_list = []

        self.conditional_sdf = None

        self.subject_global_latent = module_dict.get('subject_global_latent').to(self.device)
        feat_param_list += list(self.subject_global_latent.parameters()) 

        if module_dict.get('conditional_sdf') is not None:
            self.conditional_sdf = module_dict.get('conditional_sdf').to(self.device)
            optimization_param_list += list(self.conditional_sdf.parameters())

        if module_dict.get('inv_skinner') is not None:
            self.inv_skinner = module_dict.get('inv_skinner').to(self.device)
            self.inv_skinner_normal = module_dict.get('inv_skinner_normal').to(self.device)

        if module_dict.get('skinner') is not None:
            self.skinner = module_dict.get('skinner').to(self.device)
            self.skinner_normal = module_dict.get('skinner_normal').to(self.device)

        if module_dict.get('diffused_skinning_field') is not None:
            self.diffused_skinning_field = module_dict.get('diffused_skinning_field').to(self.device)


        if optimizer == 'Adam':
            self.optimizer = optim.Adam(optimization_param_list, lr=1e-4)
            self.feat_optimizer = optim.Adam(feat_param_list, lr=1e-4)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.dataset = dataset
        
        # checkpoints for regular modules
        self.exp_path = ROOT_DIR + 'experiments/{}/'.format(exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format(exp_name)
        if not os.path.exists(self.checkpoint_path):
            print("mkdir:", self.checkpoint_path)
            os.makedirs(self.checkpoint_path)

        # pretrained from previous experiments
        self.pretrained_conditional_sdf = pretrained_module_name_dict.get('conditional_sdf')

        self.get_pretrained_path()

    # defined member function outside of class
    individual_conditional_sdf_checkpoint = individual_conditional_sdf_checkpoint_

    def get_pretrained_path(self):

        base_path = ROOT_DIR + 'experiments/{}/'

        pretrained_conditional_sdf_path = base_path.format(self.pretrained_conditional_sdf)
        self.pretrained_conditional_sdf_checkpoint_path = pretrained_conditional_sdf_path + \
            'checkpoints/'.format(self.pretrained_conditional_sdf)

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
            model_weights = {'epoch': epoch,
                             'optimizer_state_dict': self.optimizer.state_dict()}
            if self.conditional_sdf is not None:
                model_weights.update(
                    {'conditional_sdf_state_dict': self.conditional_sdf.state_dict()})
            torch.save(model_weights, path)

    def load_checkpoint(self, number=None, path=None):
        if path is not None:
            print('Loaded checkpoint from: {}'.format(path))
            checkpoint = torch.load(path)
            if self.conditional_sdf is not None:
                self.conditional_sdf.load_state_dict(checkpoint['conditional_sdf_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            return 0
        else:
            checkpoints = glob(self.checkpoint_path + '/*')
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_path))
                return 0

            if number is None:
                checkpoints = [os.path.splitext(os.path.basename(path))[
                    0][17:] for path in checkpoints]
                checkpoints = np.array(checkpoints, dtype=int)
                checkpoints = np.sort(checkpoints)

                if checkpoints[-1] == 0:
                    print('Not loading model as this is the first epoch')
                    return 0

                path = join(self.checkpoint_path,
                            'checkpoint_epoch_{}.tar'.format(checkpoints[-1]))
            else:
                path = join(self.checkpoint_path,
                            'checkpoint_epoch_{}.tar'.format(number))

            print('Loaded checkpoint from: {}'.format(path))
            checkpoint = torch.load(path)

            epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.conditional_sdf is not None:
                self.conditional_sdf.load_state_dict(checkpoint['conditional_sdf_state_dict'])

            return epoch

    def train_model(self, epochs, pretrained=None, checkpoint=None):

        start = self.load_checkpoint(path=pretrained, number=checkpoint)
        self.individual_conditional_sdf_checkpoint()

        for epoch in range(start, epochs):
            print('Start epoch {}'.format(epoch))
            train_data_loader = self.train_dataset.get_loader()

            if epoch % 50 == 0 and epoch != start:
                self.save_checkpoint(epoch)
                self.subject_global_latent.save_features(epoch)
                
            sum_loss = None
            weights = [1, 1, 1]
            for n, batch in enumerate(tqdm(train_data_loader)):
                loss = self.train_step(batch, weights)
                if sum_loss is None:
                    sum_loss = Counter(loss)
                else:
                    sum_loss.update(Counter(loss))

            loss_str = ''
            for l in sum_loss:
                loss_str += '{}: {}, '.format(l, sum_loss[l] / len(train_data_loader))
            print(loss_str)

    def fine_tune_model(self, epochs, pretrained=True, checkpoint=None):

        start = self.load_checkpoint(path=pretrained, number=checkpoint)

        for epoch in range(start, epochs):
            print('Start epoch {}'.format(epoch))
            val_data_loader = self.val_dataset.get_loader()

            sum_loss = None
            weights = [1, 1, 1]
            for n, batch in enumerate(tqdm(val_data_loader)):
                loss = self.fine_tune_step(batch, weights)
                if sum_loss is None:
                    sum_loss = Counter(loss)
                else:
                    sum_loss.update(Counter(loss))

            if epoch % 100 == 0:
                self.subject_global_latent.save_features(epoch)

            loss_str = ''
            for l in sum_loss:
                loss_str += '{}: {}, '.format(l, sum_loss[l] / len(val_data_loader))
            print(loss_str)

    def train_step(self, batch, weights):
        self.set_feat_training_mode(train_flag=True)

        self.optimizer.zero_grad()
        self.feat_optimizer.zero_grad()

        loss_ = self.compute_loss(batch, weights, train=True, ssp_only=False)
        loss = self.sum_dict(loss_)

        loss.backward()

        self.optimizer.step()
        self.feat_optimizer.step()

        return {k: loss_[k].item() for k in loss_}

    def fine_tune_step(self, batch, weights):
        self.set_feat_training_mode(train_flag=True)

        self.feat_optimizer.zero_grad()

        loss_ = self.compute_loss(batch, weights, train=False, ssp_only=True)
        loss = self.sum_dict(loss_)

        loss.backward()

        # only update feat optimizer
        self.feat_optimizer.step()

        return {k: loss_[k].item() for k in loss_}

    def set_module_training_mode(self, train_flag):

        if self.conditional_sdf is not None:
            self.conditional_sdf.train(mode=train_flag)

    def set_feat_training_mode(self, train_flag):

        self.subject_global_latent.train(mode=train_flag)

    @staticmethod
    def sum_dict(los):
        temp = 0
        for l in los:
            temp += los[l]
        return temp

    def compute_loss(self, batch, train_flag, ssp_only):
        # overriding with child class
        pass

    def predict(self, inputs, train):
        # overriding with child class
        pass
#                                                                                                                                                                     #
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#