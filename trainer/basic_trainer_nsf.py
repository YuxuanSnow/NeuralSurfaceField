from libs.global_variable import ROOT_DIR

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR

import os
from glob import glob
import numpy as np
from os.path import join
from tqdm import tqdm
from collections import Counter

from libs.individual_checkpoint_loader import individual_pose_encoder_checkpoint_, individual_nsf_decoder_decoder_checkpoint_

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                                                     #
#                                                       inverse skinnig template, project to coarse template                                                          #
#                                                     query local feature on corresponding SMPL-D feature mesh                                                        #
#                                                           decode local feature to pose-dependent offset                                                             #
#                                                            use precomputed diffused SMPL skinning field                                                             #
#                                                                                                                                                                     #
class Basic_Trainer_nsf(object):

    def __init__(self,  module_dict, pretrained_module_name_dict, device, train_dataset=None, val_dataset=None, exp_name=None, dataset=None, optimizer='Adam'):
        self.device = device

        optimization_param_list = []
        feat_param_list = []

        self.pose_encoder = None
        self.nsf_decoder = None
        self.diffused_skinning_field = None

        self.nsf_feature_surface = module_dict.get('nsf_feature_surface').to(self.device)
        feat_param_list += list(self.nsf_feature_surface.parameters())

        if module_dict.get('inv_skinner') is not None:
            self.inv_skinner = module_dict.get('inv_skinner').to(self.device)
            self.inv_skinner_normal = module_dict.get('inv_skinner_normal').to(self.device)

        if module_dict.get('skinner') is not None:
            self.skinner = module_dict.get('skinner').to(self.device)
            self.skinner_normal = module_dict.get('skinner_normal').to(self.device)

        if module_dict.get('diffused_skinning_field') is not None:
            self.diffused_skinning_field = module_dict.get('diffused_skinning_field').to(self.device)

        if module_dict.get('pose_encoder') is not None:
            self.pose_encoder = module_dict.get('pose_encoder').to(self.device)
            optimization_param_list += list(self.pose_encoder.parameters())
            
        if module_dict.get('nsf_decoder') is not None:
            self.nsf_decoder = module_dict.get('nsf_decoder').to(self.device)
            optimization_param_list += list(self.nsf_decoder.parameters())

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(optimization_param_list, lr=1e-4)
            self.feat_optimizer = optim.Adam(feat_param_list, lr=1e-4)

            self.scheduler = StepLR(self.optimizer, step_size = 100, gamma = 0.5)
            self.feat_scheduler = StepLR(self.feat_optimizer, step_size = 100, gamma = 0.5)

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
        self.pretrained_pose_encoder = pretrained_module_name_dict.get('pose_encoder')
        self.pretrained_nsf_decoder = pretrained_module_name_dict.get('nsf_decoder')

        self.get_pretrained_path()

        self.nsf_feature_surface.query_verts_skinning(self.diffused_skinning_field, self.device)

    # defined member function outside of class
    individual_nsf_decoder_checkpoint = individual_nsf_decoder_decoder_checkpoint_
    individual_pose_encoder_checkpoint = individual_pose_encoder_checkpoint_

    def get_pretrained_path(self):

        base_path = ROOT_DIR + 'experiments/{}/'

        pretrained_pose_encoder_path = base_path.format(self.pretrained_pose_encoder)
        self.pretrained_pose_encoder_checkpoint_path = pretrained_pose_encoder_path + \
            'checkpoints/'.format(self.pretrained_pose_encoder)
        
        pretrained_nsf_decoder_path = base_path.format(self.pretrained_nsf_decoder)
        self.pretrained_nsf_decoder_checkpoint_path = pretrained_nsf_decoder_path + \
            'checkpoints/'.format(self.pretrained_nsf_decoder)

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
            model_weights = {'epoch': epoch,
                             'optimizer_state_dict': self.optimizer.state_dict(), 
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'feat_optimizer_state_dict': self.feat_optimizer.state_dict(),
                            'feat_scheduler_state_dict': self.feat_scheduler.state_dict()}
            if self.pose_encoder is not None:
                model_weights.update(
                    {'pose_encoder_state_dict': self.pose_encoder.state_dict()})
            if self.nsf_decoder is not None:
                model_weights.update(
                    {'nsf_decoder_state_dict': self.nsf_decoder.state_dict()})
            torch.save(model_weights, path)

    def load_checkpoint(self, number=None, path=None):

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

            path = join(self.checkpoint_path, 'checkpoint_epoch_{}.tar'.format(checkpoints[-1]))
        else:
            path = join(self.checkpoint_path, 'checkpoint_epoch_{}.tar'.format(number))

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)

        # load optimizers and schedulers to recover training 
        epoch = checkpoint['epoch']
        if checkpoint.get('optimizer_state_dict') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('load network optimizer')
        if checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print('load scheduler for network optimizer')
        if checkpoint.get('feat_optimizer_state_dict') is not None:
            # self.feat_optimizer.load_state_dict(checkpoint['feat_optimizer_state_dict'])
            print('load feature space optimizer')
        if checkpoint.get('feat_scheduler_state_dict') is not None:
            # self.feat_scheduler.load_state_dict(checkpoint['feat_scheduler_state_dict'])
            print('load scheduler for feature space optimizer')

        # load networks checkpoint
        if self.pose_encoder is not None:
            self.pose_encoder.load_state_dict(checkpoint['pose_encoder_state_dict'])
            print("load pose encoder at epoch {}".format(epoch))
        if self.nsf_decoder is not None:
            self.nsf_decoder.load_state_dict(checkpoint['nsf_decoder_state_dict'])
            print("load nsf decoder at epoch {}".format(epoch))

        return epoch

    def train_model(self, epochs, pretrained=None, checkpoint=None):

        start = self.load_checkpoint(path=pretrained, number=checkpoint)
        self.individual_pose_encoder_checkpoint()
        self.individual_nsf_decoder_checkpoint

        for epoch in range(start, epochs):
            print('Start epoch {}'.format(epoch))
            train_data_loader = self.train_dataset.get_loader()

            if epoch % 20 == 0 and epoch != start:
                self.save_checkpoint(epoch)
                self.nsf_feature_surface.save_features(epoch)

            w_v2v = 1e5
            w_sw = 1e2
            w_normal = 10
            w_rgl = 1e3

            if epoch < 20:
                w_rgl = 1e7
            
            w_latent_rgl = 1
            weights = torch.tensor([w_v2v, w_sw, w_normal, w_rgl, w_latent_rgl])
            
            sum_loss = None
            for n, batch in enumerate(tqdm(train_data_loader)):
                loss = self.train_step(batch, weights)
                if sum_loss is None:
                    sum_loss = Counter(loss)
                else:
                    sum_loss.update(Counter(loss))

            # apply lr scheduler after each epoch
            self.scheduler.step()
            self.feat_scheduler.step()

            loss_str = ''
            for l in sum_loss:
                loss_str += '{}: {}, '.format(l, sum_loss[l] / len(train_data_loader))
            print(loss_str)

    def fine_tune_model(self, epochs, pretrained=True, checkpoint=None):

        start = self.load_checkpoint(path=pretrained, number=checkpoint, load_feat_optimizer=True)

        for epoch in range(start, epochs):
            print('Start epoch {}'.format(epoch))
            val_data_loader = self.val_dataset.get_loader()

            w_v2v = 1e5
            w_sw = 1e2
            w_normal = 10
            w_latent_rgl = 1
            w_rgl = 1e3
            weights = torch.tensor([w_v2v, w_sw, w_normal, w_rgl, w_latent_rgl])

            sum_loss = None
            for n, batch in enumerate(tqdm(val_data_loader)):
                loss = self.fine_tune_step(batch, weights)
                if sum_loss is None:
                    sum_loss = Counter(loss)
                else:
                    sum_loss.update(Counter(loss))

            self.feat_scheduler.step()

            if epoch % 20 == 0:
                self.nsf_feature_surface.save_features(epoch)

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
    
    @staticmethod
    def set_batch_norm_running_stats(module):
        # resolves model.eval() degradation because of batch norm
        for m in module.modules():
            for child in m.children():
                if type(child) == nn.BatchNorm2d or type(child) == nn.BatchNorm1d:
                    child.track_running_stats = False
                    child.running_mean = None
                    child.running_var = None

    def set_module_training_mode(self, train_flag):

        if self.pose_encoder is not None:
            # if train_flag == False:
            #     self.set_batch_norm_running_stats(self.pose_encoder)
            self.pose_encoder.train(mode=train_flag)
        if self.nsf_decoder is not None:
            # if train_flag == False:
            #    self.set_batch_norm_running_stats(self.shape_geometry_decoder)
            self.nsf_decoder.train(mode=train_flag)

    def set_feat_training_mode(self, train_flag):
        
        # not train local feature
        self.nsf_feature_surface.train(mode=train_flag) 

    @staticmethod
    def sum_dict(los):
        temp = 0
        for l in los:
            temp += los[l]
        return temp
    
    def val_model(self, save_name='val', num_samples=-1, pretrained=None, checkpoint=None):   
        # overriding with child class
        pass
    
    def compute_loss(self, batch, train_flag, ssp_only):
        # overriding with child class
        pass

    def predict(self, inputs, train):
        # overriding with child class
        pass
#                                                                                                                                                                     #
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#