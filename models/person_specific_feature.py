import torch
from torch import nn

import os
from os.path import exists, split, join
import numpy as np
from glob import glob

# global feature for sdf
class SubjectGlobalLatentFeature(nn.Module):
    def __init__(self, num_subjects, subject_paths, pretrained_feature_exp=None, latent_size=256):

        super(SubjectGlobalLatentFeature, self).__init__()

        assert num_subjects == len(subject_paths)
        self.subject_paths = subject_paths
        self.pretrained_feature_paths = pretrained_feature_exp

        self.dim = latent_size
        epoch = self.load_features(pretrained_feature_exp)

    def save_features(self, epoch):
        for n, sp in enumerate(self.subject_paths):
            garment = sp.split('/')[-1]
            subject = sp.split('/')[-2]
            subject_garment = subject + garment
            path = join(sp, 'global_latent_{}.npy'.format(epoch))
            if not exists(sp):
                os.makedirs(sp)
            np.save(path, self.features[n].detach().cpu().numpy())
            print('Saved {} specific latent at epoch {}'.format(subject_garment, epoch))

    def load_features(self, pretrained_feature_path):
        
        self.features = []
        if pretrained_feature_path is None:
            print('Not use pretrained subject latent')
            # no pretrained feature path, try to load from current path
            flag = self.load_current_exp()
            assert(False)

        else:
            pretrained_exp_id = pretrained_feature_path[0].split('/')[-3].split('_')[3]
            if pretrained_exp_id == 'None':
                flag = self.load_current_exp()
            else:
                flag = self.load_pretrain_exp()
        
        return flag

    def load_current_exp(self):
        for n, sp in enumerate(self.subject_paths):
            flag = -1 

            garment = sp.split('/')[-1]
            subject = sp.split('/')[-2]
            subject_garment = subject + garment

            if exists(sp):
                # Load pre-saved features
                checkpoints = glob(join(sp, '*'))
                if len(checkpoints) > 0:
                    index_list = []
                    for path in checkpoints:
                        if split(path)[1].startswith('global_latent'):
                            index_list.append(split(path)[1][:-4].split('_')[2])
                    checkpoints = index_list
                    checkpoints = np.array(checkpoints, dtype=int)
                    checkpoints = np.sort(checkpoints)
                    path = join(sp, 'global_latent_{}.npy'.format(checkpoints[-1]))

                    temp = np.load(path)
                    self.features.append(torch.tensor(temp))
                    flag = max(flag, checkpoints[-1])

                    print('Load {} specific latent at epoch {}'.format(subject_garment, flag))

            if flag==-1:
                print('No {} specific latent found. Use initialized Latent.'.format(subject_garment))
                temp = torch.ones(self.dim)*0.5
                self.features.append(temp)

        self.features = torch.nn.Parameter(torch.stack(self.features, 0), requires_grad=True)    
        return flag

    def load_pretrain_exp(self):
        for n, pretrained_sp in enumerate(self.pretrained_feature_paths):
            flag = -1

            garment = pretrained_sp.split('/')[-1]
            subject = pretrained_sp.split('/')[-2]
            subject_garment = subject + garment

            if exists(pretrained_sp):
                # Load pre-saved features
                checkpoints = glob(join(pretrained_sp, '*'))
                if len(checkpoints) > 0:
                    index_list = []
                    for path in checkpoints:
                        if split(path)[1].startswith('global_latent'):
                            index_list.append(split(path)[1][:-4].split('_')[2])
                    checkpoints = index_list
                    checkpoints = np.array(checkpoints, dtype=int)
                    checkpoints = np.sort(checkpoints)
                    path = join(pretrained_sp, 'global_latent_{}.npy'.format(checkpoints[-1]))

                    temp = np.load(path)
                    self.features.append(torch.tensor(temp))
                    flag = max(flag, checkpoints[-1])

                    print('Load {} pretrained specific latent at epoch {}'.format(subject_garment, flag))

            if flag==-1:
                print('No {} specific latent found.'.format(subject_garment))
                assert(False)
                
        self.features = torch.nn.Parameter(torch.stack(self.features, 0), requires_grad=True)    
        return flag


    def get_feature(self, subject_garment_id):

        subject_specific_latent = self.features[subject_garment_id]

        return subject_specific_latent


    def forward(self, points, subject_garment_id):

        subject_specific_latent = self.features[subject_garment_id] 
    
        query_value = torch.cat([points, subject_specific_latent.unsqueeze(1).repeat((1, points.shape[2], 1)).permute(0, 2, 1)], dim=1) 

        return query_value 

