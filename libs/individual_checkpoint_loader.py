from glob import glob
import os
import numpy as np
from os.path import join
import torch

def individual_conditional_sdf_checkpoint_(self):
    # load CorrNet from the given path (load last from the given experiment)
    if self.pretrained_conditional_sdf != None:
        if self.pretrained_conditional_sdf.split("_")[-1] != 'None':
            conditional_sdf_checkpoints = glob(self.pretrained_conditional_sdf_checkpoint_path + '/*')
            if len(conditional_sdf_checkpoints) == 0:
                print('No checkpoints of sdf found at {}'.format(self.pretrained_conditional_sdf_checkpoint_path))
                return

            conditional_sdf_checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in conditional_sdf_checkpoints]
            conditional_sdf_checkpoints = np.array(conditional_sdf_checkpoints, dtype=int)
            conditional_sdf_checkpoints = np.sort(conditional_sdf_checkpoints)

            if conditional_sdf_checkpoints[-1] == 0:
                print('Not loading model as this is the first epoch')
                return

            pretrained_conditional_sdf_checkpoint_path = join(self.pretrained_conditional_sdf_checkpoint_path, 'checkpoint_epoch_{}.tar'.format(conditional_sdf_checkpoints[-1]))

            conditional_sdf_checkpoint = torch.load(pretrained_conditional_sdf_checkpoint_path)
            self.conditional_sdf.load_state_dict(conditional_sdf_checkpoint['conditional_sdf_state_dict'])
            print('Loaded pretrained conditional_sdf at epoch {}'.format(conditional_sdf_checkpoint['epoch']))
    else:
        print('Not use pretrained conditional sdf')
