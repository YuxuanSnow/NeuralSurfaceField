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


def individual_pose_encoder_checkpoint_(self):
    # load Pose Encoder from the given path (load last from the given experiment)
    if self.pretrained_pose_encoder != None:
        if self.pretrained_pose_encoder.split("_")[-1] != 'None':
            pose_encoder_checkpoints = glob(self.pretrained_pose_encoder_checkpoint_path + '/*')
            if len(pose_encoder_checkpoints) == 0:
                print('No checkpoints of pose encoder found at {}'.format(self.pretrained_pose_encoder_checkpoint_path))
                return

            pose_encoder_checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in pose_encoder_checkpoints]
            pose_encoder_checkpoints = np.array(pose_encoder_checkpoints, dtype=int)
            pose_encoder_checkpoints = np.sort(pose_encoder_checkpoints)

            if pose_encoder_checkpoints[-1] == 0:
                print('Not loading model as this is the first epoch')
                return

            pretrained_pose_encoder_checkpoint_path = join(self.pretrained_pose_encoder_checkpoint_path, 'checkpoint_epoch_{}.tar'.format(pose_encoder_checkpoints[-1]))

            pose_encoder_checkpoint = torch.load(pretrained_pose_encoder_checkpoint_path)
            self.pose_encoder.load_state_dict(pose_encoder_checkpoint['pose_encoder_state_dict'])
            print('Loaded pretrained pose encoder at epoch {}'.format(pose_encoder_checkpoint['epoch']))
    else:
        print('Not use pretrained pose encoder')


def individual_nsf_decoder_decoder_checkpoint_(self):
    # load shape decoder from the given path (load last from the given experiment)
    if self.pretrained_nsf_decoder_decoder != None:
        if self.pretrained_nsf_decoder_decoder.split("_")[-1] != 'None':
            nsf_decoder_decoder_checkpoints = glob(self.pretrained_nsf_decoder_decoder_checkpoint_path + '/*')
            if len(nsf_decoder_decoder_checkpoints) == 0:
                print('No checkpoints of nsf shape decoder found at {}'.format(self.pretrained_nsf_decoder_decoder_checkpoint_path))
                return

            nsf_decoder_decoder_checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in nsf_decoder_decoder_checkpoints]
            nsf_decoder_decoder_checkpoints = np.array(nsf_decoder_decoder_checkpoints, dtype=int)
            nsf_decoder_decoder_checkpoints = np.sort(nsf_decoder_decoder_checkpoints)

            if nsf_decoder_decoder_checkpoints[-1] == 0:
                print('Not loading model as this is the first epoch')
                return

            pretrained_nsf_decoder_decoder_checkpoint_path = join(self.pretrained_nsf_decoder_decoder_checkpoint_path, 'checkpoint_epoch_{}.tar'.format(nsf_decoder_decoder_checkpoints[-1]))

            nsf_decoder_decoder_checkpoint = torch.load(pretrained_nsf_decoder_decoder_checkpoint_path)
            self.nsf_decoder_decoder.load_state_dict(nsf_decoder_decoder_checkpoint['nsf_decoder_state_dict'])
            print('Loaded pretrained nsf shape deoder at epoch {}'.format(nsf_decoder_decoder_checkpoint['epoch']))
    else:
        print("Not use pretrained nsf shape decoder")
