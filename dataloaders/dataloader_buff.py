from os.path import split, join
import pickle as pkl
import numpy as np
import torch
from dataloaders.dataloader_base import BaseLoader
from tqdm import tqdm
import os

from scipy.spatial.transform import Rotation as ScipyRot

# Dataloader for BuFF inv skinning: process all files, without split_file;
class DataLoader_Buff_depth(BaseLoader):

    def __init__(self, mode='train', cano_available=False, proprocessed_path=None, split_file=None, batch_size=64, num_workers=12, subject_index_dict=None, num_points=30000):  
        # num_points: each frame has different number of depth point cloud. Repeat some to get same points of sample in one batch.

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode

        if proprocessed_path is None and split_file is None:
            raise ValueError("proprocessed_path and split_file cannot be both None")
        
        self.cano_available = cano_available

        if split_file is not None:
            with open(split_file, "rb") as f:
                self.split = pkl.load(f)
            self.data = self.split[mode]
            # if no split file: performing preprocessing on all files, namely iterative root finding and projecting to fusion shape!
            self.cano_available = True
        else:
            self.data = []
            for subj in os.listdir(proprocessed_path):
                for garment_action in os.listdir(join(proprocessed_path, subj)):
                    for preprocessed_npy_file in sorted(os.listdir(join(proprocessed_path, subj, garment_action))):
                        if preprocessed_npy_file.split(".")[0].endswith('cano'):
                            continue
                        self.data.append(join(proprocessed_path, subj, garment_action, preprocessed_npy_file))
                    
        self.num_points = num_points
        self.subject_index_dict = subject_index_dict

        self.scan_points_rotated, self.scan_normals_rotated, self.scan_colors = [], [], []
        self.scan_points, self.scan_normals = [], []
        self.ref_shaped_points = []
        self.hand_mask, self.feet_mask, self.head_mask = [], [], []
        self.skinning_weights = []
        self.pose, self.betas, self.trans, self.roty = [], [], [], []
        self.feature_cube_idx = []
        self.depth_map = []
        self.rgb_map = []
        self.path = []

        self.num_org_points = []

        if self.cano_available:
            self.cano_points, self.cano_normals = [], []

        self._init_dataset()
        print('Data loaded, in total {} {} examples.\n'.format(len(self.scan_points), self.mode))

    @staticmethod
    def _get_repeated_idx(num_points_input, num_points):

        assert(num_points_input <= num_points)
        idx_org = np.arange(0, num_points_input)

        num_points_rest = num_points - num_points_input
        if num_points_rest <= num_points_input:
            idx_rest = np.random.choice(np.arange(0, num_points_input), num_points_rest, replace=False)
        elif num_points_rest > num_points_input:
            num_points_rest_1 = num_points_input
            idx_rest_1 = np.random.choice(np.arange(0, num_points_input), num_points_rest_1, replace=False)
            num_points_rest_2 = num_points_rest - num_points_rest_1
            idx_rest_2 = np.random.choice(np.arange(0, num_points_input), num_points_rest_2, replace=False)
            idx_rest = np.concatenate((idx_rest_1, idx_rest_2))

        idx_list = np.concatenate((idx_org, idx_rest))

        return idx_list

    def _init_dataset(self):

        print('Loading {} data...'.format(self.mode))

        for idx, file_path in enumerate(tqdm(self.data)):
            dd = np.load(file_path, allow_pickle=True).item()

            subject = split(file_path)[0].split('/')[9][:5]                     # if local workstaiton then 9; if cluster then 12
            garment = split(file_path)[0].split('/')[10].split('_')[0]
            subject_garment = subject + "_" + garment
            subject_garment_idx = self.subject_index_dict[subject_garment]
            self.feature_cube_idx.append(torch.tensor(subject_garment_idx).long())

            num_points_input = dd['points_posed_cloth'].shape[1]

            if self.num_points >= 0:
                idx_list = self._get_repeated_idx(num_points_input, self.num_points)
            else:
                idx_list = np.arange(0, num_points_input)

            # unrotate the additional root rotation -> no need to do in training
            r_rot_y = ScipyRot.from_rotvec(dd['rot_vector'])
            scan_inv = r_rot_y.inv().apply(dd['points_posed_cloth'].transpose())
            normal_inv = r_rot_y.inv().apply(dd['normals_posed_cloth'].transpose()) # pymeshlab estimated normal is inverted
            
            self.scan_points_rotated.append(torch.tensor(dd['points_posed_cloth'].transpose()[idx_list].transpose()).float())
            self.scan_normals_rotated.append(torch.tensor(dd['normals_posed_cloth'].transpose()[idx_list].transpose()).float()) # pymeshlab estimated normal is inverted
            self.scan_colors.append(torch.tensor(dd['colors_posed_cloth'].transpose()[idx_list].transpose()).float())
            self.scan_points.append(torch.tensor(scan_inv[idx_list].transpose()).float())
            self.scan_normals.append(torch.tensor(normal_inv[idx_list].transpose()).float())
            self.ref_shaped_points.append(torch.tensor(dd['points_ref_smpl'].transpose()[idx_list].transpose()).float())
            self.skinning_weights.append(torch.tensor(dd['skinning_weights'].transpose()[idx_list].transpose()).float())
            self.pose.append(torch.tensor(dd['pose']).float())
            self.betas.append(torch.tensor(dd['betas']).float())
            self.trans.append(torch.tensor(dd['trans']).float())
            self.roty.append(torch.tensor(dd['rot_vector']).float())
            self.depth_map.append(torch.tensor(dd['depth_img']).float())
            self.rgb_map.append(torch.tensor(dd['color_img']).float())
            self.path.append(file_path)
            self.num_org_points.append(num_points_input)

            if self.cano_available:
                dd_cano = np.load(file_path.split(".")[0] + "_cano." + file_path.split(".")[1], allow_pickle=True).item()
                self.cano_points.append(torch.tensor(dd_cano['cano_points'][idx_list].transpose()).float())
                self.cano_normals.append(torch.tensor(dd_cano['cano_normals'][idx_list].transpose()).float())

    def __getitem__(self, idx):
        
        return {'scan_points': self.scan_points[idx],
                'scan_normals': self.scan_normals[idx],
                'scan_colors': self.scan_colors[idx],
                'scan_points_rotated': self.scan_points_rotated[idx],
                'scan_normals_rotated': self.scan_normals_rotated[idx],
                'pose': self.pose[idx],
                'betas': self.betas[idx],
                'ref_shaped_points': self.ref_shaped_points[idx],
                'feature_cube_idx': self.feature_cube_idx[idx],
                'skinning_weights': self.skinning_weights[idx],
                'trans': self.trans[idx],
                'roty': self.roty[idx],
                'path': self.path[idx],
                'num_org_points': self.num_org_points[idx],
                'cano_points': self.cano_points[idx],
                'cano_normals': self.cano_normals[idx]}
