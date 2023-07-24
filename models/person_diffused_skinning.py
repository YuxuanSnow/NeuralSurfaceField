import torch
from torch import nn
from torch.nn import functional as F

import trimesh
import numpy as np

from lib.sample import compute_smaple_on_body_mask_w_batch



# precomputed skinning field: BuFF, 00032, 00096
class SmoothDiffusedSkinningField(nn.Module):
    def __init__(self, subject_field_base_path, general_subject_index, general_subject_index_numer):
        """
        Person specific smoothly diffused skinning field
        subject_paths: list of paths to each subject's skinning field
        """
        super(SmoothDiffusedSkinningField, self).__init__()

        self.general_subject_index_numer = general_subject_index_numer

        # body part corresponding SMPL vertices
        left_hand_vertex_index = 2005 # palm center
        right_hand_vertex_index = 5509 # palm center
        left_foot_vertex_index = 3392 # ankle, ~5 cm above the ground
        right_foot_vertex_index = 6730 # ankle, ~5 cm above the ground
        head_vertex_index = 6493 # neck top
        self.cut_offset = 0.03
                    
        general_subject_skinning_fields = []
        general_subject_bbox_extend = []
        general_subject_bbox_center = []
        body_loc_subjects = []

        smpl_ref_shape = []

        for idx, subject_name in general_subject_index.items():
            # load skinning field
            general_subject_skinning_fields.append(torch.tensor(np.load(subject_field_base_path + subject_name + '_cano_lbs_weights_grid_float32.npy')))
            
            # load corner point of minimal shape
            subject_mesh = trimesh.load_mesh(subject_field_base_path + subject_name + '_minimal_cpose.ply')
            subject_shape = np.array(subject_mesh.vertices)
            subject_face = np.array(subject_mesh.faces)
            smpl_ref_shape.append(torch.tensor(subject_shape))

            bbox_data_min = subject_shape.min(0)
            bbox_data_max = subject_shape.max(0)
            bbox_data_extend = (bbox_data_max - bbox_data_min).max()
            bbox_grid_extend = bbox_data_extend * 1.1
            center = (bbox_data_min + bbox_data_max) / 2

            general_subject_bbox_extend.append(torch.tensor(bbox_grid_extend).float())
            general_subject_bbox_center.append(torch.tensor(center).float())

            # get body part location
            body_loc_subjects.append(torch.tensor(np.concatenate((subject_shape[left_hand_vertex_index, 0:1], subject_shape[right_hand_vertex_index, 0:1], subject_shape[left_foot_vertex_index, 1:2], subject_shape[right_foot_vertex_index, 1:2], subject_shape[head_vertex_index, 1:2]), axis=0)))

        self.subject_skinning_fields = torch.nn.Parameter(torch.stack(general_subject_skinning_fields, 0), requires_grad=False)   
        self.subject_bbox_extend = torch.nn.Parameter(torch.stack(general_subject_bbox_extend, 0), requires_grad=False)  
        self.subject_bbox_center = torch.nn.Parameter(torch.stack(general_subject_bbox_center, 0), requires_grad=False)  

        self.subject_body_loc = torch.nn.Parameter(torch.stack(body_loc_subjects, 0), requires_grad=False)   
        
        self.smpl_ref_verts = torch.stack(smpl_ref_shape, 0)
        self.smpl_ref_faces = torch.tensor(subject_face)

    @staticmethod
    def inv_transform_v(v, scale_grid, transl):
            """
            v: [b, n, 3]
            """
            v = v - transl[:, None, :]
            v = v / scale_grid[:, None, None]
            v = v * 2

            return v


    @staticmethod
    def get_w(p_xc, p_grid=1):
        n_batch, n_point, n_dim = p_xc.shape

        if n_batch * n_point == 0:
            return p_xc

        x = F.grid_sample(p_grid,
                            p_xc[:, None, None, :, :], 
                            align_corners=False,
                            padding_mode='border')[:, :, 0, 0]  

        return x 


    def get_batch_body_loc(self, subject_garment_idx):

        subject_field_idx = torch.zeros_like(subject_garment_idx)
        for i in range(len(subject_garment_idx)):
            subject_field_idx[i] = self.general_subject_index_numer[subject_garment_idx[i].item()]

        return self.subject_body_loc[subject_field_idx]
    
    def query_skinning_weights(self, query_location, subject_garment_idx):
        # a function used in root finding
        # recompute the subject garment dict to subject dict
        subject_field_idx = torch.zeros_like(subject_garment_idx)
        for i in range(len(subject_garment_idx)):
            subject_field_idx[i] = self.general_subject_index_numer[subject_garment_idx[i].item()]

        weights_grid = self.subject_skinning_fields[subject_field_idx]
        bbox_grid_extend = self.subject_bbox_extend[subject_field_idx]
        bbox_grid_center = self.subject_bbox_center[subject_field_idx]

        v_cano_in_grid_coords = self.inv_transform_v(query_location.permute(0, 2, 1), bbox_grid_extend, bbox_grid_center)
        sw = self.get_w(v_cano_in_grid_coords, weights_grid) # [B, N, 24]

        return sw


    def forward(self, query_location, subject_garment_idx, training_mode=True):
        # query skinning weights in the field in (B, 3, N)

        # recompute the subject garment dict to subject dict
        subject_field_idx = torch.zeros_like(subject_garment_idx)
        for i in range(len(subject_garment_idx)):
            subject_field_idx[i] = self.general_subject_index_numer[subject_garment_idx[i].item()]

        weights_grid = self.subject_skinning_fields[subject_field_idx]
        bbox_grid_extend = self.subject_bbox_extend[subject_field_idx]
        bbox_grid_center = self.subject_bbox_center[subject_field_idx]

        body_loc = self.subject_body_loc[subject_field_idx]

        if training_mode == False:
            self.hand_mask, self.feet_mask, self.head_mask = compute_smaple_on_body_mask_w_batch(query_location, self.cut_offset, body_loc)

        v_cano_in_grid_coords = self.inv_transform_v(query_location.permute(0, 2, 1), bbox_grid_extend, bbox_grid_center)
        sw = self.get_w(v_cano_in_grid_coords, weights_grid) # [B, N, 24]

        return {'skinning_weights': sw} 