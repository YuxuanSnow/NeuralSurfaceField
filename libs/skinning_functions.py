import torch
from torch import nn
import os
import numpy as np

from libs.serialization import ready_arguments
from libs.smplpytorch import (th_posemap_axisang, th_with_zeros, th_pack, make_list, subtract_flat_id)

from libs.global_variable import ROOT_DIR

# Forward Skiningg & Inverse Skining model with translation (for points) or not (for rotation)
class SkinModel(nn.Module):
    def __init__(self, gender='male'):
            super(SkinModel, self).__init__()
            self.load_smpl_skeleton(model_root= ROOT_DIR+'/smpl_model', gender=gender)

    def load_smpl_skeleton(self, model_root, gender):
        model_path = None
        if gender == 'male':
            model_path = os.path.join(model_root, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        elif gender == 'female':
            model_path = os.path.join(model_root, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        else:
            assert False, "Given gender not available"
        smpl_data = ready_arguments(model_path)
        self.th_v_template = torch.nn.Parameter(torch.Tensor(np.array(smpl_data['v_template'].r)).unsqueeze(0), requires_grad=False)
        self.th_J_regressor = torch.nn.Parameter(torch.Tensor(np.array(smpl_data['J_regressor'].toarray())), requires_grad=False)

        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents
        self.num_joints = len(parents)  # 24


    def get_root_transformation(self, th_pose_axisang):

        batch_size = th_pose_axisang.shape[0]
        th_pose_rotmat = th_posemap_axisang(th_pose_axisang)
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3) 

        th_v = torch.tile(self.th_v_template, (batch_size, 1, 1)) 
        th_j = torch.matmul(self.th_J_regressor, th_v)

        root_j = torch.zeros_like(th_j[:, 0, :].contiguous().view(batch_size, 3, 1)) 

        root_transformation = th_with_zeros(torch.cat([root_rot, root_j], 2))

        return root_transformation


    def compute_smpl_skeleton(self, th_pose_axisang):
        batch_size = th_pose_axisang.shape[0]
        th_pose_rotmat = th_posemap_axisang(th_pose_axisang) 
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3) 
        th_pose_rotmat = th_pose_rotmat[:, 9:]

        th_v = torch.tile(self.th_v_template, (batch_size, 1, 1)) 
        th_j = torch.matmul(self.th_J_regressor, th_v)  

        th_results = []
        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        for i in range(self.num_joints - 1):
            i_val = int(i + 1)
            joint_rot = th_pose_rotmat[:, (i_val - 1) * 9:i_val * 9].contiguous().view(batch_size, 3, 3) 
            joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1) 
            parent = make_list(self.kintree_parents)[i_val]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1) 
            joint_rel_transform = th_with_zeros(torch.cat([joint_rot, joint_j - parent_j], 2))
            th_results.append(torch.matmul(th_results[parent], joint_rel_transform))

        th_results2 = torch.zeros((batch_size, 4, 4, self.num_joints),
                                  dtype=root_j.dtype,
                                  device=root_j.device)

        for i in range(self.num_joints):
            padd_zero = torch.zeros(1, dtype=th_j.dtype, device=th_j.device)
            joint_j = torch.cat(
                [th_j[:, i],
                 padd_zero.view(1, 1).repeat(batch_size, 1)], 1) 
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2))
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp) 

        return th_results2

    def forward(self, points, pose, skinning_weights, trans=None):

        batch_size = pose.shape[0]
        if not torch.is_tensor(trans):
            trans = torch.zeros((batch_size, 3), device=points.device)

        th_results2 = self.compute_smpl_skeleton(pose)

        p_T = torch.bmm(th_results2.view(-1, 16, 24), skinning_weights).view(batch_size, 4, 4, -1)
        p_rest_shape_h = torch.cat([
            points,
            torch.ones((batch_size, 1, points.shape[2]),
                       dtype=p_T.dtype,
                       device=p_T.device),
        ], 1) 

        p_verts = (p_T * p_rest_shape_h.unsqueeze(1)).sum(2)
        p_verts = p_verts[:, :3, :] + trans[..., None]

        return {'posed_cloth_points': p_verts} 


class InvSkinModel(nn.Module):
    def __init__(self, gender='male'):
            super(InvSkinModel, self).__init__()
            self.load_smpl_skeleton(model_root= ROOT_DIR+'/smpl_model', gender=gender)

    def load_smpl_skeleton(self, model_root, gender):
        model_path = None
        if gender == 'male':
            model_path = os.path.join(model_root, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        elif gender == 'female':
            model_path = os.path.join(model_root, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        else:
            assert False, "Given gender not available"
        smpl_data = ready_arguments(model_path)
        self.th_v_template = torch.nn.Parameter(torch.Tensor(np.array(smpl_data['v_template'].r)).unsqueeze(0), requires_grad=False)
        self.th_J_regressor = torch.nn.Parameter(torch.Tensor(np.array(smpl_data['J_regressor'].toarray())), requires_grad=False)

        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents
        self.num_joints = len(parents)  # 24


    def compute_smpl_skeleton(self, th_pose_axisang):
 
        batch_size = th_pose_axisang.shape[0]

        th_pose_rotmat = th_posemap_axisang(th_pose_axisang) 
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3) 
        th_pose_rotmat = th_pose_rotmat[:, 9:] 

        th_v = torch.tile(self.th_v_template, (batch_size, 1, 1)) 
        th_j = torch.matmul(self.th_J_regressor, th_v) 

        th_results = []
        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1) 
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2))) 

        for i in range(self.num_joints - 1):
            i_val = int(i + 1)
            joint_rot = th_pose_rotmat[:, (i_val - 1) * 9:i_val * 9].contiguous().view(batch_size, 3, 3) 
            joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1) 
            parent = make_list(self.kintree_parents)[i_val]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1) 
            joint_rel_transform = th_with_zeros(torch.cat([joint_rot, joint_j - parent_j], 2))
            th_results.append(torch.matmul(th_results[parent], joint_rel_transform))

        th_results2 = torch.zeros((batch_size, 4, 4, self.num_joints),
                                  dtype=root_j.dtype,
                                  device=root_j.device)

        for i in range(self.num_joints):
            padd_zero = torch.zeros(1, dtype=th_j.dtype, device=th_j.device)
            joint_j = torch.cat(
                [th_j[:, i],
                 padd_zero.view(1, 1).repeat(batch_size, 1)], 1)
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2)) 
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp)

        return th_results2

    def forward(self, points, pose, skinning_weights, trans=None):

        batch_size = pose.shape[0]
        if not torch.is_tensor(trans):
            trans = torch.zeros((batch_size, 3), device=points.device)

        th_results2_forward = self.compute_smpl_skeleton(pose)
        th_results2 = th_results2_forward.permute(0, 3, 1, 2).inverse().permute(0, 2, 3, 1).contiguous()

        p_T = torch.bmm(th_results2.view(-1, 16, 24), skinning_weights).view(batch_size, 4, 4, -1)
        p_rest_shape_h = torch.cat([
            points,
            torch.ones((batch_size, 1, points.shape[2]),
                       dtype=p_T.dtype,
                       device=p_T.device),
        ], 1) 

        p_verts = (p_T * p_rest_shape_h.unsqueeze(1)).sum(2)
        p_verts = p_verts[:, :3, :] - trans[..., None]

        return {'cano_cloth_points': p_verts} 
    

class SkinModel_RotationOnly(nn.Module):
    # rotation only skinning
    def __init__(self, gender='male'):
            super(SkinModel_RotationOnly, self).__init__()
            self.load_smpl_skeleton(model_root= ROOT_DIR+'/smpl_model', gender=gender)

    def load_smpl_skeleton(self, model_root, gender):
        model_path = None
        if gender == 'male':
            model_path = os.path.join(model_root, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        elif gender == 'female':
            model_path = os.path.join(model_root, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        else:
            assert False, "Given gender not available"
        smpl_data = ready_arguments(model_path)
        self.th_v_template = torch.nn.Parameter(torch.Tensor(np.array(smpl_data['v_template'].r)).unsqueeze(0), requires_grad=False)
        self.th_J_regressor = torch.nn.Parameter(torch.Tensor(np.array(smpl_data['J_regressor'].toarray())), requires_grad=False)

        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents
        self.num_joints = len(parents)  # 24


    def compute_smpl_skeleton(self, th_pose_axisang):

        batch_size = th_pose_axisang.shape[0]
        th_pose_rotmat = th_posemap_axisang(th_pose_axisang)
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3)
        th_pose_rotmat = th_pose_rotmat[:, 9:]
        th_pose_map = subtract_flat_id(th_pose_rotmat)

        th_v = torch.tile(self.th_v_template, (batch_size, 1, 1))  
        th_j = torch.matmul(self.th_J_regressor, th_v)

        th_results = []

        th_results.append(root_rot) 

        for i in range(self.num_joints - 1):
            i_val = int(i + 1)
            joint_rot = th_pose_rotmat[:, (i_val - 1) * 9:i_val * 9].contiguous().view(batch_size, 3, 3)
            parent = make_list(self.kintree_parents)[i_val]

            
            th_results.append(torch.matmul(th_results[parent], joint_rot))

        th_results2 = torch.zeros((batch_size, 3, 3, self.num_joints),
                                  dtype=root_rot.dtype,
                                  device=root_rot.device)

        for i in range(self.num_joints):
            th_results2[:, :, :, i] = th_results[i]

        return th_results2

    def forward(self, normals, pose, skinning_weights):

        batch_size = pose.shape[0]
        th_results2 = self.compute_smpl_skeleton(pose)

        n_T = torch.bmm(th_results2.view(-1, 9, 24), skinning_weights).view(batch_size, 3, 3, -1)
        n_verts = (n_T * normals.unsqueeze(1)).sum(2)

        return {'posed_cloth_normals': n_verts} 


class InvSkinModel_RotationOnly(nn.Module):
    # rotation only skinning
    def __init__(self, gender='male'):
            super(InvSkinModel_RotationOnly, self).__init__()
            self.load_smpl_skeleton(model_root= ROOT_DIR+'/smpl_model', gender=gender)

    def load_smpl_skeleton(self, model_root, gender):
        model_path = None
        if gender == 'male':
            model_path = os.path.join(model_root, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        elif gender == 'female':
            model_path = os.path.join(model_root, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        else:
            assert False, "Given gender not available"
        smpl_data = ready_arguments(model_path)
        self.th_v_template = torch.nn.Parameter(torch.Tensor(np.array(smpl_data['v_template'].r)).unsqueeze(0), requires_grad=False)
        self.th_J_regressor = torch.nn.Parameter(torch.Tensor(np.array(smpl_data['J_regressor'].toarray())), requires_grad=False)

        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents
        self.num_joints = len(parents)  # 24


    def compute_smpl_skeleton(self, th_pose_axisang):

        batch_size = th_pose_axisang.shape[0]
        th_pose_rotmat = th_posemap_axisang(th_pose_axisang)
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3)
        th_pose_rotmat = th_pose_rotmat[:, 9:]
        th_pose_map = subtract_flat_id(th_pose_rotmat)

        th_v = torch.tile(self.th_v_template, (batch_size, 1, 1))   
        th_j = torch.matmul(self.th_J_regressor, th_v)

        th_results = []

        th_results.append(root_rot) 

        for i in range(self.num_joints - 1):
            i_val = int(i + 1)
            joint_rot = th_pose_rotmat[:, (i_val - 1) * 9:i_val * 9].contiguous().view(batch_size, 3, 3)

            parent = make_list(self.kintree_parents)[i_val]
            th_results.append(torch.matmul(th_results[parent], joint_rot))

        th_results2 = torch.zeros((batch_size, 3, 3, self.num_joints),
                                  dtype=root_rot.dtype,
                                  device=root_rot.device)

        for i in range(self.num_joints):
            th_results2[:, :, :, i] = th_results[i]

        return th_results2

    def forward(self, normals, pose, skinning_weights):

        batch_size = pose.shape[0]

        th_results2_forward = self.compute_smpl_skeleton(pose)
        th_results2 = th_results2_forward.permute(0, 3, 1, 2).inverse().permute(0, 2, 3, 1).contiguous()

        n_T = torch.bmm(th_results2.view(-1, 9, 24), skinning_weights).view(batch_size, 3, 3, -1)
        n_verts = (n_T * normals.unsqueeze(1)).sum(2)

        return {'cano_cloth_normals': n_verts} 