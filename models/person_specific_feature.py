from libs.global_variable import ROOT_DIR

import torch
from torch import nn

import os
from os.path import exists, split, join
import numpy as np
from glob import glob

from pytorch3d.structures.meshes import Meshes
from pytorch3d.io import load_ply

from libs.barycentric_corr_finding import face_vertices, point_to_mesh_distance, barycentric_coordinates_of_projection

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

# load smpl + d coarse template and define features 
class NSF_SurfaceVertsFeatures(nn.Module):
    def __init__(self, num_subjects, subject_paths, pretrained_feature_exp=None, feat_dim=16, data='BUFF', fusion_shape_mode='smpld_sub'):
        """
        Person specific features, represented in mesh vertices of the coarse shape.
        subject_paths: list of paths to each subject's specific features.
        add vertex based RGB color
        """
        super(NSF_SurfaceVertsFeatures, self).__init__()

        self.dim = feat_dim
        self.data_set = data
        self.fusion_shape_mode = fusion_shape_mode

        assert num_subjects == len(subject_paths)
        self.subject_paths = subject_paths
        self.pretrained_feature_paths = pretrained_feature_exp
        
        # One times subdivided dense SMPL-D mesh, [B, N, 3] & [B, F, 3]
        smpld_verts, smpld_faces = self.load_smpld()
        self.verts_num = smpld_verts.shape[1]
        self.smpl_d_mesh = Meshes(verts=smpld_verts, faces=smpld_faces)

        smpld_dense_verts, smpld_dense_faces = self.load_smpld_dense()
        self.smpl_d_dense_mesh = Meshes(verts=smpld_dense_verts, faces=smpld_dense_faces)

        # [num_subj, F, 3, 3]
        self.face_verts_loc = face_vertices(self.smpl_d_mesh.verts_padded(), self.smpl_d_mesh.faces_padded())

        epoch = self.load_features(pretrained_feature_exp)

    def load_smpld(self):

        if self.data_set == 'CAPE':
            base_smpld_path = ROOT_DIR + 'Data/CAPE/Fusion_shape'
        elif self.data_set == 'BUFF':
            base_smpld_path = ROOT_DIR + 'Data/BuFF/Fusion_shape'

        subj_smpld_verts = []
        subj_smpld_faces = []

        for n, sp in enumerate(self.subject_paths):
            garment = sp.split('/')[-1]
            subject = sp.split('/')[-2]

            print("Loading feature space manifold template")
            subject_smpld_path = join(base_smpld_path, 'smpl_D_'+subject+'_'+garment+'.ply')
            file = load_ply(f=subject_smpld_path)
            verts = file[0].float()
            faces = file[1].float()

            subj_smpld_verts.append(verts)
            subj_smpld_faces.append(faces)
            
        smpl_verts = torch.stack(subj_smpld_verts)
        smpl_faces = torch.stack(subj_smpld_faces)

        return smpl_verts, smpl_faces
    
    def load_smpld_dense(self):
        
        if self.data_set == 'CAPE':
            base_smpld_path = ROOT_DIR + 'Data/CAPE/Fusion_shape'
        elif self.data_set == 'BUFF':
            base_smpld_path = ROOT_DIR + 'Data/BuFF/Fusion_shape'

        subj_smpld_verts = []
        subj_smpld_faces = []

        for n, sp in enumerate(self.subject_paths):
            garment = sp.split('/')[-1]
            subject = sp.split('/')[-2]

            # subject_smpld_path = join(base_smpld_path, subject, garment, 'smpl_D_'+subject+'_'+garment+'.ply')
            if self.fusion_shape_mode == 'smpld':
                subject_smpld_path = join(base_smpld_path, 'smpl_D_'+subject+'_'+garment+'.ply')
            elif self.fusion_shape_mode == 'smpld_sub':
                subject_smpld_path = join(base_smpld_path, 'smpl_D_'+subject+'_'+garment+'_subdivided.ply')
            elif self.fusion_shape_mode == 'mc_128':
                subject_smpld_path = join(base_smpld_path, 'MC_'+subject+'_'+garment+'_128.ply')
            elif self.fusion_shape_mode == 'mc_256':
                subject_smpld_path = join(base_smpld_path, 'MC_'+subject+'_'+garment+'_256.ply')
            elif self.fusion_shape_mode == 'mc_512':
                subject_smpld_path = join(base_smpld_path, 'MC_'+subject+'_'+garment+'_512.ply')
            else:
                assert False, "Unknown mode for fusion shape"

            file = load_ply(f=subject_smpld_path)
            verts = file[0].float()
            faces = file[1].float()

            subj_smpld_verts.append(verts)
            subj_smpld_faces.append(faces)

        # remove torch stack for heterogous meshes (in case of MC)
        smpl_verts = subj_smpld_verts
        smpl_faces = subj_smpld_faces

        return smpl_verts, smpl_faces
    
    def save_features(self, epoch):
        for n, sp in enumerate(self.subject_paths):
            garment = sp.split('/')[-1]
            subject = sp.split('/')[-2]
            subject_garment = subject + garment
            path_feat = join(sp, 'feature_{}.npy'.format(epoch))
            if not exists(sp):
                os.makedirs(sp)
            np.save(path_feat, self.features[n].detach().cpu().numpy())
            print('Saved {} specific NSF feature at epoch {}'.format(subject_garment, epoch))

    def load_features(self, pretrained_feature_path):
        
        self.features = []
        if pretrained_feature_path is None:
            print('Not use pretrained feature cube')
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
                        if split(path)[1].startswith('feature'):
                            index_list.append(split(path)[1][:-4].split('_')[1])
                    checkpoints = index_list
                    checkpoints = np.array(index_list, dtype=int)
                    checkpoints = np.sort(checkpoints)
                    path = join(sp, 'feature_{}.npy'.format(checkpoints[-1]))

                    temp = np.load(path)
                    self.features.append(torch.tensor(temp))
                    flag = max(flag, checkpoints[-1])

                    print('Load {} specific feature model at epoch {}'.format(subject_garment, flag))

            if flag==-1:
                print('No {} specific features found. Use initialized feature mesh.'.format(subject_garment))
                temp = torch.ones(self.dim+3, self.verts_num)*0.5 # each vertex has the dimension of feature and 3 for RGB
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
                        if split(path)[1].startswith('feature'):
                            index_list.append(split(path)[1][:-4].split('_')[1])
                    checkpoints = index_list
                    checkpoints = np.array(index_list, dtype=int)
                    checkpoints = np.sort(checkpoints)
                    path = join(pretrained_sp, 'feature_{}.npy'.format(checkpoints[-1]))

                    temp = np.load(path)
                    self.features.append(torch.tensor(temp))
                    flag = max(flag, checkpoints[-1])

                    print('Load {} pretrained specific feature model at epoch {}'.format(subject_garment, flag))

            if flag==-1:
                print('No {} specific features found.'.format(subject_garment))
                assert(False)
                
        self.features = torch.nn.Parameter(torch.stack(self.features, 0), requires_grad=True)    
        return flag

    def query_verts_skinning(self, diffused_skinning_field, device):
        
        # smpld skinning weights
        query_verts = self.smpl_d_mesh.verts_padded().to(device) # [subj, V, 3]
        verts_skinning_weights = []
        for i in range(query_verts.shape[0]):
            query_verts_subj = query_verts[i]
            verts_skinning_weights_subj = diffused_skinning_field(query_verts[i:i+1].permute(0,2,1), torch.tensor([i]))['skinning_weights'] # [1, 24, V]
            verts_skinning_weights.append(verts_skinning_weights_subj)
        self.verts_skinning = torch.stack(verts_skinning_weights, dim=0).squeeze(dim=1) # only squeeze the related dimension

        # smpld dense skinning weights
        query_verts_dense = self.smpl_d_dense_mesh.verts_padded().to(device) # [subj, V, 3]
        verts_skinning_weights_dense = []
        for i in range(query_verts_dense.shape[0]):
            query_verts_subj_dense = query_verts_dense[i]
            verts_skinning_weights_subj_dense = diffused_skinning_field(query_verts_dense[i:i+1].permute(0,2,1), torch.tensor([i]))['skinning_weights'] # [1, 24, V]
            verts_skinning_weights_dense.append(verts_skinning_weights_subj_dense)
        self.verts_skinning_dense = torch.stack(verts_skinning_weights_dense, dim=0).squeeze(dim=1) # only squeeze the related dimension

    def forward(self, points=None, subject_garment_id=None, get_sw=False):
        """
        points: correspondences to canonical space. Ideally on the mesh model. batch x 3 x points
        subject: list of subect paths. same as subject_paths.
        """
        shape = points.shape
        query_points = points.permute(0, 2, 1).contiguous()

        features = self.features[:, :self.dim, :] # first dimensions for feature

        assert(features.shape[1] == self.dim)

        # [B, F, 3, 3]
        face_verts_loc_selected = self.face_verts_loc.to(points.device)[subject_garment_id].contiguous()
        # [B, F, 3, dim]
        face_verts_feat = face_vertices(features.permute(0, 2, 1).contiguous(), self.smpl_d_mesh.faces_padded().to(points.device))
        face_verts_feat_selected = face_verts_feat[subject_garment_id].contiguous()
        
        # find closest face index from points to dense smpl mesh
        residues, pts_ind, _ = point_to_mesh_distance(query_points, face_verts_loc_selected)

        # get the cloest triangles on ref smpl mesh, [BxV, 3, 3]
        closest_triangles = torch.gather(face_verts_loc_selected, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
        # calculate closest point in the triangle, [BxV, 3]
        bary_weights = barycentric_coordinates_of_projection(query_points.view(-1, 3), closest_triangles)

        # feature face tensor, [BxV, 3, dim]
        face_feat_all = torch.gather(face_verts_feat_selected, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, self.dim)).view(-1, 3, self.dim)

        # aggregate feature using barycentroc coordinate
        feature_bary = (face_feat_all * bary_weights[:, :, None]).sum(1).unsqueeze(0)  # [1, BxV, dim]
        feature_ = feature_bary.reshape(shape[0], -1, self.dim)
        
        feat = feature_.permute(0, 2, 1).contiguous()

        if get_sw == False:
            return feat # batch x dim x num_points
        else:
            # [B, F, 3, dim]
            face_verts_skinning = face_vertices(self.verts_skinning.permute(0, 2, 1).contiguous().to(points.device), self.smpl_d_mesh.faces_padded().to(points.device))
            face_verts_skinning_selected = face_verts_skinning[subject_garment_id].contiguous()

            # feature face tensor, [BxV, 3, 24]
            face_skinning_all = torch.gather(face_verts_skinning_selected, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 24)).view(-1, 3, 24)

            # aggregate feature using barycentroc coordinate
            skinning_bary = (face_skinning_all * bary_weights[:, :, None]).sum(1).unsqueeze(0)  # [1, BxV, 24]
            skinning_ = skinning_bary.reshape(shape[0], -1, 24)
            
            skinning = skinning_.permute(0, 2, 1).contiguous()

            return feat, skinning