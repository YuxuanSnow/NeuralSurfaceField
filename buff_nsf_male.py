from libs.global_variable import ROOT_DIR, position

# coarse template via inverse skinning of scan using SMPL skinning weights
import torch

torch.manual_seed(0)

import torch.nn.functional as F

from libs.skinning_functions import InvSkinModel_RotationOnly, SkinModel, InvSkinModel, SkinModel_RotationOnly

from models.person_diffused_skinning import SmoothDiffusedSkinningField 
from models.person_specific_feature import NSF_SurfaceVertsFeatures
from models.network import PoseEncoder, NeuralSurfaceDeformationField

from models.forward_passing import query_local_feature_skinning, geometry_manifold_neural_field, reposing_cano_points_fix_skinning

from dataloaders.dataloader_buff import DataLoader_Buff_depth

from tqdm import tqdm
from os.path import join, split
import numpy as np
import argparse
import os

from trainer.data_parallel import MyDataParallel

# from visualization.write_pcd import write_pcd

from trainer.basic_trainer_nsf import Basic_Trainer_nsf
from libs.sample import compute_smaple_on_body_mask_w_batch
from libs.data_io import save_result_ply

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

from pytorch3d.loss import (
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

from pytorch3d.io import save_obj, save_ply

from models.losses import chamfer_distance_s2m_m2s

class Trainer(Basic_Trainer_nsf):

    def animate(self, save_animation, poses_file_path, pretrained=None, checkpoint=None):
        epoch = self.load_checkpoint(path=pretrained, number=checkpoint)

        print('Animating with epoch {}'.format(epoch))

        poses_file = np.load(poses_file_path, allow_pickle=True).item()
        pose_all = poses_file['pose']
        transl_all = poses_file['transl']
        assert len(pose_all) == len(transl_all) 

        device = self.device

        with torch.no_grad():
            self.set_feat_training_mode(train_flag=False)

            for subj_garment in tqdm(self.val_dataset.subject_index_dict.keys()):
                feature_cube_idx = torch.tensor(self.val_dataset.subject_index_dict[subj_garment])[None].to(device)

                for n in tqdm(range(len(pose_all))):
                    device = self.device

                    pose = torch.from_numpy(pose_all[n])[None].to(device)

                    smpl_d_mesh = self.nsf_feature_surface.smpl_d_dense_mesh.to(self.device)[feature_cube_idx]
                    x_useless_sampled, _ = sample_points_from_meshes(smpl_d_mesh, num_samples=30000, return_normals=True)

                    coarse_corr = x_useless_sampled.permute(0, 2, 1).contiguous()

                    inputs = {'coarse_corr': coarse_corr,
                        'pose': pose,
                        'trans': torch.from_numpy(transl_all[n]).to(device),
                        'feature_cube_idx': feature_cube_idx}

                    logits = self.predict(inputs, train=False)

                    pred_cano_cloth_verts = logits['cano_cloth_points'].permute(0, 2, 1).contiguous()[:, 30000:]
                    pred_posed_cloth_verts = logits['posed_cloth_points'].permute(0, 2, 1).contiguous()[:, 30000:]
                    # get updated mesh
                    faces_new = self.nsf_feature_surface.smpl_d_dense_mesh.faces_padded().to(device)[feature_cube_idx]

                    subject = subj_garment.split('_')[0] 
                    garment = subj_garment.split('_')[1] 
                    save_folder = join(self.exp_path, save_animation + '_ep_{}'.format(epoch), subject, garment, str(n))

                    # to avoid [-1,-1,-1] invalid faces
                    face = faces_new[0][faces_new[0]!=torch.tensor([-1, -1, -1]).to(device)].reshape(-1, 3)

                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    save_ply(save_folder+'/posed_mesh.ply', verts=pred_posed_cloth_verts[0], faces=face)
                    save_ply(save_folder+'/cano_mesh.ply', verts=pred_cano_cloth_verts[0], faces=face)


    # test on seen subjects but unseen subject
    def test_model(self, save_name, num_samples=-1, pretrained=None, checkpoint=None):

        epoch = self.load_checkpoint(path=pretrained, number=checkpoint)

        print('Testing with epoch {}'.format(epoch))
        val_data_loader = self.val_dataset.get_loader(shuffle=False)
        
        with torch.no_grad():

            self.set_feat_training_mode(train_flag=False)
            for n, batch in enumerate(tqdm(val_data_loader)):

                device = self.device

                pose = batch.get('pose').to(device)
                feature_cube_idx = batch.get('feature_cube_idx').to(device)
                coarse_corr = batch.get('coarse_cano_points').to(device)

                inputs = {'coarse_corr': coarse_corr,
                        'pose': pose,
                        'trans': batch.get('trans').to(device).repeat(3, 1).permute(1, 0).contiguous(),
                        'feature_cube_idx': feature_cube_idx}

                logits = self.predict(inputs, train=False)

                on_body_mask = logits['on_body_mask'][:, :30000]

                pred_posed_cloth_points_corr = logits['posed_cloth_points'].permute(0, 2, 1).contiguous()[:, :30000] # [B, 30000, 3]
                pred_posed_normals_corr = logits['posed_cloth_normals'].permute(0, 2, 1).contiguous()[:, :30000] # [B, 30000, 3]
                pred_posed_cloth_verts = logits['posed_cloth_points'].permute(0, 2, 1).contiguous()[:, 30000:] # [B, ?, 3]
                pred_cano_cloth_verts = logits['cano_cloth_points'].permute(0, 2, 1).contiguous()[:, 30000:]

                gt_posed_cloth_points = batch.get('scan_points').to(device).permute(0, 2, 1).contiguous() # [B, 40000, 3]
                gt_posed_cloth_normals = batch.get('scan_normals').to(device).permute(0, 2, 1).contiguous() # [B, 40000, 3]

                scan_rot_normals = batch.get('scan_normals_rotated').permute(0,2,1)
                gt_posed_cloth_normals[scan_rot_normals[:, :, -1]<0] = -gt_posed_cloth_normals[scan_rot_normals[:, :, -1]<0]

                # get updated mesh
                faces_new = self.nsf_feature_surface.smpl_d_dense_mesh.faces_padded().to(device)[feature_cube_idx]

                add_root_rotation = True
                if add_root_rotation:
                    pose_root = batch.get('pose_root').to(device)
                    sw_root = torch.zeros((pred_posed_cloth_verts.shape[0], 24, pred_posed_cloth_verts.shape[1])).to(device).float()
                    sw_root[:, 0, :] = 1
                    trans_root = torch.tensor([0]).cuda().float()
                    # skin the root rotation
                    pred_posed_cloth_verts_rot = self.skinner(pred_posed_cloth_verts.permute(0, 2, 1).contiguous(), pose_root, sw_root, trans_root)['posed_cloth_points'].permute(0,2,1).contiguous()

                names = batch.get('path')

                for i in range(len(names)):
                    file_path = names[i]
                    subject = file_path.split('/')[position] # if local 9; if cluster 12
                    garment = split(file_path)[1].split('_')[0]
                    save_folder = join(self.exp_path, save_name + '_ep_{}'.format(epoch), subject, garment, names[0].split('/')[-1].split('.')[0])

                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    # save predicted reposed correspondence
                    save_result_ply(save_folder, points=pred_posed_cloth_points_corr[i], normals=pred_posed_normals_corr[i], gt=gt_posed_cloth_points[i][on_body_mask[i]], gt_normals=gt_posed_cloth_normals[i][on_body_mask[i]])

                    # to avoid [-1,-1,-1] invalid faces
                    face = faces_new[i][faces_new[i]!=torch.tensor([-1, -1, -1]).to(device)].reshape(-1, 3)
                    save_ply(save_folder+'/posed_mesh.ply', verts=pred_posed_cloth_verts[i], faces=face)
                    save_ply(save_folder+'/cano_mesh.ply', verts=pred_cano_cloth_verts[i], faces=face)
                    
                    if add_root_rotation:
                        save_ply(save_folder+'/posed_mesh_rot.ply', verts=pred_posed_cloth_verts_rot[i], faces=face)


    def predict(self, batch, animate=False, train=True):
        # set module status
        self.set_module_training_mode(train_flag=train)

        # Inputs are already on device
        subject_garment_id = batch.get('feature_cube_idx')
        coarse_corr = batch.get('coarse_corr')
        pose = batch.get('pose')
        trans = batch.get('trans')

        # Run network to predict correspondences and DF
        logits = {}

        smpl_d_mesh_all = self.nsf_feature_surface.smpl_d_dense_mesh.to(self.device)
        
        verts_c_sampled = smpl_d_mesh_all.verts_padded()[subject_garment_id].permute(0, 2, 1).contiguous()
        # [0:30000] coarse corr, [30000:] vertices
        x_c_coarse = torch.cat((coarse_corr, verts_c_sampled), dim=2)

        subject_field_idx = torch.zeros_like(subject_garment_id)
        for i in range(len(subject_garment_id)):
            subject_field_idx[i] = self.diffused_skinning_field.general_subject_index_numer[subject_garment_id[i].item()]
        body_loc = self.diffused_skinning_field.subject_body_loc[subject_field_idx]
        smpl_hand_mask, smpl_feet_mask, _ = compute_smaple_on_body_mask_w_batch(x_c_coarse, cut_offset=0.05, subject_loc=body_loc)
        not_on_body_mask = torch.logical_or(smpl_hand_mask, smpl_feet_mask)
        on_body_mask = ~not_on_body_mask

        # 2nd: query corresponding local feature
        feat_pose_loc, skinning_weights = query_local_feature_skinning(x_c_coarse, pose, subject_garment_id, self.nsf_feature_surface, self.pose_encoder)

        # 3rd: use neural field to predict pose-dependent canonical geometry from local features
        fine_cano_offset, fine_cano_normals = geometry_manifold_neural_field(feat_pose_loc, self.nsf_decoder)
        fine_cano_points = x_c_coarse + fine_cano_offset

        replace_hand_feet = True
        if replace_hand_feet:
            fine_cano_points.permute(0,2,1)[not_on_body_mask] = x_c_coarse.permute(0,2,1)[not_on_body_mask]

        posed_cloth_points, posed_cloth_normals = reposing_cano_points_fix_skinning(x_c_coarse, fine_cano_points, fine_cano_normals, pose, trans, subject_garment_id, self.diffused_skinning_field, skinner, skinner_normal, skinning_weights=skinning_weights)
        
        logits.update({ 'cano_cloth_displacements': fine_cano_offset,
                        'cano_cloth_points': fine_cano_points,
                        'cano_cloth_normals': fine_cano_normals,
                        'posed_cloth_points': posed_cloth_points,
                        'posed_cloth_normals': posed_cloth_normals,
                        'geometric_feat': feat_pose_loc,
                        "on_body_mask": on_body_mask})

        return logits


    def compute_loss(self, batch, weights, train=True, ssp_only=False):

        device = self.device

        pose = batch.get('pose').to(device)
        feature_cube_idx = batch.get('feature_cube_idx').to(device)
        trans = batch.get('trans').to(device).repeat(3, 1).permute(1, 0).contiguous()
        coarse_corr = batch.get('coarse_cano_points').to(device)

        inputs = {'coarse_corr': coarse_corr,
                  'pose': pose,
                  'trans': trans,
                  'feature_cube_idx': feature_cube_idx}

        logits = self.predict(inputs, train=train)

        on_body_mask = logits['on_body_mask'][:, :30000]

        pred_posed_cloth_points_corr = logits['posed_cloth_points'].permute(0, 2, 1).contiguous()[:, :30000] # [B, N, 3]
        pred_posed_normals_corr = logits['posed_cloth_normals'].permute(0, 2, 1).contiguous()[:, :30000] # [B, N, 3]
        pred_cano_cloth_displacements = logits['cano_cloth_displacements'].permute(0, 2, 1).contiguous()# [B, N+V, 3]
        pred_geometric_feature = logits['geometric_feat'].permute(0, 2, 1).contiguous()[:, :, :64]# [B, N+V, 64]
        
        # get updated mesh
        pred_cano_cloth_verts = logits['posed_cloth_points'].permute(0, 2, 1).contiguous()[:, 30000:] # [B, V, 3]
        faces_new = self.nsf_feature_surface.smpl_d_dense_mesh.faces_padded().to(device)[feature_cube_idx]
        new_meshes = Meshes(pred_cano_cloth_verts, faces_new)
        x_pred_sampled, x_pred_sampled_normal = sample_points_from_meshes(new_meshes, num_samples=30000, return_normals=True)

        # from visualization.write_pcd import write_pcd
        # write_pcd('visualization/test.pcd', x_c_coarse.permute(0, 2, 1)[0].detach().cpu().numpy(), x_c_coarse.permute(0, 2, 1)[0].detach().cpu().numpy())
        # write_pcd('visualization/test.pcd', x_pred_sampled[0].detach().cpu().numpy(), x_pred_sampled_normal[0].detach().cpu().numpy())

        loss_edge = mesh_edge_loss(new_meshes)
        # mesh normal consistency
        loss_normal = mesh_normal_consistency(new_meshes)
        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(new_meshes, method="uniform")

        gt_posed_cloth_points = batch.get('scan_points').to(device).permute(0, 2, 1).contiguous()
        gt_posed_cloth_normals = batch.get('scan_normals').to(device).permute(0, 2, 1).contiguous()
        # normals direction wrongly estimated
        scan_rot_normals = batch.get('scan_normals_rotated').permute(0,2,1)
        gt_posed_cloth_normals[scan_rot_normals[:, :, -1]<0] = -gt_posed_cloth_normals[scan_rot_normals[:, :, -1]<0]

        # chamfer distance
        s2m = 0
        for i in range(on_body_mask.shape[0]):
            s2m_, _, _, _ = chamfer_distance_s2m_m2s(x_pred_sampled, gt_posed_cloth_points[:, on_body_mask[i, :30000]], x_normals=x_pred_sampled_normal, y_normals=gt_posed_cloth_normals[:, on_body_mask[i, :30000]])
            s2m += s2m_
        s2m /= on_body_mask.shape[0]

        # v2v distance
        v2v_dist_posed_cloth = F.mse_loss(pred_posed_cloth_points_corr[on_body_mask], gt_posed_cloth_points[on_body_mask], reduction='none').sum(-1).mean()
        normal_loss_posed_cloth = F.l1_loss(pred_posed_normals_corr[on_body_mask], gt_posed_cloth_normals[on_body_mask], reduction='none').sum(-1).mean()

        rgl_displacement = torch.mean((pred_cano_cloth_displacements[:, 30000:] ** 2).sum(-1))
        rgl_latent = torch.mean(pred_geometric_feature ** 2)

        w_v2v, _, w_normal, w_rgl, w_latent_rgl = weights

        loss = {}

        loss.update({'v2v_posed': v2v_dist_posed_cloth * w_v2v,
                    'normal_posed': normal_loss_posed_cloth * w_normal,
                    'chamfer': s2m * w_v2v,
                    # 'chamfer_normal': (s2m_normal + m2s_normal) * w_normal,
                    'mesh_utils': 1e3 * loss_edge + 1e1 * loss_normal + 1e2 * loss_laplacian,
                    'rgl': rgl_displacement * w_rgl * 10,
                    # 'latent_rgl': rgl_latent * w_latent_rgl,
                    # 'smooth_rgl': rgl_feature_space_smoothness * w_latent_rgl,
                    # 'edr': edr_regularization * 1e5
                    })
                    
        return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Model')
    # experiment id for folder suffix
    parser.add_argument('-exp_id', '--exp_id', type=str)
    parser.add_argument('-pretrained_exp', '--pretrained_exp', type=str)
    parser.add_argument('-batch_size', '--batch_size', default=8, type=int)
    parser.add_argument('-split_file', '--split_file', type=str)
    parser.add_argument('-epochs', '--epochs', default=300, type=int)
    # val, ft, pose_track, animate, detail_recon
    parser.add_argument('-mode', '--mode', default='train', type=str)
    parser.add_argument('-fusion_shape', '--fusion_shape', default='smpld_sub', type=str)
    parser.add_argument('-save_name', '--save_name', default='smpld_sub', type=str)

    args = parser.parse_args()

    args.subject_paths = [
        ROOT_DIR + 'experiments/PoseImplicit_exp_id_{}/00032/shortlong'.format(args.exp_id),
        ROOT_DIR + 'experiments/PoseImplicit_exp_id_{}/00032/shortshort'.format(args.exp_id),
        ROOT_DIR + 'experiments/PoseImplicit_exp_id_{}/00096/shortlong'.format(args.exp_id),
        ROOT_DIR + 'experiments/PoseImplicit_exp_id_{}/00096/shortshort'.format(args.exp_id)
    ]

    args.pretrained_feature_exp_path = [
        ROOT_DIR + 'experiments/PoseImplicit_exp_id_{}/00032/shortlong'.format(args.pretrained_exp),
        ROOT_DIR + 'experiments/PoseImplicit_exp_id_{}/00032/shortshort'.format(args.pretrained_exp),
        ROOT_DIR + 'experiments/PoseImplicit_exp_id_{}/00096/shortlong'.format(args.pretrained_exp),
        ROOT_DIR + 'experiments/PoseImplicit_exp_id_{}/00096/shortshort'.format(args.pretrained_exp)
    ]

    args.num_subjects = len(args.subject_paths)

    subject_index_dict = {}
    subject_index_dict.update({"00032_shortlong": 0,
                               "00032_shortshort": 1,
                               "00096_shortlong": 2,
                               "00096_shortshort": 3})

    # multi subj query: for one subject with different garments, only use one skinning field
    general_subject_index = {}
    general_subject_index_numer = {}
    for key, value in subject_index_dict.items():
        if key.startswith('00032'):
            general_subject_index.update({'{}'.format(value): '00032'})
            general_subject_index_numer.update({value: 0})
        if key.startswith('00096'):
            general_subject_index.update({'{}'.format(value): '00096'})
            general_subject_index_numer.update({value: 1})

    print("Split file: ", args.split_file)

    exp_name = 'PoseImplicit_exp_id_{}'.format(args.exp_id)
    pretrained_exp_name = 'PoseImplicit_exp_id_{}'.format(args.pretrained_exp)

    pretrained_module_dict = {
        'pose_encoder': pretrained_exp_name,
        'nsf_decoder': pretrained_exp_name
    }
    
    # for local feature query based on SMPL-D mesh template and decoding local feature to the pose-dependent offset
    nsf_feature_surface = MyDataParallel(NSF_SurfaceVertsFeatures(args.num_subjects, args.subject_paths, pretrained_feature_exp=args.pretrained_feature_exp_path, feat_dim=64, data='BUFF', fusion_shape_mode=args.fusion_shape))
    pose_encoder = MyDataParallel(PoseEncoder(in_features=72, out_features=24))
    # 64: point-wise feature
    # 24: pose features (global)
    # 3: query location
    nsf_decoder = MyDataParallel(NeuralSurfaceDeformationField(feat_in=64+24+3, hidden_sz=256)) # conditioned on pose 

    # for forward skinning
    inv_skinner = MyDataParallel(InvSkinModel(gender='male'))
    inv_skinner_normal = MyDataParallel(InvSkinModel_RotationOnly(gender='male'))
    skinner = MyDataParallel(SkinModel(gender='male'))
    skinner_normal = MyDataParallel(SkinModel_RotationOnly(gender='male'))

    # smoothly diffused skinning field
    precomputed_skinning_field_base_path = ROOT_DIR + 'diffused_smpl_skinning_field/'
    diffused_skinning_field = MyDataParallel(SmoothDiffusedSkinningField(subject_field_base_path=precomputed_skinning_field_base_path, general_subject_index=general_subject_index, general_subject_index_numer=general_subject_index_numer))
    
    module_dict = {
        'pose_encoder': pose_encoder,
        'diffused_skinning_field': diffused_skinning_field,
        'nsf_feature_surface': nsf_feature_surface, # local feature query
        'nsf_decoder': nsf_decoder,
        'inv_skinner': inv_skinner,
        'inv_skinner_normal': inv_skinner_normal,
        'skinner': skinner,
        'skinner_normal': skinner_normal,
    }

    if args.mode == 'train':
        train_dataset = DataLoader_Buff_depth(mode='train', nsf_cano_available=True, batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict)  
        val_dataset = DataLoader_Buff_depth(mode='val', nsf_cano_available=True, batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict, num_points=30000)  
        trainer = Trainer(module_dict, pretrained_module_dict, device=torch.device("cuda"), train_dataset=train_dataset, val_dataset=val_dataset, exp_name=exp_name)

        trainer.train_model(args.epochs)

    if args.mode == 'test':

        val_dataset = DataLoader_Buff_depth(mode='val', nsf_cano_available=True, batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict, num_points=30000)  
        trainer = Trainer(module_dict, pretrained_module_dict, device=torch.device("cuda"), val_dataset=val_dataset, exp_name=exp_name)

        trainer.test_model(args.save_name)

    if args.mode == 'fine_tune':

        val_dataset = DataLoader_Buff_depth(mode='val', nsf_cano_available=True, batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict, num_points=30000)  
        trainer = Trainer(module_dict, pretrained_module_dict, device=torch.device("cuda"), val_dataset=val_dataset, exp_name=exp_name)

        trainer.fine_tune_model(args.epochs)

    if args.mode == 'animate':

        val_dataset = DataLoader_Buff_depth(mode='val', nsf_cano_available=True, batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict, num_points=30000)  
        trainer = Trainer(module_dict, pretrained_module_dict, device=torch.device("cuda"), val_dataset=val_dataset, exp_name=exp_name)

        trainer.animate('animate_'+args.save_name, poses_file_path='data_animation/aist_poses.npy')

