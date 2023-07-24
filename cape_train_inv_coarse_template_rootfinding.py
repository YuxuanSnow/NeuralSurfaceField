# coarse template via inverse skinning of scan using SMPL skinning weights
import torch
import torch.nn.functional as F

from libs.skinning_functions import InvSkinModel, InvSkinModel_RotationOnly, SkinModel, SkinModel_RotationOnly
from models.person_specific_feature import SubjectGlobalLatentFeature
from models.person_diffused_skinning import SmoothDiffusedSkinningField 

from dataloaders.dataloader_cape import DataLoader_Cape_depth

from trainer.data_parallel import MyDataParallel
from trainer.basic_trainer_sdf import Basic_Trainer_sdf

from tqdm import tqdm
from os.path import join, split
import numpy as np
import argparse
import os

from models.igr_sdf_net import IGRSDFNet, condition_reconstruction
from libs.smpl_paths import SmplPaths

from models.basic_trainer import *
from utils.save_mesh_io import *
from utils.barycentric_corr_finding import * 
from models.canonicalization import search

import open3d as o3d

class Trainer(Basic_Trainer_sdf):
    
    # test on seen subjects but unseen subject
    def test_model(self, save_name, num_samples=-1, pretrained=None, checkpoint=None):

        epoch = self.load_checkpoint(path=pretrained, number=checkpoint)

        print('Testing with epoch {}'.format(epoch))
        val_data_loader = self.val_dataset.get_loader(shuffle=False)

        test_s2m, test_m2s, test_lnormal = 0, 0, 0
        with torch.no_grad():

            self.set_feat_training_mode(train_flag=True)
            count = 0
            for n, batch in enumerate(tqdm(val_data_loader)):

                device = self.device

                self.set_module_training_mode(train_flag=False)

                feature_cube_idx = batch.get('feature_cube_idx').to(device)

                feature = self.subject_global_latent.get_feature(feature_cube_idx)

                bbox_min = np.stack([-1.5, -1.5, -1.5], 0).astype(np.float32)
                bbox_max = np.stack([1.5, 1.5, 1.5], 0).astype(np.float32)

                verts, faces, normals, values = condition_reconstruction(self.conditional_ndf, device, feature, resolution=512, thresh=0.00001, b_min=bbox_min, b_max=bbox_max, texture_net=None)

                names = batch.get('path')

                for i in range(len(names)):
                    file_path = names[i]
                    name = split(file_path)[1]
                    front_or_back = file_path.split('/')[-6]
                    dataset = split(split(file_path)[0])[1]
                    save_folder = join(self.exp_path, save_name + '_ep_{}'.format(epoch), dataset, front_or_back, name.split('.')[0])

                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(verts)
                    mesh.triangles = o3d.utility.Vector3iVector(faces)
                    o3d.io.write_triangle_mesh(os.path.join(save_folder, 'coarse_shape.ply'), mesh)
                

    def predict(self, batch, train=True):
        # set module status
        self.set_module_training_mode(train_flag=train)

        # Inputs are already on device
        subject_garment_id = batch.get('feature_cube_idx')
        posed_points = batch.get('posed_points')
        pose = batch.get('pose')
        nn_smpl_skinning_weights = batch.get('smpl_nn_skinning_weights')
        nn_smpl_corr = batch.get('smpl_nn_cano')
        trans = batch.get('trans')

        # Run network to predict correspondences and DF
        logits = {}

        # inv_posed_points = self.inv_skinner(posed_points, pose, nn_smpl_skinning_weights, trans)['cano_cloth_points']

        x_c_rootfinding, skinning_weights_xc = search(posed_points, nn_smpl_corr, pose, trans, self.skinner, self.diffused_skinning_field, subject_garment_id, sw=None)
        # x_c_nn, skinning_weights_nn = search(posed_points, nn_smpl_corr, pose, trans, self.skinner, self.diffused_skinning_field, subject_garment_id, sw=nn_smpl_skinning_weights)

        # get on body points
        on_body_mask = compute_smaple_on_body_mask(x_c_rootfinding.permute(0, 2, 1), self.left_hand_x, self.right_hand_x, self.left_foot_y, self.right_foot_y)
        on_body_points = x_c_rootfinding.permute(0, 2, 1)[:, on_body_mask[0]]

        # random_sample_smpl
        smpl_points, smpl_normals = sample_points_from_meshes(self.smpl_mesh_, num_samples=50000, return_normals=True)
        on_hand_feet_mask = ~compute_smaple_on_body_mask(smpl_points, self.left_hand_x, self.right_hand_x, self.left_foot_y, self.right_foot_y)
        on_hand_feet_points = smpl_points[:, on_hand_feet_mask[0]]

        replaced_cano_pts = torch.cat((on_body_points, on_hand_feet_points), dim=1)

        query_value = self.subject_global_latent(replaced_cano_pts.permute(0, 2, 1), subject_garment_id) # [B, 256+3, num_points]

        logits.update(self.conditional_ndf(query_value))  # sdf_surface, nml_surface
        logits.update({'z_global': query_value[:, 3:, :]})      # [B, 256, N]
        logits.update({'hand_feet_normals': smpl_normals[:, on_hand_feet_mask[0]]})  # [B, N, 3]
        logits.update({'xc_skinning_weights': skinning_weights_xc}) # [B, 24, N]
        logits.update({'on_body_mask': on_body_mask})  # [B, N]

        return logits


    def compute_loss(self, batch, weights, train=True, ssp_only=False):

        device = self.device

        posed_points = batch.get('scan_points').to(device)
        posed_normals = batch.get('scan_normals').to(device)
        pose = batch.get('pose').to(device)
        trans = batch.get('trans').to(device).repeat(3, 1).permute(1, 0)
        nn_smpl_skinning_weights = batch.get('skinning_weights').to(device)
        nn_smpl_cano_corr = batch.get('ref_shaped_points').to(device)

        feature_cube_idx = batch.get('feature_cube_idx').to(device)

        inputs = {'posed_points': posed_points,
                  'posed_normals': posed_normals,
                  'pose': pose,
                  'trans': trans,
                  'smpl_nn_skinning_weights': nn_smpl_skinning_weights,
                  'smpl_nn_cano': nn_smpl_cano_corr,
                  'feature_cube_idx': feature_cube_idx}

        logits = self.predict(inputs, train=train)

        pred_sdf = logits['sdf_surface'].squeeze()                             # [B, N1]
        pred_normal = logits['nml_surface'].permute(0, 2, 1)   
        used_z_global = logits['z_global'] 

        on_body_mask = logits['on_body_mask']
        hand_feet_normals = logits['hand_feet_normals']

        queried_skinning_weights = logits['xc_skinning_weights']
        # inv skin normal using queried skinning weights
        inv_posed_normals = self.inv_skinner_normal(posed_normals, pose, queried_skinning_weights)['cano_cloth_normals'].permute(0, 2, 1)
        inv_body_normals = inv_posed_normals[:, on_body_mask[0]]

        replaced_cano_normals = torch.cat((inv_body_normals, hand_feet_normals), dim=1)

        # --------------------------------
        # ------------ losses ------------
        # udf for on-surface & off-surface: prediction - gt
        # regularization for latent code
        # on-surface specific: direction of pred gradient - gt_normal
        # eikonal term: the normal of the grad should be 1
        
        w_udf, w_normal, w_regular = weights
        w_eikonal = w_normal / 10
        w_normal = w_normal / 10

        loss_sdf = F.l1_loss(pred_sdf, torch.zeros_like(pred_sdf).to(device), reduction='none').mean()
        loss_normal = torch.norm(pred_normal - replaced_cano_normals, p=2, dim=2).mean() # debug
        loss_eikonal = (torch.norm(pred_normal, p=2, dim=2) - 1).pow(2).mean()
        loss_regularization = torch.mean(used_z_global ** 2)

        loss = {}
        loss.update({'sdf': loss_sdf * w_udf,
                     'normal': loss_normal * w_normal,
                     'eikonal': loss_eikonal * w_eikonal})
                     # 'latent_regularization': loss_regularization * w_regular

        return loss


if __name__ == "__main__":
    #     import os
    #     os.environ['CUDA_LAUNCH_BLOCKING']='1'

    parser = argparse.ArgumentParser(description='Run Model')
    # experiment id for folder suffix
    parser.add_argument('exp_id', type=str)
    parser.add_argument('-pretrained_sdf_exp', type=str)
    parser.add_argument('-pretrained_feature_exp', type=str)
    parser.add_argument('-batch_size', default=8, type=int)
    parser.add_argument('-split_file', '--split_file', type=str)
    parser.add_argument('-epochs', default=10000, type=int)
    parser.add_argument('-checkpoint', default=None, type=int)
    parser.add_argument('-pretrained', default=None, type=str)
    parser.add_argument('-extra_feat', default=0, type=int)
    # val, ft, pose_track, animate, detail_recon
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-save_name', default='', type=str)
    parser.add_argument('-num_samples', default=-1, type=int)

    args = parser.parse_args()

    args.subject_paths = [
        '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_{}/00032/longshort'.format(args.exp_id),
        '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_{}/00032/shortlong'.format(args.exp_id),
        '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_{}/00032/shortshort'.format(args.exp_id)
    ]

    args.pretrained_feature_exp_path = [
        '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_{}/00032/longshort'.format(args.pretrained_feature_exp),
        '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_{}/00032/shortlong'.format(args.pretrained_feature_exp),
        '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_{}/00032/shortshort'.format(args.pretrained_feature_exp)
    ]

    args.num_subjects = len(args.subject_paths)

    subject_index_dict = {}
    subject_index_dict.update({"00032_longshort": 0,
                               "00032_shortlong": 1,
                               "00032_shortshort": 2})

    print("Split file: ", args.split_file)

    exp_name = 'PoseImplicit_exp_id_{}'.format(args.exp_id)
    pretrained_sdf_name = 'PoseImplicit_exp_id_{}'.format(args.pretrained_sdf_exp)

    pretrained_module_dict = {
        'conditional_sdf': pretrained_sdf_name
    }

    inv_skinner = MyDataParallel(InvSkinModel())
    inv_skinner_normal = MyDataParallel(InvSkinModel_RotationOnly())
    subject_global_latent = MyDataParallel(SubjectGlobalLatentFeature(num_subjects=args.num_subjects, subject_paths=args.subject_paths, pretrained_feature_exp=args.pretrained_feature_exp_path, latent_size=256))
    # 256: global latent code
    # 3: query points on SMPL surface

    # smoothly diffused skinning field
    precomputed_skinning_field_base_path = '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/diffused_smpl_skinning_field/'
    diffused_skinning_field = MyDataParallel(SmoothDiffusedSkinningField(subject_field_base_path=precomputed_skinning_field_base_path, subject_index_dict=subject_index_dict))
    
    bbox_min = np.stack([-1.5, -1.5, -1.5], 0).astype(np.float32)
    bbox_max = np.stack([1.5, 1.5, 1.5], 0).astype(np.float32)
    
    conditional_sdf = MyDataParallel(IGRSDFNet(cond=256, bbox_min=bbox_min, bbox_max=bbox_max))

    # for forward skinning
    skinner = MyDataParallel(SkinModel())
    skinner_normal = MyDataParallel(SkinModel_RotationOnly())

    module_dict = {
        'inv_skinner': inv_skinner,
        'inv_skinner_normal': inv_skinner_normal,
        'diffused_skinning_field': diffused_skinning_field,
        'subject_global_latent': subject_global_latent,
        'conditional_sdf': conditional_sdf,
        'skinner': skinner,
        'skinner_normal': skinner_normal,
    }

    if args.mode == 'train':

        train_dataset = DataLoader_Cape_depth(mode='train', batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict)  
        val_dataset = DataLoader_Cape_depth(mode='val', batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict)  
        trainer = Trainer(module_dict, pretrained_module_dict, device=torch.device("cuda"), train_dataset=train_dataset, val_dataset=val_dataset, exp_name=exp_name)

        trainer.train_model(args.epochs, pretrained=args.pretrained, checkpoint=args.checkpoint)

    if args.mode == 'val':
        val_dataset = DataLoader_Cape_depth(mode='val', batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict)  
        trainer = Trainer(module_dict, pretrained_module_dict, device=torch.device("cuda"), train_dataset=None, val_dataset=val_dataset, exp_name=exp_name)

        trainer.test_model(args.save_name, args.num_samples, pretrained=args.pretrained, checkpoint=args.checkpoint)

    if args.mode == 'fine_tune':

        val_dataset = DataLoader_Cape_depth(mode='val', batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict)  
        trainer = Trainer(module_dict, pretrained_module_dict, device=torch.device("cuda"), train_dataset=None, val_dataset=val_dataset, exp_name=exp_name)

        trainer.fine_tune_model(args.epochs, pretrained=args.pretrained, checkpoint=args.checkpoint)

