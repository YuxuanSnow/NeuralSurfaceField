ROOT_DIR = '/home/yuxuan/project/NeuralSurfaceField/'

# coarse template via inverse skinning of scan using SMPL skinning weights
import torch

torch.manual_seed(0)

import torch.nn.functional as F

from libs.skinning_functions import InvSkinModel_RotationOnly, SkinModel, InvSkinModel, SkinModel_RotationOnly
from models.person_diffused_skinning import SmoothDiffusedSkinningField 

from models.person_specific_feature import SubjectGlobalLatentFeature
from models.igr_sdf_net import IGRSDFNet, condition_reconstruction

from dataloaders.dataloader_buff import DataLoader_Buff_depth

from tqdm import tqdm
from os.path import join, split
import numpy as np
import argparse
import os

from trainer.data_parallel import MyDataParallel

from trainer.basic_trainer_sdf import Basic_Trainer_sdf
from libs.sample import compute_smaple_on_body_mask_w_batch

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

class Trainer(Basic_Trainer_sdf):
    
    # test on seen subjects but unseen subject
    def test_model(self, save_name='Recon', num_samples=-1, pretrained=None, checkpoint=None):

        epoch = self.load_checkpoint(path=pretrained, number=checkpoint)

        print('Testing with epoch {}'.format(epoch))
        val_data_loader = self.val_dataset.get_loader(shuffle=False)

        with torch.no_grad():

            self.set_feat_training_mode(train_flag=True)
            
            for n, batch in enumerate(tqdm(val_data_loader)):

                device = self.device

                self.set_module_training_mode(train_flag=False)

                feature_cube_idx = batch.get('feature_cube_idx').to(device)

                feature = self.subject_global_latent.get_feature(feature_cube_idx)

                bbox_min = np.stack([-1.5, -1.5, -1.5], 0).astype(np.float32)
                bbox_max = np.stack([1.5, 1.5, 1.5], 0).astype(np.float32)

                verts, faces, normals, values = condition_reconstruction(self.conditional_sdf, device, feature, resolution=256, thresh=0.00001, b_min=bbox_min, b_max=bbox_max, texture_net=None)

                names = batch.get('path')

                for i in range(len(names)):
                    file_path = names[i]
                    subject = file_path.split('/')[12]
                    name = split(file_path)[1]
                    front_or_back = file_path.split('/')[-6]
                    # dataset = split(split(file_path)[0])[1]
                    save_folder = join(self.exp_path, save_name + '_ep_{}'.format(epoch), subject, front_or_back, name.split('.')[0])

                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    import open3d as o3d
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(verts)
                    mesh.triangles = o3d.utility.Vector3iVector(faces)
                    o3d.io.write_triangle_mesh(os.path.join(save_folder, 'coarse_shape.ply'), mesh)
                

    def predict(self, batch, train=True):
        # set module status
        self.set_module_training_mode(train_flag=train)

        # Inputs are already on device
        subject_garment_id = batch.get('feature_cube_idx')
        cano_points = batch.get('cano_points')
        cano_normals = batch.get('cano_normals')

        # Run network to predict correspondences and DF
        logits = {}
         # get on body points
        subject_field_idx = torch.zeros_like(subject_garment_id)
        for i in range(len(subject_garment_id)):
            subject_field_idx[i] = self.diffused_skinning_field.general_subject_index_numer[subject_garment_id[i].item()]

        body_loc = self.diffused_skinning_field.subject_body_loc[subject_field_idx]

        hand_mask, feet_mask, head_mask = compute_smaple_on_body_mask_w_batch(cano_points, cut_offset=0.03, subject_loc=body_loc)
        on_body_mask = ~torch.logical_or(hand_mask, feet_mask)

        # get on body points
        on_body_points = cano_points.permute(0,2,1).contiguous()[on_body_mask][None, :]

        smpl_verts = self.diffused_skinning_field.smpl_ref_verts.to(subject_garment_id.device)[subject_garment_id]
        smpl_faces = self.diffused_skinning_field.smpl_ref_faces.to(subject_garment_id.device)
        smpl_mesh = Meshes(verts=[smpl_verts[0].float()], faces=[smpl_faces.float()])

        # random_sample_smpl
        smpl_points, smpl_normals = sample_points_from_meshes(smpl_mesh, num_samples=50000, return_normals=True)
        # on_hand_feet_mask = ~compute_smaple_on_body_mask(smpl_points, self.left_hand_x, self.right_hand_x, self.left_foot_y, self.right_foot_y)
        smpl_hand_mask, smpl_feet_mask, smpl_head_mask = compute_smaple_on_body_mask_w_batch(smpl_points.permute(0,2,1), cut_offset=0.05, subject_loc=body_loc)
        on_hand_feet_mask = torch.logical_or(smpl_hand_mask, smpl_feet_mask)
        on_hand_feet_points = smpl_points[on_hand_feet_mask][None, :]

        on_scalp_mask = torch.topk(smpl_points[:, :, 1], 1000)[1]
        on_scalp_points = smpl_points[:, on_scalp_mask[0], :]

        replaced_cano_pts = torch.cat((on_body_points, on_hand_feet_points, on_scalp_points), dim=1)

        query_value = self.subject_global_latent(replaced_cano_pts.permute(0, 2, 1), subject_garment_id) # [B, 256+3, num_points]

        logits.update(self.conditional_sdf(query_value))  # sdf_surface, nml_surface
        logits.update({'z_global': query_value[:, 3:, :]})      # [B, 256, N]
        logits.update({'hand_feet_normals': smpl_normals[:, on_hand_feet_mask[0]]})  # [B, N, 3]
        logits.update({'scalp_normals': smpl_normals[:, on_scalp_mask[0]]})  # [B, N, 3]
        logits.update({'query_location': replaced_cano_pts})
        logits.update({'on_body_mask': on_body_mask})  # [B, N]

        return logits

    def compute_loss(self, batch, weights, train=True, ssp_only=False):

        device = self.device

        name = batch.get('path')

        cano_points = batch.get('cano_points').to(device)
        cano_normals = batch.get('cano_normals').to(device)
        feature_cube_idx = batch.get('feature_cube_idx').to(device)

        inputs = {'cano_points': cano_points,
                  'feature_cube_idx': feature_cube_idx}

        logits = self.predict(inputs, train=train)

        pred_sdf = logits['sdf_surface'].squeeze()                             # [B, N1]
        pred_normal = logits['nml_surface'].permute(0, 2, 1)   
        used_z_global = logits['z_global'] 

        on_body_mask = logits['on_body_mask']
        hand_feet_normals = logits['hand_feet_normals']
        scalp_normals = logits['scalp_normals']
        # inv skin normal using queried skinning weights
        inv_posed_normals_ = cano_normals.permute(0,2,1)
        inv_body_normals = inv_posed_normals_[:, on_body_mask[0]]

        replaced_cano_normals = torch.cat((inv_body_normals, hand_feet_normals, scalp_normals), dim=1)

        # debug 
        debug = True
        if debug:
            replaced_points = logits.get('query_location')[0].cpu().numpy()
            replaced_normals = replaced_cano_normals[0].cpu().numpy()

            import open3d as o3d

            # write function which uses open3d to write point cloud
            def write_pcd(path, points, normals):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.normals = o3d.utility.Vector3dVector(normals)
                o3d.io.write_point_cloud(path, pcd)

            path_cano = '/home/yuxuan/project/NeuralSurfaceField/visualization/cano_replaced.ply'

            # write cano point cloud
            write_pcd(path_cano, replaced_points, replaced_normals)

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

    parser = argparse.ArgumentParser(description='Run Model')
    # experiment id for folder suffix
    parser.add_argument('-exp_id', '--exp_id', type=str)
    parser.add_argument('-pretrained_exp', '--pretrained_exp', type=str)
    parser.add_argument('-batch_size', '--batch_size', default=8, type=int)
    parser.add_argument('-split_file', '--split_file', type=str)
    parser.add_argument('-epochs', default=10000, type=int)
    # val, ft, pose_track, animate, detail_recon
    parser.add_argument('-mode', '--mode', default='train', type=str)
    parser.add_argument('-save_name', '--save_name', default='Recon_256_', type=str)

    args = parser.parse_args()

    args.subject_paths = [
        '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_{}/00032/shortlong'.format(args.exp_id),
        '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_{}/00096/shortlong'.format(args.exp_id)
    ]

    args.pretrained_feature_exp_path = [
        '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_{}/00032/shortlong'.format(args.pretrained_exp),
        '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_{}/00096/shortlong'.format(args.pretrained_exp)
    ]

    args.num_subjects = len(args.subject_paths)

    subject_index_dict = {}
    subject_index_dict.update({"00032_shortlong": 0,
                               "00096_shortlong": 1})

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
    pretrained_sdf_name = 'PoseImplicit_exp_id_{}'.format(args.pretrained_exp)

    pretrained_module_dict = {
        'conditional_sdf': pretrained_sdf_name
    }

    subject_global_latent = MyDataParallel(SubjectGlobalLatentFeature(num_subjects=args.num_subjects, subject_paths=args.subject_paths, pretrained_feature_exp=args.pretrained_feature_exp_path, latent_size=256))
    # 256: global latent code
    # 3: query points on SMPL surface

    # smoothly diffused skinning field
    precomputed_skinning_field_base_path = ROOT_DIR + 'diffused_smpl_skinning_field/'
    diffused_skinning_field = MyDataParallel(SmoothDiffusedSkinningField(subject_field_base_path=precomputed_skinning_field_base_path, general_subject_index=general_subject_index, general_subject_index_numer=general_subject_index_numer))
    
    bbox_min = np.stack([-1.5, -1.5, -1.5], 0).astype(np.float32)
    bbox_max = np.stack([1.5, 1.5, 1.5], 0).astype(np.float32)
    
    conditional_sdf = MyDataParallel(IGRSDFNet(cond=256, bbox_min=bbox_min, bbox_max=bbox_max))

    # for forward skinning
    inv_skinner = MyDataParallel(InvSkinModel(gender='male'))
    inv_skinner_normal = MyDataParallel(InvSkinModel_RotationOnly(gender='male'))
    skinner = MyDataParallel(SkinModel(gender='male'))
    skinner_normal = MyDataParallel(SkinModel_RotationOnly(gender='male'))

    module_dict = {
        'inv_skinner': inv_skinner,
        'inv_skinner_normal': inv_skinner_normal,
        'skinner': skinner,
        'skinner_normal': skinner_normal,
        'diffused_skinning_field': diffused_skinning_field,
        'subject_global_latent': subject_global_latent,
        'conditional_sdf': conditional_sdf,
    }

    if args.mode == 'train':

        train_dataset = DataLoader_Buff_depth(mode='train', batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict)  
        val_dataset = DataLoader_Buff_depth(mode='val', batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict)  
        trainer = Trainer(module_dict, pretrained_module_dict, device=torch.device("cuda"), train_dataset=train_dataset, val_dataset=val_dataset, exp_name=exp_name)

        trainer.train_model(args.epochs)

    if args.mode == 'val':
        val_dataset = DataLoader_Buff_depth(mode='val', batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict)  
        trainer = Trainer(module_dict, pretrained_module_dict, device=torch.device("cuda"), train_dataset=None, val_dataset=val_dataset, exp_name=exp_name)

        trainer.test_model(args.save_name)

    if args.mode == 'generate':

        val_dataset = DataLoader_Buff_depth(mode='val', batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict)  
        trainer = Trainer(module_dict, pretrained_module_dict, device=torch.device("cuda"), train_dataset=None, val_dataset=val_dataset, exp_name=exp_name)

        trainer.generate(args.epochs, pretrained=args.pretrained, checkpoint=args.checkpoint)

