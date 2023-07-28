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

import open3d as o3d

class Trainer(Basic_Trainer_sdf):
    
    # test on seen subjects but unseen subject
    def produce_fusion_shape(self, save_name='Recon_256', num_samples=-1, pretrained=None, checkpoint=None):

        epoch = self.load_checkpoint(path=pretrained, number=checkpoint)

        print('Testing with epoch {}'.format(epoch))
        val_data_loader = self.val_dataset.get_loader(shuffle=False)

        with torch.no_grad():

            self.set_feat_training_mode(train_flag=False)
            
            for n, batch in enumerate(tqdm(val_data_loader)):

                device = self.device

                self.set_module_training_mode(train_flag=False)

                feature_cube_idx = batch.get('feature_cube_idx').to(device)

                feature = self.subject_global_latent.get_feature(feature_cube_idx)

                bbox_min = np.stack([-1.5, -1.5, -1.5], 0).astype(np.float32)
                bbox_max = np.stack([1.5, 1.5, 1.5], 0).astype(np.float32)

                verts, faces, normals, values = condition_reconstruction(self.conditional_sdf, device, feature, resolution=int(save_name.split('_')[1]), thresh=0.00001, b_min=bbox_min, b_max=bbox_max, texture_net=None)

                names = batch.get('path')

                for i in range(len(names)):
                    file_path = names[i]
                    subject = file_path.split('/')[9] # if local 9; if cluster 12
                    garment = split(file_path)[1].split('_')[0]
                    save_folder = join(self.exp_path, save_name + '_ep_{}'.format(epoch), subject, garment)

                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    import open3d as o3d
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(verts)
                    mesh.triangles = o3d.utility.Vector3iVector(faces)
                    o3d.io.write_triangle_mesh(os.path.join(save_folder, 'coarse_shape.ply'), mesh)
                
    def project_fusion_shape(self, save_name='Recon', num_samples=-1, pretrained=None, checkpoint=None):

        # Don't use split file, process all files. Please refer to dataloader_buff.py
        epoch = self.load_checkpoint(path=pretrained, number=checkpoint)

        print('Projecting with epoch {}'.format(epoch))
        file_enumerator = self.dataset.get_loader(shuffle=False)
        device = self.device

        for n, batch in enumerate(tqdm(file_enumerator)):

            cano_points = batch.get('cano_points').to(device)
            feature_cube_idx = batch.get('feature_cube_idx').to(device)

            for param in subject_global_latent.parameters():
                param.requires_grad = False

            # 1st: get inverse skinning canonical points
            x_c_coarse = cano_points.clone()

            for i in range(10):
                # 2nd: get sdf and gradient to the coarse template surface via neural field
                query_value_sdf = subject_global_latent(x_c_coarse, feature_cube_idx) # [B, 256+3, num_points]
                logits = {}
                logits.update(self.conditional_sdf(query_value_sdf))  # sdf_surface, nml_surface
                sdf_value = logits.get('sdf_surface')                     # [B, 1, N]
                normal_value = logits.get('nml_surface')       

                # 3rd: project pose-dependent canonical template onto coarse template surface
                x_c_coarse = x_c_coarse + (-sdf_value.repeat(1, 3, 1).detach() * normal_value.detach()) # [B, 3, N]

            # save the projected canonical points
            path_batch = batch.get('path')
            num_org_points = batch.get('num_org_points')

            # debug 
            debug = True
            if debug:
                import open3d as o3d

                # write function which uses open3d to write point cloud
                def write_pcd(path, points):
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    o3d.io.write_point_cloud(path, pcd)

                names = batch.get('path')
                file_path = names[0]
                subject = file_path.split('/')[9] # if local 9; if cluster 12
                garment = split(file_path)[1].split('_')[0]
                save_folder = join(self.exp_path, 'debug_projection', subject, garment)

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                path_cano = os.path.join(save_folder, 'cano_corr_{}.ply'.format(names[0].split('/')[-1]))

                # write cano point cloud
                write_pcd(path_cano, x_c_coarse[0].permute(1,0).contiguous().cpu().numpy())
                
            for i in range(len(path_batch)):
                
                file_path = path_batch[i]
                file_path_cano = file_path.split(".")[0] + "_cano." + file_path.split(".")[1]

                dd = np.load(file_path_cano, allow_pickle=True).item()
                dd.update(
                    {"coarse_cano_points": x_c_coarse.permute(0,2,1).contiguous()[i][:num_org_points[i]].cpu().numpy()}
                )

                np.save(file_path_cano, dd)

    def fit_smpl_fusion_shape(self, pretrained=None, checkpoint=None):

        from pytorch3d.loss import (
            mesh_edge_loss, 
            mesh_laplacian_smoothing
        )
        from pytorch3d.ops import SubdivideMeshes
        from pytorch3d.io import save_ply

        epoch = self.load_checkpoint(path=pretrained, number=checkpoint)
        print('Fitting SMPL with epoch {}'.format(epoch))

        self.set_module_training_mode(train_flag=False)
        self.set_feat_training_mode(train_flag=False)

        for subj_gar in self.val_dataset.subject_index_dict.keys():

            subj_gar_idx = self.val_dataset.subject_index_dict.get(subj_gar)
            
            # smpl mesh of minimal shape
            smpl_verts = self.diffused_skinning_field.smpl_ref_verts.to(self.device)[subj_gar_idx]
            smpl_faces = self.diffused_skinning_field.smpl_ref_faces.to(self.device)
            smpl_mesh = Meshes(verts=[smpl_verts.float()], faces=[smpl_faces.float()])

            deform_verts = torch.full(smpl_mesh.verts_packed().shape, 0.0, device=self.device, requires_grad=True)
            optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

            # Number of optimization steps
            Niter = 1000
            # Weight for the sdf loss
            w_sdf = 1
            # Weight for mesh edge loss
            w_edge = 5.0 
            # Weight for mesh laplacian smoothing
            w_laplacian = 0.1 
            # Plot period for the losses
            plot_period = 100
            loop = tqdm(range(Niter))

            for i in loop:
                # Initialize optimizer
                optimizer.zero_grad()

                # Deform the mesh
                new_src_mesh = smpl_mesh.offset_verts(deform_verts)
                smpl_points, smpl_normals = sample_points_from_meshes(new_src_mesh, num_samples=50000, return_normals=True)
                
                # and (b) the edge length of the predicted mesh
                loss_edge = mesh_edge_loss(new_src_mesh)
                # mesh laplacian smoothing
                loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
                
                query_value = self.subject_global_latent(smpl_points.permute(0, 2, 1), torch.tensor([subj_gar_idx]).cuda()) # [B, 256+3, num_points]
                
                logits = {}
                logits.update(self.conditional_sdf(query_value))  # sdf_surface, nml_surface
                logits.update({'z_global': query_value[:, 3:, :]})      # [B, 256, N]
                pred_sdf = logits['sdf_surface'].squeeze()               # [B, N1]

                # all sampled points should have sdf equals zero
                loss_sdf = F.l1_loss(pred_sdf, torch.zeros_like(pred_sdf), reduction='none').mean()
                loss = loss_sdf * w_sdf  + loss_edge * w_edge + loss_laplacian * w_laplacian
                
                # Print the losses
                loop.set_description('total_loss = %.6f' % loss)

                # Plot mesh
                if i % plot_period == 0:

                    pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(np.concatenate((new_src_mesh.verts_packed().detach().cpu().numpy(), mc_mesh.verts_packed().detach().cpu().numpy())))
                    pcd.points = o3d.utility.Vector3dVector((new_src_mesh.verts_packed().detach().cpu().numpy()))
                    # o3d.visualization.draw_geometries([pcd])

                # Optimization step
                loss.backward()
                optimizer.step()

            fusion_shape_save_dir = ROOT_DIR + 'Data/BuFF/Fusion_shape/'

            if not os.path.exists(fusion_shape_save_dir):
                os.makedirs(fusion_shape_save_dir)

            save_path = fusion_shape_save_dir + 'smpl_D_' + subj_gar + '.ply'
            save_subdivided_path = fusion_shape_save_dir + 'smpl_D_' + subj_gar + '_subdivided.ply'

            mesh_divider = SubdivideMeshes()
            smpld_mesh_sub = mesh_divider(new_src_mesh)

            save_ply(save_path, verts=new_src_mesh.verts_packed(), faces=new_src_mesh.faces_packed())
            save_ply(save_subdivided_path, verts=smpld_mesh_sub.verts_packed(), faces=smpld_mesh_sub.faces_packed())

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
        debug = False
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

            names = batch.get('path')
            file_path = names[0]
            subject = file_path.split('/')[9] # if local 9; if cluster 12
            garment = split(file_path)[1].split('_')[0]
            save_folder = join(self.exp_path, 'debug_inv_skinning', subject, garment)

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            path_cano = os.path.join(save_folder, 'cano_corr_{}.ply'.format(names[0].split('/')[-1]))

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
    parser.add_argument('-epochs', '--epochs', default=300, type=int)
    # val, ft, pose_track, animate, detail_recon
    parser.add_argument('-mode', '--mode', default='train', type=str)
    parser.add_argument('-save_name', '--save_name', default='Recon_512', type=str)

    args = parser.parse_args()

    args.subject_paths = [
        ROOT_DIR + 'experiments/PoseImplicit_exp_id_{}/00032/shortlong'.format(args.exp_id),
        ROOT_DIR + 'experiments/PoseImplicit_exp_id_{}/00096/shortlong'.format(args.exp_id)
    ]

    args.pretrained_feature_exp_path = [
        ROOT_DIR + 'experiments/PoseImplicit_exp_id_{}/00032/shortlong'.format(args.pretrained_exp),
        ROOT_DIR + 'experiments/PoseImplicit_exp_id_{}/00096/shortlong'.format(args.pretrained_exp)
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

    if args.mode == 'fusion_shape':
        val_dataset = DataLoader_Buff_depth(mode='val', batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict)  
        trainer = Trainer(module_dict, pretrained_module_dict, device=torch.device("cuda"), train_dataset=None, val_dataset=val_dataset, exp_name=exp_name)

        trainer.produce_fusion_shape(args.save_name)

    if args.mode == 'projection':

        # preprocessed path 
        preprocessed_buff_path = ROOT_DIR + 'Data/BuFF/buff_release_rot_const/sequences'

        dataset = DataLoader_Buff_depth(cano_available=True, proprocessed_path=preprocessed_buff_path, batch_size=args.batch_size, num_workers=4, subject_index_dict=subject_index_dict)  
        trainer = Trainer(module_dict, pretrained_module_dict, device=torch.device("cuda"), dataset=dataset, exp_name=exp_name)

        trainer.project_fusion_shape()

    if args.mode == 'fit_smpl':

        val_dataset = DataLoader_Buff_depth(mode='val', batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict)  
        trainer = Trainer(module_dict, pretrained_module_dict, device=torch.device("cuda"), train_dataset=None, val_dataset=val_dataset, exp_name=exp_name)

        trainer.fit_smpl_fusion_shape()


