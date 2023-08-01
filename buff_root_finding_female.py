from libs.global_variable import ROOT_DIR
from libs.global_variable import position

# find pose-dependent canonical correspondence via inverse skinning
import torch
from os.path import join, split
import os

from libs.canonicalization_root_finding import search
from libs.skinning_functions import InvSkinModel_RotationOnly, SkinModel, InvSkinModel, SkinModel_RotationOnly
from models.person_diffused_skinning import SmoothDiffusedSkinningField 

from dataloaders.dataloader_buff import DataLoader_Buff_depth_rootfinding

from trainer.data_parallel import MyDataParallel
from trainer.basic_trainer_invskinning import Basic_Trainer_invskinning

from tqdm import tqdm
import numpy as np
import argparse

from libs.barycentric_corr_finding import point_to_mesh_distance, face_vertices

from pytorch3d.structures import Meshes

class Trainer(Basic_Trainer_invskinning):

    def canonicalize_points(self):

        file_enumerator = self.dataset.get_loader(shuffle=False)
        device = self.device

        for n, batch in enumerate(tqdm(file_enumerator)):

            posed_points = batch.get('scan_points').to(device)
            posed_normals = batch.get('scan_normals').to(device) 
            pose = batch.get('pose').to(device)
            trans = batch.get('trans').to(device).repeat(3, 1).permute(1, 0)
            nn_smpl_cano_corr = batch.get('ref_shaped_points').to(device)
            feature_cube_idx = batch.get('feature_cube_idx').to(device)
            
            x_c_rootfinding, skinning_weights_xc = search(posed_points, nn_smpl_cano_corr, pose, trans, self.skinner, self.diffused_skinning_field, feature_cube_idx, sw=None)

            inv_posed_points = x_c_rootfinding.to(device).permute(0,2,1) # [:, on_body_mask]

            scan_rot_normals = batch.get('scan_normals_rotated').permute(0,2,1)
            posed_normals.permute(0,2,1)[scan_rot_normals[:, :, -1]<0] = -posed_normals.permute(0,2,1)[scan_rot_normals[:, :, -1]<0]
            inv_posed_normal = self.inv_skinner_normal(posed_normals, pose, skinning_weights_xc)['cano_cloth_normals'].permute(0, 2, 1)

            smpl_verts = self.diffused_skinning_field.smpl_ref_verts.to(device)[feature_cube_idx]
            smpl_faces = self.diffused_skinning_field.smpl_ref_faces.to(device)
            smpl_mesh = Meshes(verts=[smpl_verts[0].float()], faces=[smpl_faces.float()])

            face_verts_loc = face_vertices(smpl_mesh.verts_padded(), smpl_mesh.faces_padded()).contiguous()
            residues, pts_ind, _ = point_to_mesh_distance(inv_posed_points, face_verts_loc)
          
            valid_mask = residues<0.002

            inv_posed_points_ = inv_posed_points[valid_mask][None] # [B, N, 3]
            inv_posed_normal_ = inv_posed_normal[valid_mask][None] # [B, N, 3]

            # debug 
            debug = False
            if debug:

                import open3d as o3d
                # write function which uses open3d to write point cloud
                def write_pcd(path, points, normals):
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.normals = o3d.utility.Vector3dVector(normals)
                    o3d.io.write_point_cloud(path, pcd)

                names = batch.get('path')
                file_path = names[0]
                subject = file_path.split('/')[position] # if local 9; if cluster 12
                garment = split(file_path)[1].split('_')[0]
                save_folder = join("./visualization", 'debug_inv_skinning', subject, garment)

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                path_cano = os.path.join(save_folder, 'cano_corr_{}.ply'.format(names[0].split('/')[-1]))

                # write cano point cloud
                write_pcd(path_cano, inv_posed_points_.cpu().numpy()[0], inv_posed_normal_.cpu().numpy()[0])

            path_batch = batch.get('path')
            
            for i in range(len(path_batch)):
                file_path = path_batch[i]

                file_new = {
                    "cano_points": inv_posed_points_[i].cpu().numpy(),
                    "cano_normals": inv_posed_normal_[i].cpu().numpy(),
                    "valid_mask": valid_mask[i].cpu().numpy()
                }

                file_path_new = file_path.split(".")[0] + "_cano." + file_path.split(".")[1]

                np.save(file_path_new, file_new)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('-batch_size', default=1, type=int)

    args = parser.parse_args()

    subject_index_dict = {}
    subject_index_dict.update({"03223_shortlong": 0,
                               "03223_shortshort": 1})

    # multi subj query: for one subject with different garments, only use one skinning field
    general_subject_index = {}
    general_subject_index_numer = {}
    for key, value in subject_index_dict.items():
        if key.startswith('03223'):
            general_subject_index.update({'{}'.format(value): '03223'})
            general_subject_index_numer.update({value: 0})

    # gender is important for skinning
    inv_skinner = MyDataParallel(InvSkinModel(gender='female'))
    inv_skinner_normal = MyDataParallel(InvSkinModel_RotationOnly(gender='female'))
    skinner = MyDataParallel(SkinModel(gender='female'))
    skinner_normal = MyDataParallel(SkinModel_RotationOnly(gender='female'))

    # smoothly diffused skinning field
    precomputed_skinning_field_base_path = ROOT_DIR + 'diffused_smpl_skinning_field/'
    diffused_skinning_field = MyDataParallel(SmoothDiffusedSkinningField(subject_field_base_path=precomputed_skinning_field_base_path, general_subject_index=general_subject_index, general_subject_index_numer=general_subject_index_numer))

    module_dict = {
        'diffused_skinning_field': diffused_skinning_field,
        'inv_skinner': inv_skinner,
        'inv_skinner_normal': inv_skinner_normal,
        'skinner': skinner,
        'skinner_normal': skinner_normal
    }

    # preprocessed path 
    preprocessed_buff_path = ROOT_DIR + 'Data_female/BuFF/buff_release_rot_const/sequences'

    dataset = DataLoader_Buff_depth_rootfinding(proprocessed_path=preprocessed_buff_path, batch_size=args.batch_size, num_workers=4, subject_index_dict=subject_index_dict)  
    trainer = Trainer(module_dict, device=torch.device("cuda"), dataset=dataset)

    trainer.canonicalize_points()