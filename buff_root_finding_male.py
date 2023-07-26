ROOT_DIR = '/home/yuxuan/project/NeuralSurfaceField/'

# find pose-dependent canonical correspondence via inverse skinning
import torch

from libs.canonicalization_root_finding import search
from libs.skinning_functions import InvSkinModel_RotationOnly, SkinModel, InvSkinModel, SkinModel_RotationOnly
from models.person_diffused_skinning import SmoothDiffusedSkinningField 

from dataloaders.dataloader_buff import DataLoader_Buff_depth

from trainer.data_parallel import MyDataParallel
from trainer.basic_trainer_invskinning import Basic_Trainer_invskinning

from tqdm import tqdm
import numpy as np
import argparse

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

            path_batch = batch.get('path')

            num_org_points = batch.get('num_org_points')
            
            for i in range(len(path_batch)):
                file_path = path_batch[i]
                num_org_points_i = num_org_points[i]

                file_new = {
                    "cano_points": inv_posed_points[i][:num_org_points_i].cpu().numpy(),
                    "cano_normals": inv_posed_normal[i][:num_org_points_i].cpu().numpy(),
                }

                file_path_new = file_path.split(".")[0] + "_cano." + file_path.split(".")[1]

                np.save(file_path_new, file_new)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('-batch_size', default=1, type=int)

    args = parser.parse_args()

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

    # gender is important for skinning
    inv_skinner = MyDataParallel(InvSkinModel(gender='male'))
    inv_skinner_normal = MyDataParallel(InvSkinModel_RotationOnly(gender='male'))
    skinner = MyDataParallel(SkinModel(gender='male'))
    skinner_normal = MyDataParallel(SkinModel_RotationOnly(gender='male'))

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
    preprocessed_buff_path = ROOT_DIR + 'Data/BuFF/buff_release_rot_const/sequences'

    dataset = DataLoader_Buff_depth(proprocessed_path=preprocessed_buff_path, batch_size=args.batch_size, num_workers=4, subject_index_dict=subject_index_dict)  
    trainer = Trainer(module_dict, device=torch.device("cuda"), dataset=dataset)

    trainer.canonicalize_points()