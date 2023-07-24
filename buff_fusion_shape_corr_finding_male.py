# coarse template via inverse skinning of scan using SMPL skinning weights
import torch
import torch.nn.functional as F

from lib.canonicalization_root_finding import search
from lib.skinning_functions import InvSkinModel, InvSkinModel_RotationOnly, SkinModel, SkinModel_RotationOnly
from models.person_specific_feature import SubjectGlobalLatentFeature
from models.person_diffused_skinning import SmoothDiffusedSkinningField 
from models.igr_sdf_net import IGRSDFNet

from dataloaders.dataloader_buff import DataLoader_Buff_depth

from trainer.data_parallel import MyDataParallel
from trainer.basic_trainer_sdf import Basic_Trainer_sdf

from tqdm import tqdm
from os.path import join, split
import numpy as np
import argparse
import os


class Trainer(Basic_Trainer_sdf):

    def test_model(self, save_name, num_samples=-1, pretrained=None, checkpoint=None):

        epoch = self.load_checkpoint(path=pretrained, number=checkpoint)

        print('Testing with epoch {}'.format(epoch))
        val_data_loader = self.val_dataset.get_loader(shuffle=True)

        self.set_feat_training_mode(train_flag=False)
        count = 0
        for n, batch in enumerate(tqdm(val_data_loader)):

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

            logits = self.predict(inputs, train=False)

            # points
            inv_posed_points = logits.get('inv_posed_points').to(device).permute(0,2,1) # [:, on_body_mask]
            
            # normals
            scan_rot_normals = batch.get('scan_normals_rotated').permute(0,2,1)
            queried_skinning_weights = logits['xc_skinning_weights']
            posed_normals.permute(0,2,1)[scan_rot_normals[:, :, -1]<0] = -posed_normals.permute(0,2,1)[scan_rot_normals[:, :, -1]<0]
            
            inv_posed_normal = self.inv_skinner_normal(posed_normals, pose, queried_skinning_weights)['cano_cloth_normals'].permute(0, 2, 1)
            names = batch.get('path')

            for i in range(len(names)):
                file_path = names[i]
                name = split(file_path)[1]
                front_or_back = file_path.split('/')[-5]
                dataset = split(split(file_path)[0])[1]
                subject = file_path.split('/')[12]
                save_folder = join(self.exp_path, save_name + '_ep_{}'.format(epoch), subject, front_or_back, name.split('.')[0])

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                
                save_path_inv_normals = save_folder + '/cano_corr_normal.npy' # save inverse skinned normals in canonical space
                np.save(save_path_inv_normals, inv_posed_normal[i].cpu().numpy())

                save_path_inv_points = save_folder + '/cano_corr.npy'         # save inverse skinned points in canonical space
                np.save(save_path_inv_points, inv_posed_points[i].cpu().numpy())

            if num_samples > 0:
                if count >= num_samples:
                    break
            
            count += 1

    def predict(self, batch, animate=False, train=True):
    
        self.set_module_training_mode(train_flag=train)

        subject_garment_id = batch.get('feature_cube_idx')
        posed_points = batch.get('posed_points')
        posed_normals = batch.get('posed_normals')
        pose = batch.get('pose')
        nn_smpl_skinning_weights = batch.get('smpl_nn_skinning_weights')
        trans = batch.get('trans')
        nn_smpl_corr = batch.get('smpl_nn_cano') 

        # Run network to predict correspondences and DF
        logits = {}

        x_c_rootfinding, skinning_weights_xc = search(posed_points, nn_smpl_corr, pose, trans, self.skinner, self.diffused_skinning_field, subject_garment_id, sw=None)
        # x_c_rootfinding, skinning_weights_xc = search(posed_points, nn_smpl_corr, pose, trans, self.skinner, self.diffused_skinning_field, subject_garment_id, sw=nn_smpl_skinning_weights)

        logits.update({'inv_posed_points': x_c_rootfinding})
        logits.update({'xc_skinning_weights': skinning_weights_xc})

        return logits


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
    parser.add_argument('-epochs', default=4000, type=int)
    parser.add_argument('-checkpoint', default=None, type=int)
    parser.add_argument('-pretrained', default=None, type=str)
    parser.add_argument('-extra_feat', default=0, type=int)
    # val, ft, pose_track, animate, detail_recon
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-save_name', default='', type=str)
    parser.add_argument('-num_samples', default=-1, type=int)

    args = parser.parse_args()

    args.subject_paths = [
        '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_{}/00032/shortlong'.format(args.exp_id),
        '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_{}/00096/shortlong'.format(args.exp_id)
    ]

    args.pretrained_feature_exp_path = [
        '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_{}/00032/shortlong'.format(args.pretrained_feature_exp),
        '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_{}/00096/shortlong'.format(args.pretrained_feature_exp)
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
    pretrained_sdf_name = 'PoseImplicit_exp_id_{}'.format(args.pretrained_sdf_exp)

    pretrained_module_dict = {
        'conditional_sdf': pretrained_sdf_name
    }

    inv_skinner = MyDataParallel(InvSkinModel(gender='male'))
    inv_skinner_normal = MyDataParallel(InvSkinModel_RotationOnly(gender='male'))
    subject_global_latent = MyDataParallel(SubjectGlobalLatentFeature(num_subjects=args.num_subjects, subject_paths=args.subject_paths, pretrained_feature_exp=args.pretrained_feature_exp_path, latent_size=256))
    # 256: global latent code
    # 3: query points on SMPL surface

    # smoothly diffused skinning field
    precomputed_skinning_field_base_path = '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/diffused_smpl_skinning_field/'
    diffused_skinning_field = MyDataParallel(SmoothDiffusedSkinningField(subject_field_base_path=precomputed_skinning_field_base_path, general_subject_index=general_subject_index, general_subject_index_numer=general_subject_index_numer))
    
    bbox_min = np.stack([-1.5, -1.5, -1.5], 0).astype(np.float32)
    bbox_max = np.stack([1.5, 1.5, 1.5], 0).astype(np.float32)
    
    conditional_sdf = MyDataParallel(IGRSDFNet(cond=256, bbox_min=bbox_min, bbox_max=bbox_max))

    # for forward skinning
    skinner = MyDataParallel(SkinModel(gender='male'))
    skinner_normal = MyDataParallel(SkinModel_RotationOnly(gender='male'))

    module_dict = {
        'inv_skinner': inv_skinner,
        'inv_skinner_normal': inv_skinner_normal,
        'diffused_skinning_field': diffused_skinning_field,
        'subject_global_latent': subject_global_latent,
        'conditional_sdf': conditional_sdf,
        'skinner': skinner,
        'skinner_normal': skinner_normal,
    }

    if args.mode == 'val':
        val_dataset = DataLoader_Buff_depth(mode='val', batch_size=args.batch_size, num_workers=4, split_file=args.split_file, subject_index_dict=subject_index_dict)  
        trainer = Trainer(module_dict, pretrained_module_dict, device=torch.device("cuda"), train_dataset=None, val_dataset=val_dataset, exp_name=exp_name)

        trainer.test_model(args.save_name, args.num_samples, pretrained=args.pretrained, checkpoint=args.checkpoint)
