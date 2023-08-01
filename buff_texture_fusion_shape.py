from libs.global_variable import ROOT_DIR
from libs.global_variable import position

# coarse template via inverse skinning of scan using SMPL skinning weights
import torch

torch.manual_seed(0)

from tqdm import tqdm
from os.path import join, split
import numpy as np
import random
import os

from pytorch3d.ops.knn import knn_points

from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes

def get_verts_and_faces(path):

    verts = load_ply(path)[0].cuda()
    faces = load_ply(path)[1].cuda()

    return verts, faces

def vertex_color_nn(subject_garment_name, suffix=None, use_frames=10):
    # for instance, subject_garment_name = '00032_shortlong'
    fusion_shape_base_path = join(ROOT_DIR, 'Data', 'BuFF', 'Fusion_shape')
    preprocessed_base_path = join(ROOT_DIR, 'Data', 'BuFF', 'buff_release_rot_const', 'sequences', subject_garment_name.split('_')[0])

    smpl_d_shape_path = join(fusion_shape_base_path, 'smpl_D_'+subject_garment_name+'{}.ply'.format(suffix))
    smpl_d_verts, smpl_d_faces = get_verts_and_faces(smpl_d_shape_path)

    available_preprocessed = []

    for garment_action in os.listdir(preprocessed_base_path):
        if not garment_action.startswith(subject_garment_name.split('_')[1]):
            continue
        garment_action_path = join(preprocessed_base_path, garment_action)

        for npy_file_name in os.listdir(garment_action_path):
            
                if npy_file_name.split('.')[0].endswith('cano'):
                    continue
                frame_npy_path = os.path.join(garment_action_path, npy_file_name)

                available_preprocessed.append(frame_npy_path)

    used_frames_paths = random.sample(available_preprocessed, use_frames)

    corr_list = []
    color_list = []
    for used_frames_path in used_frames_paths:
        dd = np.load(used_frames_path, allow_pickle=True).item()
        dd_cano = np.load(used_frames_path.split(".")[0] + "_cano." + used_frames_path.split(".")[1], allow_pickle=True).item()

        valid_mask = dd_cano['valid_mask']
        coarse_cano_points = dd_cano['coarse_cano_points']
        scan_points_color = dd['colors_posed_cloth'].transpose()[valid_mask]

        assert coarse_cano_points.shape[0] == scan_points_color.shape[0]

        corr_list.append(coarse_cano_points)
        color_list.append(scan_points_color)

    corr_list = np.concatenate([np.array(i) for i in corr_list])
    color_list = np.concatenate([np.array(i) for i in color_list])

    corr_list = torch.tensor(corr_list).cuda()
    color_list = torch.tensor(color_list).cuda()

    dist, idx, _ = knn_points(p1=smpl_d_verts[None, :], p2=corr_list[None, :])
    verts_location = smpl_d_verts
    verts_color = color_list[idx[0,:,0], :]
    verts_color[dist[0,:,0] > 1e-3, :] = 0.5

    debug = True
    if debug:
        import open3d as o3d

        pcd_cano = o3d.geometry.PointCloud()
        pcd_cano.points = o3d.utility.Vector3dVector(verts_location.cpu().numpy())
        pcd_cano.colors = o3d.utility.Vector3dVector(verts_color.cpu().numpy())

        mesh_cano = o3d.geometry.TriangleMesh()
        mesh_cano.vertices = o3d.utility.Vector3dVector(verts_location.cpu().numpy())
        mesh_cano.vertex_colors = o3d.utility.Vector3dVector(verts_color.cpu().numpy())
        mesh_cano.triangles = o3d.utility.Vector3iVector(smpl_d_faces.cpu().numpy())

        # o3d.visualization.draw_geometries([mesh_cano, pcd_cano])

        textured_smpl_d_shape_path = smpl_d_shape_path.split(".")[0] + "_texture." + smpl_d_shape_path.split(".")[1]
        o3d.io.write_triangle_mesh(textured_smpl_d_shape_path, mesh_cano)


if __name__ == "__main__":
    subject_garment_name = ['00032_shortlong', '00032_shortshort', '00096_shortlong', '00096_shortshort']

    for i in subject_garment_name:
        print("processing subject_garment_name: ", i)
        vertex_color_nn(i, suffix='_subdivided', use_frames=5)
    

