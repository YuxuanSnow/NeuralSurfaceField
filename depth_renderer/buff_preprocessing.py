import numpy as np
import os
from libs.smpl_paths import SmplPaths
import torch
import pickle

from libs.skinning_functions import SkinModel, InvSkinModel

from random import randrange

from libs.torch_functions import np2tensor, tensor2np

from pytorch3d.structures.meshes import Meshes

from scipy.spatial.transform import Rotation as ScipyRot

from pytorch3d.renderer.mesh.rasterize_meshes import pix_to_non_square_ndc

import math

from libs.sample import compute_smaple_on_body_mask_wo_batch
from libs.data_io import generate_point_cloud

from libs.barycentric_corr_finding import face_vertices, point_to_mesh_distance, barycentric_coordinates_of_projection

import open3d as o3d

import pymeshlab

scale = 1

# undo root + body mask
def buff_preprocessing_depth_rot_trans(subject_gender, subject_action_frame_ply, subject_shape_pkl, save_name, rasterizer, renderer, r_rot_y, skinner, inv_skinner, location_dict, debug=False):
    # sequence frame in npz
    textured_mesh = o3d.io.read_triangle_mesh(subject_action_frame_ply, True)
    subject_action_frame_cape_npz = subject_action_frame_ply.replace('.ply', '.npz')
    frame_npz_loaded = np.load(subject_action_frame_cape_npz)

    # load posed and shaped naked smpl mesh, load theta, betas, trans, and centralize it
    with open(subject_shape_pkl, 'rb') as f:
        data = pickle.load(f)
    pose_org = frame_npz_loaded['pose']
    betas = data['betas'] 
    trans = frame_npz_loaded['transl'] 

    buff_verts_color = np.asarray(textured_mesh.vertex_colors)
    buff_verts = np.asarray(textured_mesh.vertices) / 1000 - trans
    buff_faces = np.asarray(textured_mesh.triangles)

    pose_inv = np.copy(pose_org)
    pose_inv[3:] = 0
    pose_inv_cuda = torch.tensor(pose_inv)[None].cuda().float()
    trans_cuda = torch.tensor([0]).cuda().float()

    posed_verts_cuda = torch.tensor(buff_verts)[None].permute(0,2,1).cuda().float()
    pose_inv_cuda = torch.tensor(pose_inv)[None].cuda().float()
    sw_root = np.zeros((posed_verts_cuda.shape[2], 24))
    sw_root[:, 0] = 1
    sw_root_cuda = torch.tensor(sw_root)[None].permute(0,2,1).cuda().float()
    trans_cuda = torch.tensor([0]).cuda().float()

    # inverse skin the root rotation then manually add rotation
    buff_verts_undo_root = inv_skinner(posed_verts_cuda, pose_inv_cuda, sw_root_cuda, trans_cuda)['cano_cloth_points'].permute(0,2,1).cpu().numpy()[0]
    buff_verts_undo_root_add_rot = np2tensor(r_rot_y.apply(buff_verts_undo_root))

    texture = TexturesVertex([np2tensor(buff_verts_color).float().cuda()])
    textured_buff_mesh = Meshes(verts=[np2tensor(buff_verts_undo_root_add_rot).float().cuda()], faces=[np2tensor(buff_faces).float().cuda()], textures=texture)
    
    rgb_img, fragments_buff = renderer(textured_buff_mesh) # render the buff textured mesh

    # unproject the rendered RGB and depth image to obtain coloful partial pcd
    x_screen = torch.nonzero((fragments_buff.zbuf+1)[0, :, :, 0]).t()[1]
    y_screen = torch.nonzero((fragments_buff.zbuf+1)[0, :, :, 0]).t()[0]
    depth__ = fragments_buff.zbuf[0, y_screen, x_screen, 0]

    x_ndc_re = -pix_to_non_square_ndc(x_screen.squeeze(), 640*scale, 576*scale)
    y_ndc_re = -pix_to_non_square_ndc(y_screen.squeeze(), 576*scale, 640*scale)
    depth = depth__[None, None, :]

    xy_ndc_re = torch.stack([x_ndc_re, y_ndc_re])[None, :]
    xy_depth = torch.cat((xy_ndc_re, depth), dim=1).permute(0, 2, 1)

    unproj_pcd_loc = renderer.shader.cameras.unproject_points(xy_depth, from_ndc=True) # [1, N, 3]
    unproj_pcd_color = rgb_img[0, y_screen, x_screen, :3]

    #------------------ Posed Shaped Naked SMPL -------------------# 

    pose = np.copy(pose_org)
    pose[:3] = 0

    sp = SmplPaths(gender=subject_gender)
    smpl = sp.get_smpl()
    smpl.pose[:] = pose
    smpl.betas[:10] = betas
    smpl.trans[:] = 0
    smpl_skinning = smpl.weights.r
    smpl_vertices = smpl.r
    smpl_vertices.setflags(write=1)
    smpl_vertices = r_rot_y.apply(smpl_vertices)

    registered_smpl_mesh = Meshes(verts=np2tensor(smpl_vertices).float().cuda()[None, :], faces=np2tensor(smpl.f.astype('float32')).cuda()[None, :])

    # ------------ project partial point cloud onto SMPL ----------#

    face_verts_loc = face_vertices(registered_smpl_mesh.verts_padded(), registered_smpl_mesh.faces_padded()).to('cuda').contiguous()
    face_verts_sw = face_vertices(torch.tensor(smpl_skinning)[None, :].cuda(), registered_smpl_mesh.faces_padded()).to('cuda').contiguous()

    # find closest face index from points to dense smpl mesh
    residues, pts_ind, _ = point_to_mesh_distance(unproj_pcd_loc, face_verts_loc)

    # get the cloest triangles on ref smpl mesh, [BxV, 3, 3]
    closest_triangles = torch.gather(face_verts_loc, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    # calculate closest point in the triangle, [BxV, 3]
    bary_weights = barycentric_coordinates_of_projection(unproj_pcd_loc.view(-1, 3), closest_triangles)

    # feature face tensor, [BxV, 3, dim]
    face_sw_all = torch.gather(face_verts_sw, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 24)).view(-1, 3, 24)

    # aggregate feature using barycentroc coordinate
    sw_bary = (face_sw_all * bary_weights[:, :, None]).sum(1).unsqueeze(0)  # [1, BxV, dim]
    sw_ = sw_bary.reshape(1, -1, 24)
    nn_smpl_sw = sw_.permute(0,2,1).contiguous().float()

    valid_mask = residues<0.15
    captured_pcd_loc_valid = tensor2np(unproj_pcd_loc[valid_mask])
    captured_pcd_color_valid = tensor2np(unproj_pcd_color[None, :][valid_mask])
    captured_pcd_nn_smpl_sw_valid = tensor2np(nn_smpl_sw.permute(0,2,1)[valid_mask])

    # use PyMeshLab to compute normals; Open3d and pytorch don't have good normal estimation
    m = pymeshlab.Mesh(captured_pcd_loc_valid)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "point_wo_normal")

    ms.compute_normal_for_point_clouds(k=10)
    a = ms.current_mesh()

    captured_pcd_loc_valid = a.vertex_matrix()
    captured_pcd_normal_valid = a.vertex_normal_matrix()

    #----------------- Unposed Shaped Naked SMPL -------------------#
    sp = SmplPaths(gender=subject_gender)
    shaped_neutral_smpl = sp.get_smpl()
    shaped_neutral_smpl.pose[:] = 0
    shaped_neutral_smpl.betas[:10] = betas
    shaped_neutral_smpl.trans[:] = 0
    shaped_neutral_smpl_vertices = shaped_neutral_smpl.r
    shaped_neutral_smpl_vertices.setflags(write=1)

    shaped_smpl_mesh = Meshes(verts=np2tensor(shaped_neutral_smpl_vertices).float().cuda()[None, :], faces=np2tensor(smpl.f.astype('float32')).cuda()[None, :])

    face_unposed_verts_loc = face_vertices(shaped_smpl_mesh.verts_padded(), shaped_smpl_mesh.faces_padded()).to('cuda').contiguous()

    # feature face tensor, [BxV, 3, dim]
    face_unposed_verts_all = torch.gather(face_unposed_verts_loc, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)

    # aggregate feature using barycentroc coordinate
    canonical_smpl_bary = (face_unposed_verts_all * bary_weights[:, :, None]).sum(1).unsqueeze(0)  # [1, BxV, dim]
    canonical_smpl_ = canonical_smpl_bary.reshape(1, -1, 3)
    canonical_smpl_corr = canonical_smpl_.contiguous().float()
    canonical_smpl_corr_valid = canonical_smpl_corr[valid_mask]

    location_tensor = torch.tensor([location_dict.get('left_hand_x'), location_dict.get('right_hand_x'), location_dict.get('left_foot_y'), location_dict.get('right_foot_y'), location_dict.get('head_y')]).cuda()

    # get mask
    hand_mask, feet_mask, head_mask, points_on_body = compute_smaple_on_body_mask_wo_batch(canonical_smpl_corr_valid, location_dict.get('cut_offset'), location_tensor)

    data = {'points_posed_cloth': captured_pcd_loc_valid.astype('float32').transpose(1, 0),    # scan,  points
            'normals_posed_cloth': captured_pcd_normal_valid.astype('float32').transpose(1, 0),  # scan,  normals
            'colors_posed_cloth': captured_pcd_color_valid.astype('float32').transpose(1, 0),  # scan,  normals
            'pose': pose.astype('float32'),                                   # pose parameter without root
            'pose_root': pose_inv.astype('float32'),                          # root pose
            'betas': betas.astype('float32'),                                 # shape parameter
            'trans': 0,                                                       # translation parameter
            'points_ref_smpl': tensor2np(canonical_smpl_corr_valid).astype('float32').transpose(1, 0),  # correspondence of scan on canonical space
            'skinning_weights': captured_pcd_nn_smpl_sw_valid.astype('float32').transpose(1, 0),
            'hand_mask': tensor2np(hand_mask),
            'feet_mask': tensor2np(feet_mask),
            'head_mask': tensor2np(head_mask),
            'rot_vector': r_rot_y.as_rotvec(),
            'depth_img': tensor2np(fragments_buff.zbuf[0, :, :, 0]).astype('float32'),
            'color_img': tensor2np(rgb_img[0, :, :, :3]).astype('float32')}          
    debug=False
    if debug:
        save_name_scan = "/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/pc_scan.ply"
        save_name_corr = "/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/pc_corr.ply"
        save_name_on_body_cano = "/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/pc_on_body_cano.ply"
        save_name_on_body_posed = "/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/pc_on_body_posed.ply"
        save_name_cano_verts = "/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/pc_v_cano.ply"
        save_name_scan_verts = "/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/pc_v_posed.ply"
        generate_point_cloud(captured_pcd_loc_valid, save_name_scan, colors=captured_pcd_color_valid)
        generate_point_cloud(tensor2np(canonical_smpl_corr_valid), save_name_corr, colors=captured_pcd_color_valid)
        generate_point_cloud(tensor2np(points_on_body), save_name_on_body_cano)
        generate_point_cloud(tensor2np(np2tensor(captured_pcd_loc_valid)[~hand_mask]), save_name_on_body_posed)
        generate_point_cloud(tensor2np(shaped_smpl_mesh.verts_packed()), save_name_cano_verts)
        generate_point_cloud(tensor2np(textured_buff_mesh.verts_packed()), save_name_scan_verts)
   
    np.save(save_name, data)
    
    return 0


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seq', default='00096')
    parser.add_argument('--version', default='depth_rot_trans')
    parser.add_argument('--rot', default='const')

    args = parser.parse_args()

    buff_path = '/mnt/qb/work/ponsmoll/yxue80/project/Data/BuFF/buff_release/'
    
    if args.version == 'depth_rot_trans':
        if args.rot == 'const':
            save_path = '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/Data/BuFF/buff_release_rot_const/'
            print("Preprocessing depth with constent rot view around y axis")
        else:
            assert(False)

    with open(buff_path + 'misc/subj_genders.pkl', 'rb') as f:
        gender_table_buff = pickle.load(f)

    subject_idx = str(args.seq)

    skinner = SkinModel().cuda()
    inv_skinner = InvSkinModel().cuda()

    ######################################################################################################
    # set up simulated Kinect Azure camera, render depth map and reconstruct the depth point cloud   
    from pytorch3d.renderer import (
    PerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRasterizer, SoftPhongShader, PointLights, Materials, MeshRendererWithFragments, TexturesVertex
    )
    # R, T = look_at_view_transform(2.7, 0, 0)
    R = torch.tensor([[[-1.,  0.,  0.],
                    [ 0.,  1.,  0.],
                    [ 0.,  0., -1.]]]).cuda()
    T = torch.tensor([[-0.0000, 0.2, 2.7000]]).cuda()

    # BEHAVE: Kinect azure depth camera 0 parameters
    intrinsics = torch.tensor([[[502.9671325683594*scale, 0.0, 322.229736328125*scale, 0.0],
                                [0.0, 503.04168701171875*scale, 329.3377685546875*scale, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0, 0.0]]]).cuda()
    image_size_height = 576*scale
    image_size_width = 640*scale
    image_size = torch.tensor([image_size_height, image_size_width]).unsqueeze(0).cuda()

    cameras_front = PerspectiveCameras(R=R, T=T, in_ndc=False, K=intrinsics, image_size=image_size).cuda()

    lights = PointLights(location=((0, 0, 5),), ambient_color=[[0.7, 0.7, 0.7]],
                             diffuse_color=[[0.3, 0.3, 0.3]], specular_color=((0.0, 0.0, 0.0),)).cuda()

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0, 0, 0))
    material = Materials(ambient_color=[[1, 1, 1]], diffuse_color=[[1, 1, 1]], specular_color=[[0, 0, 0]]).cuda()
    
    raster_settings_kinect = RasterizationSettings(
            image_size=(image_size_height, image_size_width),
            blur_radius=0,
            faces_per_pixel=1,  # number of faces to save per pixel
            bin_size=0
        )

    rasterizer = MeshRasterizer(
        cameras=cameras_front, 
        raster_settings=raster_settings_kinect
    )

    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras_front,
            raster_settings=raster_settings_kinect
        ),
            shader=SoftPhongShader(
            lights=lights,
            blend_params=blend_params,
            materials=material,
            cameras=cameras_front
        )
    )
    ########################################################################################################

    left_hand_vertex_index = 2005 # palm center
    right_hand_vertex_index = 5509 # palm center
    left_foot_vertex_index = 3392 # ankle, ~5 cm above the ground
    right_foot_vertex_index = 6730 # ankle, ~5 cm above the ground
    head_vertex_index = 6493 # neck top
    cut_offset = 0.03

    from pytorch3d.io import load_ply
    file = load_ply(f='/mnt/qb/work/ponsmoll/yxue80/project/Data/BuFF/buff_release/minimal_body_shape/'+args.seq+'/'+args.seq+'_minimal.ply')
    subject_mesh = Meshes(verts=[file[0].float()], faces=[file[1].float()]).cuda()
    smpl_vertices = subject_mesh.verts_packed()

    left_hand_x = smpl_vertices[left_hand_vertex_index, 0]
    right_hand_x = smpl_vertices[right_hand_vertex_index, 0]
    left_foot_y = smpl_vertices[left_foot_vertex_index, 1]
    right_foot_y = smpl_vertices[right_foot_vertex_index, 1]
    head_y = smpl_vertices[head_vertex_index, 1]

    body_part_location = {
        'left_hand_x': left_hand_x,
        'right_hand_x': right_hand_x,
        'left_foot_y': left_foot_y,
        'right_foot_y': right_foot_y,
        'head_y': head_y,
        'cut_offset': cut_offset
    }

    from tqdm import tqdm
    import time
    gender = gender_table_buff[subject_idx]
    sequences_path = buff_path + 'sequences/' + str(subject_idx)
    save_sequences_path = save_path + 'sequences/' + str(subject_idx)

    for action in tqdm(os.listdir(sequences_path)):
        action_path = os.path.join(sequences_path, action)
        save_action_path = os.path.join(save_sequences_path, action)
        
        i = 0

        for ply_file_name in tqdm(os.listdir(action_path)):
            
            if ply_file_name.endswith('.ply'):
                frame_texture_mesh_path = os.path.join(action_path, ply_file_name)
                shape_pkl_path = buff_path + 'minimal_body_shape/' + str(subject_idx) + '/' + str(subject_idx) + '_param.pkl'
                save_name = os.path.split(ply_file_name)[1].split('.')[0] + '_' + os.path.split(ply_file_name)[1].split('.')[1] + '.npy'
                
                if not os.path.exists(save_action_path):
                    os.makedirs(save_action_path)
                
                save_path = os.path.join(save_action_path, save_name)

                if os.path.exists(save_path):
                    continue

                # canonical_scan_as_debug(gender, frame_npz_path, shape_pkl_path, debug=True)
                
                if args.version == 'depth_rot_trans':
                    rot_euler = 0
                    if args.rot == 'const':
                        rot_euler = -360 * i * 2 / (len(os.listdir(action_path)))
                        i += 1
                    else:
                        assert(False)
                    r_rot_y = ScipyRot.from_euler('y', rot_euler, degrees=True)
                    buff_preprocessing_depth_rot_trans(gender, frame_texture_mesh_path, shape_pkl_path, save_path, rasterizer=rasterizer, renderer=renderer, r_rot_y=r_rot_y, skinner=skinner, inv_skinner=inv_skinner, location_dict=body_part_location, debug=False)

            time.sleep(0.01)

