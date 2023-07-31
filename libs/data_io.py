import torch
import numpy as np

from libs.torch_functions import tensor2np
from os.path import join

def generate_point_cloud(points, save_name, colors=None):

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_name, pcd)


def save_result_ply(save_dir, points, faces=None, colors=None, normals=None, patch_color=None, 
                    texture=None, coarse_pts=None, fine_pts=None, gt=None, gt_normals=None, epoch=None):
    # works on single pcl, i.e. [#num_pts, 3], no batch dimension

    normal_fn = 'pred.ply'
    normal_fn = join(save_dir, normal_fn)
    points = tensor2np(points)

    if normals is None: 
        customized_export_ply(normal_fn, v=points)

    if faces is not None and colors is not None:
        faces = tensor2np(faces)
        colors = tensor2np(colors) * 255
        meshcolor_fn = normal_fn.replace('pred.ply', 'pred_mesh.ply')
        customized_export_ply(meshcolor_fn, v=points, f=faces, v_c=colors) # use normals as mesh verts color

    if normals is not None and faces is None:
        normals = tensor2np(normals)
        color_normal = vertex_normal_2_vertex_color(normals)
        customized_export_ply(normal_fn, v=points, v_n=normals, v_c=color_normal)

    if patch_color is not None:
        patch_color = tensor2np(patch_color)
        if patch_color.max() < 1.1:
            patch_color = (patch_color*255.).astype(np.ubyte)
        pcolor_fn = normal_fn.replace('pred.ply', 'pred_patchcolor.ply')
        customized_export_ply(pcolor_fn, v=points, v_c=patch_color)
    
    if texture is not None:
        texture = tensor2np(texture)
        if texture.max() < 1.1:
            texture = (texture*255.).astype(np.ubyte)
        texture_fn = normal_fn.replace('pred.ply', 'pred_texture.ply')
        customized_export_ply(texture_fn, v=points, v_c=texture)

    if coarse_pts is not None:
        coarse_pts = tensor2np(coarse_pts)
        coarse_fn = normal_fn.replace('pred.ply', 'coarse_corr.ply')
        customized_export_ply(coarse_fn, v=coarse_pts)

    if fine_pts is not None:
        fine_pts = tensor2np(fine_pts)
        fine_fn = normal_fn.replace('pred.ply', 'fine_corr.ply')
        customized_export_ply(fine_fn, v=fine_pts)

    if gt is not None: 
        gt = tensor2np(gt)
        gt_fn = normal_fn.replace('pred.ply', 'gt.ply')
        customized_export_ply(gt_fn, v=gt, v_n=gt_normals)


def save_multiple_ply(save_dir, a_ply=None, b_ply=None, c_ply=None, d_ply=None, e_ply=None, f_ply=None, epoch=None):
    # works on single pcl, i.e. [#num_pts, 3], no batch dimension

    normal_fn = 'a.ply'
    normal_fn = join(save_dir, normal_fn)
   
    if a_ply is not None:
        points = tensor2np(a_ply)
        customized_export_ply(normal_fn, v=points)

    if b_ply is not None:
        coarse_ply = tensor2np(b_ply)
        coarse_fn = normal_fn.replace('a.ply', 'b.ply')
        customized_export_ply(coarse_fn, v=coarse_ply)

    if c_ply is not None:
        coarse_ply = tensor2np(c_ply)
        coarse_fn = normal_fn.replace('a.ply', 'c.ply')
        customized_export_ply(coarse_fn, v=coarse_ply)

    if d_ply is not None:
        coarse_ply = tensor2np(d_ply)
        coarse_fn = normal_fn.replace('a.ply', 'd.ply')
        customized_export_ply(coarse_fn, v=coarse_ply)

    if e_ply is not None:
        coarse_ply = tensor2np(e_ply)
        coarse_fn = normal_fn.replace('a.ply', 'e.ply')
        customized_export_ply(coarse_fn, v=coarse_ply)

    if f_ply is not None:
        coarse_ply = tensor2np(f_ply)
        coarse_fn = normal_fn.replace('a.ply', 'f.ply')
        customized_export_ply(coarse_fn, v=coarse_ply)


def vertex_normal_2_vertex_color(vertex_normal):
    # Normalize vertex normal
    if torch.is_tensor(vertex_normal):
        vertex_normal = vertex_normal.detach().cpu().numpy()
    normal_length = ((vertex_normal**2).sum(1))**0.5
    normal_length = normal_length.reshape(-1, 1)
    vertex_normal /= normal_length
    # Convert normal to color:
    color = vertex_normal * 255/2.0 + 128
    return color.astype(np.ubyte)


def customized_export_ply(outfile_name, v, f = None, v_n = None, v_c = None, f_c = None, e = None):

    v_n_flag=False
    v_c_flag=False
    f_c_flag=False

    N_v = v.shape[0]
    assert(v.shape[1] == 3)
    if not type(v_n) == type(None):
        assert(v_n.shape[0] == N_v)
        if type(v_n) == 'torch.Tensor':
            v_n = v_n.detach().cpu().numpy()
        v_n_flag = True
    if not type(v_c) == type(None):
        assert(v_c.shape[0] == N_v)
        v_c_flag = True
        if v_c.shape[1] == 3:
            # warnings.warn("Vertex color does not provide alpha channel, use default alpha = 255")
            alpha_channel = np.zeros((N_v, 1), dtype = np.ubyte)+255
            v_c = np.hstack((v_c, alpha_channel))

    N_f = 0
    if not type(f) == type(None):
        N_f = f.shape[0]
        assert(f.shape[1] == 3)
        if not type(f_c) == type(None):
            assert(f_c.shape[0] == f.shape[0])
            f_c_flag = True
            if f_c.shape[1] == 3:
                # warnings.warn("Face color does not provide alpha channel, use default alpha = 255")
                alpha_channel = np.zeros((N_f, 1), dtype = np.ubyte)+255
                f_c = np.hstack((f_c, alpha_channel))
    N_e = 0
    if not type(e) == type(None):
        N_e = e.shape[0]

    with open(outfile_name, 'w') as file:
        # Header
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex %d\n'%(N_v))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')

        if v_n_flag:
            file.write('property float nx\n')
            file.write('property float ny\n')
            file.write('property float nz\n')
        if v_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        file.write('element face %d\n'%(N_f))
        file.write('property list uchar int vertex_indices\n')
        if f_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        if not N_e == 0:
            file.write('element edge %d\n'%(N_e))
            file.write('property int vertex1\n')
            file.write('property int vertex2\n')

        file.write('end_header\n')

        # Main body:
        # Vertex
        if v_n_flag and v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f %d %d %d %d\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_n[i,0], v_n[i,1], v_n[i,2], \
                    v_c[i,0], v_c[i,1], v_c[i,2], v_c[i,3]))
        elif v_n_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_n[i,0], v_n[i,1], v_n[i,2]))
        elif v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %d %d %d %d\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_c[i,0], v_c[i,1], v_c[i,2], v_c[i,3]))
        else:
            for i in range(0, N_v):
                file.write('%f %f %f\n'%\
                    (v[i,0], v[i,1], v[i,2]))
        # Face
        if f_c_flag:
            for i in range(0, N_f):
                file.write('3 %d %d %d %d %d %d %d\n'%\
                    (f[i,0], f[i,1], f[i,2],\
                    f_c[i,0], f_c[i,1], f_c[i,2], f_c[i,3]))
        else:
            for i in range(0, N_f):
                file.write('3 %d %d %d\n'%\
                    (f[i,0], f[i,1], f[i,2]))

        # Edge
        if not N_e == 0:
            for i in range(0, N_e):
                file.write('%d %d\n'%(e[i,0], e[i,1]))
