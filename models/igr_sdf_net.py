import os
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
# import functools

from skimage import measure

import numpy as np


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))

class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

class Embedder:
    def __init__(self, **kwargs):
        
        self.kwargs = kwargs
        self.create_embedding_fn()
        
        
    def create_embedding_fn(self):
        
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0, input_dims=3):
    
    if i == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def reconstruction(net, cuda, resolution=256, thresh=0.5, b_min=-1, b_max=1, texture_net=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # Then we define the lambda function for cell evaluation
    color_flag = False if texture_net is None else True

    def eval_func(points):
        samples = points.unsqueeze(0)
        pred = net.query(samples)[0][0]
        return pred

    def batch_eval(points, num_samples=4096):
        num_pts = points.shape[1]
        sdf = []
        num_batches = num_pts // num_samples
        for i in range(num_batches):
            sdf.append(
                eval_func(points[:, i * num_samples:i * num_samples + num_samples])
            )
        if num_pts % num_samples:
            sdf.append(
                eval_func(points[:, num_batches * num_samples:])
            )
        if num_pts == 0:
            return None
        sdf = torch.cat(sdf)
        return sdf

    # Then we evaluate the grid    
    max_level = int(math.log2(resolution))
    sdf = eval_progressive(batch_eval, 4, max_level, cuda, b_min, b_max, thresh)

    # calculate matrix
    mat = np.eye(4)
    length = b_max - b_min
    mat[0, 0] = length[0] / sdf.shape[0]
    mat[1, 1] = length[1] / sdf.shape[1]
    mat[2, 2] = length[2] / sdf.shape[2]
    mat[0:3, 3] = b_min

    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf, thresh, gradient_direction='ascent')
    except:
        print('error cannot marching cubes')
        return -1

    # transform verts into world coordinate system
    verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
    verts = verts.T
    if np.linalg.det(mat) > 0:
        faces = faces[:,[0,2,1]]

    if color_flag:
        torch_verts = torch.Tensor(verts).unsqueeze(0).permute(0,2,1).to(cuda)

        with torch.no_grad():
            _, last_layer_feature, point_local_feat = net.query(torch_verts, return_last_layer_feature=True)
            vertex_colors = texture_net.query(point_local_feat, last_layer_feature)
            vertex_colors = vertex_colors.squeeze(0).permute(1,0).detach().cpu().numpy()
        return verts, faces, normals, values, vertex_colors
    else:
        return verts, faces, normals, values

def eval_progressive(batch_eval, min_level, max_level, cuda, bmin, bmax, thresh=0.0):
    steps = [i for i in range(min_level, max_level+1)]

    b_min = torch.tensor(bmin).to(cuda)
    b_max = torch.tensor(bmax).to(cuda)

    def build_smooth_conv3D(in_channels=1, out_channels=1, kernel_size=3, padding=1):
        smooth_conv = torch.nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, padding=padding
        )
        smooth_conv.weight.data = torch.ones(
            (kernel_size, kernel_size, kernel_size), 
            dtype=torch.float32
        ).reshape(in_channels, out_channels, kernel_size, kernel_size, kernel_size) / (kernel_size**3)
        smooth_conv.bias.data = torch.zeros(out_channels)
        return smooth_conv

    # init
    smooth_conv3x3 = build_smooth_conv3D(in_channels=1, out_channels=1, kernel_size=3, padding=1).to(cuda)

    arrange = torch.linspace(0, 2**steps[-1], 2**steps[0]+1).long().to(cuda)
    coords = torch.stack(torch.meshgrid([
        arrange, arrange, arrange
    ])) # [3, 2**step+1, 2**step+1, 2**step+1]
    coords = coords.view(3, -1).t() # [N, 3]
    calculated = torch.zeros(
        (2**steps[-1]+1, 2**steps[-1]+1, 2**steps[-1]+1), dtype=torch.bool
    ).to(cuda)
        
    gird8_offsets = torch.stack(torch.meshgrid([
        torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1])
    ])).int().to(cuda).view(3, -1).t() #[27, 3]

    with torch.no_grad():
        for step in steps:
            resolution = 2**step + 1
            stride = 2**(steps[-1]-step)

            if step == steps[0]:
                coords2D = (coords.float() / (2**steps[-1]+1) * (b_max - b_min) + b_min).float()
                sdf_all = batch_eval(
                    coords2D.t(),
                ).view(resolution, resolution, resolution)
                coords_accum = coords / stride
                coords_accum = coords_accum.long()
                calculated[coords[:, 0], coords[:, 1], coords[:, 2]] = True

            else:
                valid = F.interpolate(
                    (sdf_all>thresh).view(1, 1, *sdf_all.size()).float(), 
                    size=resolution, mode="trilinear", align_corners=True
                )[0, 0]
                
                sdf_all = F.interpolate(
                    sdf_all.view(1, 1, *sdf_all.size()), 
                    size=resolution, mode="trilinear", align_corners=True
                )[0, 0]

                coords_accum *= 2

                is_boundary = (valid > 0.0) & (valid < 1.0)
                is_boundary = smooth_conv3x3(is_boundary.float().view(1, 1, *is_boundary.size()))[0, 0] > 0

                is_boundary[coords_accum[:, 0], coords_accum[:, 1], coords_accum[:, 2]] = False

                # coords = is_boundary.nonzero() * stride
                coords = torch.nonzero(is_boundary) * stride
                coords2D = (coords.float() / (2**steps[-1]+1) * (b_max - b_min) + b_min).float()
                # coords2D = coords.float() / (2**steps[-1]+1)
                sdf = batch_eval(
                    coords2D.t(), 
                ) #[N]
                if sdf is not None:
                    sdf_all[is_boundary] = sdf
                voxels = coords / stride
                voxels = voxels.long()
                coords_accum = torch.cat([
                    voxels, 
                    coords_accum
                ], dim=0).unique(dim=0)
                calculated[coords[:, 0], coords[:, 1], coords[:, 2]] = True

                for n_iter in range(14):
                    sdf_valid = valid[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
                    idxs_danger = ((sdf_valid==1) & (sdf<thresh)) | ((sdf_valid==0) & (sdf>thresh)) #[N,]
                    coords_danger = coords[idxs_danger, :] #[N, 3]
                    if coords_danger.size(0) == 0:
                        break

                    coords_arround = coords_danger.int() + gird8_offsets.view(-1, 1, 3) * stride
                    coords_arround = coords_arround.reshape(-1, 3).long()
                    coords_arround = coords_arround.unique(dim=0)
                    
                    coords_arround[:, 0] = coords_arround[:, 0].clamp(0, calculated.size(0)-1)
                    coords_arround[:, 1] = coords_arround[:, 1].clamp(0, calculated.size(1)-1)
                    coords_arround[:, 2] = coords_arround[:, 2].clamp(0, calculated.size(2)-1)

                    coords = coords_arround[
                        calculated[coords_arround[:, 0], coords_arround[:, 1], coords_arround[:, 2]] == False
                    ]
                    
                    if coords.size(0) == 0:
                        break
                    
                    coords2D = (coords.float() / (2**steps[-1]+1) * (b_max - b_min) + b_min).float()
                    # coords2D = coords.float() / (2**steps[-1]+1)
                    sdf = batch_eval(
                        coords2D.t(), 
                    ) #[N]

                    voxels = coords / stride
                    voxels = voxels.long()
                    sdf_all[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = sdf
                    
                    coords_accum = torch.cat([
                        voxels, 
                        coords_accum
                    ], dim=0).unique(dim=0)
                    calculated[coords[:, 0], coords[:, 1], coords[:, 2]] = True

        return sdf_all.data.cpu().numpy()

def condition_reconstruction(net, cuda, feature, resolution=256, thresh=0.5, b_min=-1, b_max=1, texture_net=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # Then we define the lambda function for cell evaluation
    color_flag = False if texture_net is None else True

    def eval_func(points, feature):
        samples = points.unsqueeze(0) # [1, 3, N]
        samples_feat = torch.cat([samples, feature.repeat((1, samples.shape[2], 1)).permute(0, 2, 1)], dim=1)
        pred = net.query(samples_feat)[0][0]
        return pred

    def batch_eval(points, feature, num_samples=4096):
        num_pts = points.shape[1]
        sdf = []
        num_batches = num_pts // num_samples
        for i in range(num_batches):
            sdf.append(
                eval_func(points[:, i * num_samples:i * num_samples + num_samples], feature)
            )
        if num_pts % num_samples:
            sdf.append(
                eval_func(points[:, num_batches * num_samples:], feature)
            )
        if num_pts == 0:
            return None
        sdf = torch.cat(sdf)
        return sdf

    def eval_progressive_new(batch_eval, feature, min_level, max_level, cuda, bmin, bmax, thresh=0.0):
        steps = [i for i in range(min_level, max_level+1)]

        b_min = torch.tensor(bmin).to(cuda)
        b_max = torch.tensor(bmax).to(cuda)

        def build_smooth_conv3D(in_channels=1, out_channels=1, kernel_size=3, padding=1):
            smooth_conv = torch.nn.Conv3d(
                in_channels=in_channels, out_channels=out_channels, 
                kernel_size=kernel_size, padding=padding
            )
            smooth_conv.weight.data = torch.ones(
                (kernel_size, kernel_size, kernel_size), 
                dtype=torch.float32
            ).reshape(in_channels, out_channels, kernel_size, kernel_size, kernel_size) / (kernel_size**3)
            smooth_conv.bias.data = torch.zeros(out_channels)
            return smooth_conv

        # init
        smooth_conv3x3 = build_smooth_conv3D(in_channels=1, out_channels=1, kernel_size=3, padding=1).to(cuda)

        arrange = torch.linspace(0, 2**steps[-1], 2**steps[0]+1).long().to(cuda)
        coords = torch.stack(torch.meshgrid([
            arrange, arrange, arrange
        ])) # [3, 2**step+1, 2**step+1, 2**step+1]
        coords = coords.view(3, -1).t() # [N, 3]
        calculated = torch.zeros(
            (2**steps[-1]+1, 2**steps[-1]+1, 2**steps[-1]+1), dtype=torch.bool
        ).to(cuda)
            
        gird8_offsets = torch.stack(torch.meshgrid([
            torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1])
        ])).int().to(cuda).view(3, -1).t() #[27, 3]

        with torch.no_grad():
            for step in steps:
                resolution = 2**step + 1
                stride = 2**(steps[-1]-step)

                if step == steps[0]:
                    coords2D = (coords.float() / (2**steps[-1]+1) * (b_max - b_min) + b_min).float()
                    sdf_all = batch_eval(
                        coords2D.t(), feature
                    ).view(resolution, resolution, resolution)
                    coords_accum = coords / stride
                    coords_accum = coords_accum.long()
                    calculated[coords[:, 0], coords[:, 1], coords[:, 2]] = True

                else:
                    valid = F.interpolate(
                        (sdf_all>thresh).view(1, 1, *sdf_all.size()).float(), 
                        size=resolution, mode="trilinear", align_corners=True
                    )[0, 0]
                    
                    sdf_all = F.interpolate(
                        sdf_all.view(1, 1, *sdf_all.size()), 
                        size=resolution, mode="trilinear", align_corners=True
                    )[0, 0]

                    coords_accum *= 2

                    is_boundary = (valid > 0.0) & (valid < 1.0)
                    is_boundary = smooth_conv3x3(is_boundary.float().view(1, 1, *is_boundary.size()))[0, 0] > 0

                    is_boundary[coords_accum[:, 0], coords_accum[:, 1], coords_accum[:, 2]] = False

                    # coords = is_boundary.nonzero() * stride
                    coords = torch.nonzero(is_boundary) * stride
                    coords2D = (coords.float() / (2**steps[-1]+1) * (b_max - b_min) + b_min).float()
                    # coords2D = coords.float() / (2**steps[-1]+1)
                    sdf = batch_eval(
                        coords2D.t(), feature
                    ) #[N]
                    if sdf is not None:
                        sdf_all[is_boundary] = sdf
                    voxels = coords / stride
                    voxels = voxels.long()
                    coords_accum = torch.cat([
                        voxels, 
                        coords_accum
                    ], dim=0).unique(dim=0)
                    calculated[coords[:, 0], coords[:, 1], coords[:, 2]] = True

                    for n_iter in range(14):
                        sdf_valid = valid[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
                        idxs_danger = ((sdf_valid==1) & (sdf<thresh)) | ((sdf_valid==0) & (sdf>thresh)) #[N,]
                        coords_danger = coords[idxs_danger, :] #[N, 3]
                        if coords_danger.size(0) == 0:
                            break

                        coords_arround = coords_danger.int() + gird8_offsets.view(-1, 1, 3) * stride
                        coords_arround = coords_arround.reshape(-1, 3).long()
                        coords_arround = coords_arround.unique(dim=0)
                        
                        coords_arround[:, 0] = coords_arround[:, 0].clamp(0, calculated.size(0)-1)
                        coords_arround[:, 1] = coords_arround[:, 1].clamp(0, calculated.size(1)-1)
                        coords_arround[:, 2] = coords_arround[:, 2].clamp(0, calculated.size(2)-1)

                        coords = coords_arround[
                            calculated[coords_arround[:, 0], coords_arround[:, 1], coords_arround[:, 2]] == False
                        ]
                        
                        if coords.size(0) == 0:
                            break
                        
                        coords2D = (coords.float() / (2**steps[-1]+1) * (b_max - b_min) + b_min).float()
                        # coords2D = coords.float() / (2**steps[-1]+1)
                        sdf = batch_eval(
                            coords2D.t(), feature
                        ) #[N]

                        voxels = coords / stride
                        voxels = voxels.long()
                        sdf_all[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = sdf
                        
                        coords_accum = torch.cat([
                            voxels, 
                            coords_accum
                        ], dim=0).unique(dim=0)
                        calculated[coords[:, 0], coords[:, 1], coords[:, 2]] = True

            return sdf_all.data.cpu().numpy()

    # Then we evaluate the grid    
    max_level = int(math.log2(resolution))
    sdf = eval_progressive_new(batch_eval, feature, 4, max_level, cuda, b_min, b_max, thresh)

    # calculate matrix
    mat = np.eye(4)
    length = b_max - b_min
    mat[0, 0] = length[0] / sdf.shape[0]
    mat[1, 1] = length[1] / sdf.shape[1]
    mat[2, 2] = length[2] / sdf.shape[2]
    mat[0:3, 3] = b_min

    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf, thresh, gradient_direction='ascent')
    except:
        print('error cannot marching cubes')
        return -1

    # transform verts into world coordinate system
    verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
    verts = verts.T
    if np.linalg.det(mat) > 0:
        faces = faces[:,[0,2,1]]

    if color_flag:
        torch_verts = torch.Tensor(verts).unsqueeze(0).permute(0,2,1).to(cuda)

        with torch.no_grad():
            _, last_layer_feature, point_local_feat = net.query(torch_verts, return_last_layer_feature=True)
            vertex_colors = texture_net.query(point_local_feat, last_layer_feature)
            vertex_colors = vertex_colors.squeeze(0).permute(1,0).detach().cpu().numpy()
        return verts, faces, normals, values, vertex_colors
    else:
        return verts, faces, normals, values




class MLP(nn.Module):
    def __init__(self, filter_channels, res_layers=[], last_op=None, nlactiv='softplus', norm='weight'):
        super(MLP, self).__init__()

        self.filters = nn.ModuleList()
        
        if last_op == 'sigmoid':
            self.last_op = nn.Sigmoid()
        elif last_op == 'tanh':
            self.last_op = nn.Tanh()
        elif last_op == 'softmax':
            self.last_op = nn.Softmax(dim=1)
        else:
            self.last_op = None

        self.res_layers = res_layers
        for l in range(0, len(filter_channels) - 1):
            if l in res_layers:
                if norm == 'weight' and l != len(filter_channels) - 2:
                    self.filters.append(
                        nn.utils.weight_norm(nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1)))
                else:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
            else:
                if norm == 'weight' and l != len(filter_channels) - 2:
                    self.filters.append(nn.utils.weight_norm(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1)))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))
        
        self.nlactiv = None
        if nlactiv == 'leakyrelu':
            self.nlactiv = nn.LeakyReLU()
        elif nlactiv == 'softplus':
            self.nlactiv = nn.Softplus(beta=100, threshold=20)
        elif nlactiv == 'relu':
            self.nlactiv = nn.ReLU()
        elif nlactiv == 'mish':
            self.nlactiv = Mish()
        elif nlactiv == 'elu':
            self.nlactiv = nn.ELU(0.1)
        elif nlactiv == 'sin':
            self.nlactiv = Sin()

    def forward(self, feature, return_last_layer_feature = False):
        '''
        :param feature: list of [BxC_inxN] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        y = feature
        y0 = feature
        last_layer_feature = None
        for i, f in enumerate(self.filters):
            if i in self.res_layers:
                y = f(torch.cat([y, y0], 1))
            else:
                y = f(y)

            if i != len(self.filters) - 1 and self.nlactiv is not None:
                y = self.nlactiv(y)

            if i == len(self.filters) - 2 and return_last_layer_feature:
                last_layer_feature = y.clone()
                last_layer_feature = last_layer_feature.detach()

        if self.last_op:
            y = self.last_op(y)

        if not return_last_layer_feature:
            return y
        else:
            return y, last_layer_feature

class BaseIMNet3d(nn.Module):
    # implicit net
    def __init__(self, positional_encoding=False, cond=0, bbox_min=[-1.0,-1.0,-1.0],
                 bbox_max=[1.0,1.0,1.0]
                 ):
        super(BaseIMNet3d, self).__init__()

        self.name = 'base_imnet3d'
        mlp_ch_dim = [3+cond, 512, 512, 512, 343, 512, 512, 1]

        if positional_encoding: # use positional encoding
            self.embedder, mlp_ch_dim[0] = get_embedder(4, input_dims=mlp_ch_dim[0])
        else:
            self.embedder = None

        self.mlp = MLP(
            filter_channels=mlp_ch_dim,
            res_layers=[4],
            last_op=None,
            nlactiv='softplus',
            norm='weight')

        init_net(self)

        self.register_buffer('bbox_min', torch.Tensor(bbox_min)[None,:,None])
        self.register_buffer('bbox_max', torch.Tensor(bbox_max)[None,:,None])


    def query(self, points):
        '''
        Given 3D points, query the network predictions for each point.
        args:
            points: (B, 3, N)
        return:
            (B, C, N)
        '''

        if self.embedder is not None:
            embedded_points = self.embedder(points.permute(0,2,1)).permute(0,2,1)

        return self.mlp(embedded_points)

    def forward(self, points):
        
        return self.query(points)

class IGRSDFNet(BaseIMNet3d):
    def __init__(self, cond, bbox_min, bbox_max):
        super(IGRSDFNet, self).__init__(cond=cond, bbox_min=bbox_min, bbox_max=bbox_max)

        self.name = 'neural_sdf_bpsigr'

    def normalize_points(self, query_points, bmin=None, bmax=None):

        N = query_points.size(2)

        if bmin is None:
            bmin = self.bbox_min
        if bmax is None:
            bmax = self.bbox_max

        points = query_points[:, :3, :] # [B, 3, N]
        global_feat = query_points[:, 3:, :] # [B, 256, N]

        points_nc3d = 2.0 * (points - bmin) / (bmax - bmin) - 1.0 # normalized coordiante
        points_nc3d = points_nc3d.clamp(min=-1.0, max=1.0)

        return points_nc3d

    def query(self, query_points, bmin=None, bmax=None):
        '''
        Given 3D points, query the network predictions for each point.
        args:
            points: (B, 3, N)
        return:
            (B, C, N)
        '''
        N = query_points.size(2)

        if bmin is None:
            bmin = self.bbox_min
        if bmax is None:
            bmax = self.bbox_max

        points = query_points[:, :3, :] # [B, 3, N]
        global_feat = query_points[:, 3:, :] # [B, 256, N]

        points_nc3d = 2.0 * (points - bmin) / (bmax - bmin) - 1.0 # normalized coordiante
        points_nc3d = points_nc3d.clamp(min=-1.0, max=1.0)

        # mask for valid input points (inside BB)
        in_bbox = (points_nc3d[:, 0] >= -1.0) & (points_nc3d[:, 0] <= 1.0) &\
                  (points_nc3d[:, 1] >= -1.0) & (points_nc3d[:, 1] <= 1.0) &\
                  (points_nc3d[:, 2] >= -1.0) & (points_nc3d[:, 2] <= 1.0)
        in_bbox = in_bbox[:,None].float()
        
        points_scaled_query = torch.cat((points_nc3d, global_feat), dim=1)

        w0 = 1.0

        embedded_points = points_scaled_query.clone()
        if self.embedder is not None:
            embedded_points = self.embedder(points_scaled_query.permute(0,2,1)).permute(0,2,1)

        return -in_bbox*self.mlp(w0*embedded_points)-(1.0-in_bbox)

    def compute_normal(self, query_points, bmin=None, bmax=None, normalize=False, return_pred=False):
        '''
        since image sampling operation does not have second order derivative,
        normal can be computed only via finite difference (forward differentiation)
        '''
        N = query_points.size(2)

        if bmin is None:
            bmin = self.bbox_min
        if bmax is None:
            bmax = self.bbox_max

        points = query_points[:, :3, :] # [B, 3, N]
        global_feat = query_points[:, 3:, :] # [B, 256, N]

        with torch.enable_grad():
            points.requires_grad_()

            w0 = 1.0

            points_nc3d = 2.0 * (points - bmin) / (bmax - bmin) - 1.0 # normalized coordiante
            points_nc3d = points_nc3d.clamp(min=-1.0, max=1.0)

            points_scaled_query = torch.cat((points_nc3d, global_feat), dim=1)
            
            embedded_points = points_scaled_query.clone()
            if self.embedder is not None:
                embedded_points = self.embedder(points_scaled_query.permute(0,2,1)).permute(0,2,1)

            pred = self.mlp(w0*embedded_points)
            normal = autograd.grad(
                    [pred.sum()], [points], 
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            
            if normalize:
                normal = F.normalize(normal, dim=1, eps=1e-6)

            if return_pred:
                return normal, pred
            else:
                return normal

    def get_error(self, res):
        '''
        based on https://arxiv.org/pdf/2002.10099.pdf
        '''
        err_dict = {}

        error_ls = nn.L1Loss()(res['sdf_surface'], torch.zeros_like(res['sdf_surface']))
        error_nml = torch.norm(res['nml_surface'] - res['nml_gt'], p=2, dim=1).mean()
        
        nml_reg = torch.cat((res['nml_surface']), dim=2)
        # error_reg = self.opt['lambda_reg'] * (torch.norm(res['nml_igr'], p=2, dim=1) - 1).mean().pow(2)
        error_reg = (torch.norm(nml_reg, p=2, dim=1) - 1).pow(2).mean()
            
        err_dict['LS'] = error_ls.item()
        err_dict['N'] = error_nml.item()
        err_dict['R'] = error_reg.item()
        error = error_ls + error_nml + error_reg

        return error, err_dict


    def forward(self, pts_surface):
        '''
        args:
            feat: (B, C, D, H, W)
            pts_surface: (B, 3, N)
            pts_body: (B, 3, N*)
            pts_bbox: (B, 3, N**)
            normals: (B, 3, N)
        '''
        # set volumetric feature
        nml_surface, sdf_surface = self.compute_normal(query_points=pts_surface, 
                                                        normalize=False,
                                                        return_pred=True)


        res = {'sdf_surface': sdf_surface, 'nml_surface': nml_surface}

        return res


if __name__ == "__main__":

    class MyDataParallel(torch.nn.DataParallel):
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)


    sdf_net = MyDataParallel(IGRSDFNet())

    checkpoint_path = '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/experiments/PoseImplicit_exp_id_71/checkpoints/checkpoint_epoch_900.tar'
    checkpoint = torch.load(checkpoint_path)

    sdf_net.load_state_dict(checkpoint['sdf_state_dict'])

    verts, faces, normals, values = reconstruction(sdf_net, torch.device("cuda"), resolution=256, thresh=0.0, b_min=np.array([-1.5, -1.5, -1.5]), b_max=np.array([1.5, 1.5, 1.5]), texture_net=None)

    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    mesh.visual
    print("a")