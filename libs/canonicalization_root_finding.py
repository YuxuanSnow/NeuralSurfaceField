import torch
from torch.nn import functional as F

def query_weights(xc, diffuse_skinning_field, subject_garment_id, mask=None):
    # xc: [B, N, 3]

    def inv_transform_v(v, scale_grid, transl):
        """
        v: [b, n, 3]
        """
        v = v - transl[:, None, :]
        v = v / scale_grid[:, None, None]
        v = v * 2

        return v

    def get_w(p_xc, p_grid=1):
        n_batch, n_point, n_dim = p_xc.shape

        if n_batch * n_point == 0:
            return p_xc
        
        x = F.grid_sample(p_grid, # [2, 24, 256, 256, 256]
                        p_xc[:, None, None, :, :], # [2, 1, 1, N, 3],
                        align_corners=False,
                        padding_mode='border')[:, :, 0, 0]  # [B, 24, N]

        return x 

    subject_field_idx = torch.zeros_like(subject_garment_id)
    for i in range(len(subject_garment_id)):
        subject_field_idx[i] = diffuse_skinning_field.general_subject_index_numer[subject_garment_id[i].item()]

    weights_grid = diffuse_skinning_field.subject_skinning_fields[subject_field_idx]
    bbox_grid_extend = diffuse_skinning_field.subject_bbox_extend[subject_field_idx]
    bbox_grid_center = diffuse_skinning_field.subject_bbox_center[subject_field_idx]

    v_cano_in_grid_coords = inv_transform_v(xc, bbox_grid_extend, bbox_grid_center)
    sw = get_w(v_cano_in_grid_coords, weights_grid) # [B, N, 24]

    return sw

def forward_skinning(cano_points, pose, trans, skinner, diffuse_skinning_field, subject_garment_id, sw=None):
    """
    points: batch x num_points x 3
    skinning_weights: batch x 24 x num_points
    trans: batch x 3

    return batch x num_points x 3
    """

    cano_points = cano_points.permute(0, 2, 1) # [B, 3, N]
    
    batch_size = pose.shape[0]
    tf = skinner.compute_smpl_skeleton(pose)

    if sw is None:
        skinning_weights = query_weights(cano_points.permute(0,2,1), diffuse_skinning_field, subject_garment_id) # [B, 24, N], use skinning weights field
    else:
        skinning_weights = sw.clone() # use NN skinning weights

    # import ipdb; ipdb.set_trace()
    # Skinning
    p_T = torch.bmm(tf.view(-1, 16, 24), skinning_weights).view(batch_size, 4, 4, -1)
    p_rest_shape_h = torch.cat([
        cano_points,
        torch.ones((batch_size, 1, cano_points.shape[2]),
                    dtype=p_T.dtype,
                    device=p_T.device),
    ], 1) # homogenous coordinate

    p_verts = (p_T * p_rest_shape_h.unsqueeze(1)).sum(2)
    p_verts = p_verts[:, :3, :] + trans[..., None]

    p_verts = p_verts.permute(0, 2, 1) # [B, N, 3]

    return p_verts 


# gradient of the forward skinning function
def gradient(xc, pose, trans, skinner, diffused_skinning_field, subject_garment_id, sw=None):
    """Get gradients df/dx

    Args:
        xc (tensor): canonical points. shape: [B, N, 3]

    Returns:
        grad (tensor): gradients. shape: [B, N, 3, 3]
    """
    xc.requires_grad_(True) # [B, N, 3]

    xd = forward_skinning(xc, pose, trans, skinner, diffused_skinning_field, subject_garment_id, sw=sw) # [B, N, 3]

    grads = []
    for i in range(xd.shape[-1]): # for each output dimension
        d_out = torch.zeros_like(xd, requires_grad=False, device=xd.device)
        d_out[:, :, i] = 1
        grad = torch.autograd.grad(
            outputs=xd,
            inputs=xc,
            grad_outputs=d_out,
            create_graph=False,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads.append(grad)

    return torch.stack(grads, dim=-2)


# root finding using Broyden's method
def broyden(g, pose, trans, x_init, J_inv_init, max_steps=1, cvg_thresh=1e-5, dvg_thresh=1, eps=1e-6):
    """Find roots of the given function g(x) = 0.
    This function is impleneted based on https://github.com/locuslab/deq.

    Tensor shape abbreviation:
        N: number of points
        D: space dimension
    Args:
        g (function): the function of which the roots are to be determined. shape: [N, D, 1]->[N, D, 1]
        x_init (tensor): initial value of the parameters. shape: [N, D, 3]
        J_inv_init (tensor): initial value of the inverse Jacobians. shape: [N, D, D]

        max_steps (int, optional): max number of iterations. Defaults to 50.
        cvg_thresh (float, optional): covergence threshold. Defaults to 1e-5.
        dvg_thresh (float, optional): divergence threshold. Defaults to 1.
        eps (float, optional): a small number added to the denominator to prevent numerical error. Defaults to 1e-6.

    Returns:
        result (tensor): root of the given function. shape: [N, D, 1]
        diff (tensor): corresponding loss. [N]
        valid_ids (tensor): identifiers of converged points. [N]
    """

    # initialization
    x = x_init.clone().detach()
    J_inv = J_inv_init.clone().detach()
    
    ids_val = torch.ones(x.shape[0]).bool()

    gx = g(x, pose, trans, mask=ids_val) # function value
    update = -J_inv.bmm(gx) # update amount

    x_opt = x.clone()
    gx_norm_opt = torch.linalg.norm(gx.squeeze(-1), dim=-1)

    delta_gx = torch.zeros_like(gx)
    delta_x = torch.zeros_like(x)

    ids_val = torch.ones_like(gx_norm_opt).bool()

    for solvestep in range(max_steps):
        # ic(solvestep)

        # update paramter values
        delta_x[ids_val] = update
        x[ids_val] += delta_x[ids_val] # update
        delta_gx[ids_val] = g(x, pose, trans, mask=ids_val) - gx[ids_val] # difference of function value
        gx[ids_val] += delta_gx[ids_val] # update saved function value

        # store values with minial loss
        gx_norm = torch.linalg.norm(gx.squeeze(-1), dim=-1)
        ids_opt = gx_norm < gx_norm_opt
        gx_norm_opt[ids_opt] = gx_norm.clone().detach()[ids_opt]
        x_opt[ids_opt] = x.clone().detach()[ids_opt] # update root if the function value is smaller than the last step

        # exclude converged and diverged points from furture iterations
        ids_val = (gx_norm_opt > cvg_thresh) & (gx_norm < dvg_thresh)
        if ids_val.sum() <= 0:
            break
        
        # Broydon's method, only compute Jacobian at the first epoch

        # compute paramter update for next iter
        vT = (delta_x[ids_val]).transpose(-1, -2).bmm(J_inv[ids_val])
        a = delta_x[ids_val] - J_inv[ids_val].bmm(delta_gx[ids_val]) #nominator
        b = vT.bmm(delta_gx[ids_val]) # denominator
        b[b >= 0] += eps
        b[b < 0] -= eps
        u = a / b
        
        # ic(ids_val)
        # ic(delta_x[ids_val].shape, J_inv[ids_val].shape, u.bmm(vT).shape)
        # ids_val_expanded = ids_val.reshape(-1,1,1).expand(-1,3,3)
        ubmmvT = u.bmm(vT)
        # ic(ids_val_expanded.shape, J_inv.shape)

        # J_inv[ids_val_expanded].add_(ubmmvT.reshape(-1).contiguous())
        # J_inv[ids_val].add_(ubmmvT)
        J_inv[ids_val] += ubmmvT

        update = -J_inv[ids_val].bmm(gx[ids_val])

    return {'result': x_opt, 'diff': gx_norm_opt, 'valid_ids': gx_norm_opt < cvg_thresh}


def search(x_posed, x_smpl_nn_cano, pose, trans, skinner, diffused_skinning_field, subject_garment_id, sw=None):
    # x_posed: B, 3, N
    # x_smpl_nn_cano: B, 3, N
    # pose: B, 24
    # trans: B, 3

    xd = x_posed.permute(0, 2, 1).contiguous() # [B, N, 3]
    xc_init = x_smpl_nn_cano.permute(0, 2, 1).contiguous() # [B, N, 3]

    J_init = gradient(xc_init, pose, trans, skinner, diffused_skinning_field, subject_garment_id, sw)
    J_inv_init = J_init.inverse()

    xc_init = xc_init.reshape(-1, 3, 1) # reformulate to [BxN, 3, 1]
    J_inv_init = J_inv_init.flatten(0, 1) # [BxN, 3, 3]

    def _func(xc_opt, pose, trans, mask=None):
        # reshape to [B,N,3] for other functions
        xc_opt = xc_opt.reshape(pose.shape[0], -1, 3)
        xd_opt = forward_skinning(xc_opt, pose, trans, skinner, diffused_skinning_field, subject_garment_id, sw)
        error = xd_opt - xd
        # reshape to [?,D,1] for boryden
        error = error.flatten(0, 1)[mask].unsqueeze(-1) # [BxN, 3, 1]
        return error

    # run broyden without grad
    with torch.no_grad():
        result = broyden(_func, pose, trans, xc_init, J_inv_init)

    xc_opt = result['result'].reshape(xd.shape[0], -1, 3) # [B, N, 3]

    if sw is None:
        skinning_weights = query_weights(xc_opt, diffused_skinning_field, subject_garment_id)
    else:
        skinning_weights = sw.clone()

    return xc_opt.permute(0, 2, 1), skinning_weights # [B, 3, N], [B, 24. N]