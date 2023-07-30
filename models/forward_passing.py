import torch

def query_local_feature_skinning(x_c_coarse, pose, subject_idx, local_feature_model, pose_encoder):
    # input: x_c_coarse [B, 3, N], pose [B, 72], subject_idx [B]
    # module: local_feature_model, pose_encoder
    # output: local feature together with pose feature and coarse corr, is input of neural field

    # 4th: query feature using mesh feature
    if local_feature_model is not None:
        if x_c_coarse is None:
            assert(False)
        feat, skinning = local_feature_model(x_c_coarse, subject_idx, get_sw=True) # [B, 64, N], query using explicit feature space

    # 5th: create input of shape decoder: local feature, pose feature, query location
    pose_corr = pose_encoder(pose)['pose_corr'] # [B, 24]
    feat_pose = torch.cat((feat, pose_corr.unsqueeze(1).repeat((1, x_c_coarse.shape[2], 1)).permute(0, 2, 1).contiguous()), dim=1) # 
    feat_pose_loc = torch.cat((feat_pose, x_c_coarse), dim=1) # [B, 64+24+3, N]

    return feat_pose_loc, skinning

def geometry_manifold_neural_field(feat_pose_loc, shape_decoder):
    # input: feat_pose_loc [B, 91, N]
    # module: shape_decoder
    # output: cano_geometry_offset [B, 3, N], cano_geometry_normals [B, 3, N]

    cano_geometry = shape_decoder(feat_pose_loc)
    cano_geometry_offset = cano_geometry.get('cano_cloth_displacements')
    cano_geometry_normals = cano_geometry.get('cano_cloth_normals')
    
    return cano_geometry_offset, cano_geometry_normals

def reposing_cano_points_fix_skinning(coarse_cano_points, fine_cano_points, fine_cano_normals, pose, trans, subject_idx, pre_diffused_sw_field, skinner, skinner_normal, skinning_weights=None):
    # input: fine_cano_points [B, 3, N], pose [B, 72], trans [B, 3], subject_idx [B]
    # module: pre_diffused_sw_field, skinner, skinner_normal
    # output: posed_cloth_points [B, 3, N], posed_cloth_normals [B, 3, N]

    # get skinning weights
    if skinning_weights is None:
        skinning_weights = pre_diffused_sw_field(coarse_cano_points, subject_idx)['skinning_weights']

    # 6th: foward skinning back to the pose space
    posed_cloth_points = skinner(fine_cano_points, pose, skinning_weights, trans)['posed_cloth_points']

    if fine_cano_normals is not None:
        posed_cloth_normals = skinner_normal(fine_cano_normals, pose, skinning_weights)['posed_cloth_normals']

        return posed_cloth_points, posed_cloth_normals
    
    return posed_cloth_points, None
