import torch


def compute_smaple_on_body_mask_wo_batch(points, cut_offset, subj_loc):
    # points: [N, 3]
    # cut_offset: 0.03
    # subject_loc: [5]

    subject_unsqueeze_loc = subj_loc[None, :]
    subject_num_loc = subject_unsqueeze_loc.repeat(points.shape[0], 1) # [N, 5]

    left_hand_x = subject_num_loc[:, 0]
    right_hand_x = subject_num_loc[:, 1]
    left_foot_y = subject_num_loc[:, 2]
    right_foot_y = subject_num_loc[:, 3]
    head_y = subject_num_loc[:, 4]

    points_on_hand_indices = (points[:, 0] > left_hand_x - cut_offset) | (points[:, 0] < right_hand_x + cut_offset)
    points_on_foot_indices = (points[:, 1] < left_foot_y + cut_offset) | (points[:, 1] < right_foot_y + cut_offset)
    points_on_head_indices = (points[:, 1] > head_y - cut_offset)
    points_on_hand_foot_indices = torch.logical_or(points_on_hand_indices, points_on_foot_indices)
    points_on_head_hand_feet_indices = torch.logical_or(points_on_head_indices, points_on_hand_foot_indices)

    points_on_body = points[~points_on_head_hand_feet_indices]

    return points_on_hand_indices, points_on_foot_indices, points_on_head_indices, points_on_body


def compute_smaple_on_body_mask_w_batch(points, cut_offset, subject_loc):
    # points: [B, 3, N]
    # cut_offset: 0.03
    # subject_loc: [B, 5]

    points_permute = points.permute(0, 2, 1).contiguous() # [B, N, 3]

    subject_unsqueeze_loc = subject_loc[:, None, :]
    subject_num_loc = subject_unsqueeze_loc.repeat(1, points.shape[-1], 1) # [B, N, 5]

    left_hand_x = subject_num_loc[:, :, 0]
    right_hand_x = subject_num_loc[:, :, 1]
    left_foot_y = subject_num_loc[:, :, 2]
    right_foot_y = subject_num_loc[:, :, 3]
    head_y = subject_num_loc[:, :, 4]

    points_on_hand_indices = (points_permute[:, :, 0] > left_hand_x - cut_offset) | (points_permute[:, :, 0] < right_hand_x + cut_offset)
    points_on_foot_indices = (points_permute[:, :, 1] < left_foot_y + cut_offset) | (points_permute[:, :, 1] < right_foot_y + cut_offset)
    points_on_head_indices = (points_permute[:, :, 1] > head_y - cut_offset)

    return points_on_hand_indices, points_on_foot_indices, points_on_head_indices