import torch
from kaolin.metrics.trianglemesh import point_to_mesh_distance

def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, dim]
    """

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]

def barycentric_coordinates_of_projection(points, vertices):
    ''' https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    '''
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    
    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    #(p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    p = points

    q = v0
    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    s = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = 1e-6
    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights


def barycentric_correspondence_finding(point_set_1, smpl_face_verts_1, smpl_face_verts_2):
    # find closest triangle of point set 1 on smpl mesh 1 (face_vets_1)
    # find corresponding point set on smpl mesh 2 (face_verts_2)
    # point_set_1: [B, N, 3]
    # smpl_verts_1: [B, F, 3, 3]
    # smpl_verts_2: [B, F, 3, 3]
    # smpl_faces: [B, F, 3]

    batch_size = point_set_1.shape[0]
    smpl_face_verts_1 = smpl_face_verts_1.repeat(batch_size, 1, 1, 1)
    smpl_face_verts_2 = smpl_face_verts_1.repeat(batch_size, 1, 1, 1)
    
    # find closest face index from corr1 to ref smpl mesh
    residues, pts_ind, _ = point_to_mesh_distance(point_set_1, smpl_face_verts_1)
    # get the cloest triangles on ref smpl mesh
    closest_triangles_mesh_1 = torch.gather(smpl_face_verts_1, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    # calculate closest point in the triangle
    bary_weights = barycentric_coordinates_of_projection(point_set_1.view(-1, 3), closest_triangles_mesh_1)

    # find corresponding triangle on smpl mesh 2 and use barycentric interpolation to find weights.
    closest_triangles_mesh_2 = torch.gather(smpl_face_verts_2, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    pts_set_2 = (closest_triangles_mesh_2 * bary_weights[:, :, None]).sum(1).unsqueeze(0)

    return pts_set_2.view(batch_size, -1, 3)


def barycentric_closest_points_finding(point_set_1, smpl_face_verts_1, smpl_face_verts_2):
    # find closest triangle of point set 1 on smpl mesh 1 (face_vets_1)
    # find corresponding point set on smpl mesh 2 (face_verts_2)
    # point_set_1: [B, N, 3]
    # smpl_verts_1: [B, F, 3, 3]
    # smpl_verts_2: [B, F, 3, 3]
    # smpl_faces: [B, F, 3]

    batch_size = point_set_1.shape[0]
    if smpl_face_verts_1.shape[0] != batch_size:
        smpl_face_verts_1 = smpl_face_verts_1.repeat(batch_size, 1, 1, 1)
    if smpl_face_verts_2.shape[0] != batch_size:
        smpl_face_verts_2 = smpl_face_verts_2.repeat(batch_size, 1, 1, 1)
    
    # find closest face index from corr1 to ref smpl mesh
    residues, pts_ind, _ = point_to_mesh_distance(point_set_1, smpl_face_verts_1)
    # get the cloest triangles on ref smpl mesh
    closest_triangles_mesh_1 = torch.gather(smpl_face_verts_1, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    # calculate closest point in the triangle
    bary_weights = barycentric_coordinates_of_projection(point_set_1.view(-1, 3), closest_triangles_mesh_1)

    pts_set_1 = (closest_triangles_mesh_1 * bary_weights[:, :, None]).sum(1).unsqueeze(0)

    # find corresponding triangle on smpl mesh 2 and use barycentric interpolation to find weights.
    closest_triangles_mesh_2 = torch.gather(smpl_face_verts_2, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    pts_set_2 = (closest_triangles_mesh_2 * bary_weights[:, :, None]).sum(1).unsqueeze(0)

    return pts_set_1.view(batch_size, -1, 3), pts_set_2.view(batch_size, -1, 3)


def repeat_tensor_with_bs(tensor, batch_size):
    return tensor.repeat(batch_size, 1, 1)


def compute_smaple_on_body_mask(smpl_points, left_hand_x, right_hand_x, left_foot_y, right_foot_y, cut_offset=0.03):
    # function: compute the on body mask of points 
    # input: smpl_points [B, N, 3]
    # input: hand and feet location of the subject
    # output: mask and points of on body points [B, N]

    points_on_hand_indices = (smpl_points[:, :, 0] > left_hand_x - cut_offset) | (smpl_points[:, :, 0] < right_hand_x + cut_offset)
    points_on_foot_indices = (smpl_points[:, :, 1] < left_foot_y + cut_offset) | (smpl_points[:, :, 1] < right_foot_y + cut_offset)
    points_on_hand_foot_indices = torch.logical_or(points_on_hand_indices, points_on_foot_indices)

    points_on_body_mask = ~(points_on_hand_foot_indices)

    return points_on_body_mask