import torch
from typing import Union
from pytorch3d.ops import knn_points, knn_gather

# s2m and m2s chamfer distance
def chamfer_distance_s2m_m2s(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
):
    # input: prediction x, [B, N, 3]
    # input: target shape y, [B, N, 3]
    # output: x_y, per coordinate distance
    # output: y_x, per coordinate distance
    # output: s2m: squared chamfer distance of x_y
    # output: m2s: squared chamfer distance of y_x
    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    batchsize_source, lengths_source, dim_source = x.shape
    lengths_source = (
        torch.ones(batchsize_source, dtype=torch.long, device=x.device)
        * lengths_source
    )

    batchsize_target, lengths_target, dim_target = y.shape
    lengths_target = (
        torch.ones(batchsize_target, dtype=torch.long, device=y.device)
        * lengths_target
    )

    y_nn = knn_points(y, x, lengths1=lengths_target, lengths2=lengths_source, K=1)
    x_nn = knn_points(x, y, lengths1=lengths_source, lengths2=lengths_target, K=1)

    y_coords_near = knn_gather(x, y_nn.idx, lengths_source)[..., 0, :]
    x_coords_near = knn_gather(y, x_nn.idx, lengths_target)[..., 0, :]
    x_y = y - y_coords_near  # (N, P2)
    y_x = x - x_coords_near  # (N, P2)

    s2m = torch.square(torch.norm(x_y, dim=2)).mean()
    m2s = torch.square(torch.norm(y_x, dim=2)).mean()

    if return_normals:
        y_normals_near = knn_gather(x_normals, y_nn.idx, lengths_source)[..., 0, :]
        x_normals_near = knn_gather(y_normals, x_nn.idx, lengths_source)[..., 0, :]
        x_y_norm = y_normals - y_normals_near
        y_x_norm = x_normals - x_normals_near

        s2m_normal = abs(x_y_norm).sum(-1).mean()
        m2s_normal = abs(y_x_norm).sum(-1).mean()

        return s2m, m2s, s2m_normal, m2s_normal
    else:
        x_y_norm = None
        y_x_norm = None

        return s2m, m2s