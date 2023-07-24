import torch

def np2tensor(x, device=None):
    return torch.tensor(x, device=device)


def tensor2np(x):
    return x.detach().cpu().numpy()
