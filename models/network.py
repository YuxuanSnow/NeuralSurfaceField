import torch
from torch import nn

class PoseEncoder(nn.Module):
    def __init__(self, in_features=72, out_features=1):
        super(PoseEncoder, self).__init__()

        encoder = [
            module for module in [
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, out_features)
            ] if module is not None
        ]
        self.encoder = nn.Sequential(*encoder)

    def forward(self, data):
        feat = self.encoder(data)
        return {'pose_corr': feat}
    

class NeuralSurfaceDeformationField(nn.Module):
    def __init__(self, feat_in=3+256+24, hidden_sz=256, actv_fn='softplus'):
        self.hsize = hidden_sz
        super(NeuralSurfaceDeformationField, self).__init__()
        self.conv1 = nn.Conv1d(feat_in, self.hsize, 1)
        self.conv2 = nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv3 = nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv4 = nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv5 = nn.Conv1d(self.hsize+feat_in, self.hsize, 1)
        self.conv6 = nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7 = nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8 = nn.Conv1d(self.hsize, 3, 1)

        self.conv6N = nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7N = nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8N = nn.Conv1d(self.hsize, 3, 1)


        self.bn1 = nn.BatchNorm1d(self.hsize)
        self.bn2 = nn.BatchNorm1d(self.hsize)
        self.bn3 = nn.BatchNorm1d(self.hsize)
        self.bn4 = nn.BatchNorm1d(self.hsize)

        self.bn5 = nn.BatchNorm1d(self.hsize)
        self.bn6 = nn.BatchNorm1d(self.hsize)
        self.bn7 = nn.BatchNorm1d(self.hsize)

        self.bn6N = nn.BatchNorm1d(self.hsize)
        self.bn7N = nn.BatchNorm1d(self.hsize)

        self.actv_fn = nn.ReLU() if actv_fn=='relu' else nn.Softplus()

    def forward(self, x):
        # x: points on the fusion shape surface 
        
        x1 = self.actv_fn(self.bn1(self.conv1(x)))
        x2 = self.actv_fn(self.bn2(self.conv2(x1)))
        x3 = self.actv_fn(self.bn3(self.conv3(x2)))
        x4 = self.actv_fn(self.bn4(self.conv4(x3)))
        x5 = self.actv_fn(self.bn5(self.conv5(torch.cat([x,x4],dim=1)))) # skip connection

        # position pred
        x6 = self.actv_fn(self.bn6(self.conv6(x5)))
        x7 = self.actv_fn(self.bn7(self.conv7(x6)))
        x8 = self.conv8(x7)

        # normals pred
        xN6 = self.actv_fn(self.bn6N(self.conv6N(x5)))
        xN7 = self.actv_fn(self.bn7N(self.conv7N(xN6)))
        xN8 = self.conv8N(xN7)

        # x8: displacement from fusion shape to clothed body in canonical space
        # xN8: normal of the clothed body in canonical space

        return {'cano_cloth_displacements': x8, 'cano_cloth_normals': xN8} 
