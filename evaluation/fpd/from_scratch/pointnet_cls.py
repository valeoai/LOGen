import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
import numpy as np

class Pointnet(nn.Module):
    def __init__(self, k=40, input_channels=3, return_act=False):
        super(Pointnet, self).__init__()
        channel = input_channels
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        # self.num_features=256
        self.num_features=1803
        self.return_act = return_act
    def forward(self, x):  # return_act default False (for FPD), True for Pytorchs KPD,FPD
        x1, trans, trans_feat = self.feat(x)
        x2 = F.relu(self.bn1(self.fc1(x1)))
        x3 = F.relu(self.bn2(self.dropout(self.fc2(x2))))
        x4 = self.fc3(x3)
        x = F.log_softmax(x4, dim=1)
        actv = torch.cat((x1, x2, x3, x4), dim=1)
        if self.return_act:
            return actv
        else:
            return x, trans_feat, actv


class SPVCNN(nn.Module):
    def __init__(self,gen_model, input_channels):
        super(SPVCNN, self).__init__()
        self.main_path = "/home/nsamet/scania/nsamet"
        self.channel = input_channels
        self.gen_feats = np.load(f"{self.main_path}/{gen_model}_replacement_val_dataset_{self.channel}CH_spvcnn_instance_feats.npz", allow_pickle=True)['feat_data'].item()
        self.org_feats = np.load(f"{self.main_path}/org_val_{self.channel}CH_spvcnn_instance_feats.npz", allow_pickle=True)['feat_data'].item()
        self.num_features=96
    def forward(self, x):

        scn_name, data_type = x
        scn_id = scn_name[0].split("/")[-1]
        cls_id = scn_name[0].split("/")[-2]
        if data_type == "gen":
            feat = self.gen_feats[cls_id][scn_id]
        if data_type == "real":
            feat = self.org_feats[cls_id][scn_id]
        feat = torch.from_numpy(np.expand_dims(feat, axis=0)).cuda()
        # feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return feat



class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
