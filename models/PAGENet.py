from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import sys
sys.path.append('code/PAGE')
import lib.utils.etw_pytorch_utils as pt_utils
from collections import namedtuple
from lib.pspnet import PSPNet, Modified_PSPNet
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
import torch.nn.functional as F
import time


psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

modified_psp_models = {
    'resnet18': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda:Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda:Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        # self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class Pointnet2MSG(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(
        self, input_channels=6, use_xyz=True
    ):
        super(Pointnet2MSG, self).__init__()

        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=2048,
                radii=[0.0175, 0.025],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.025, 0.05],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=use_xyz,
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + input_channels, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        _, N, _ = pointcloud.size()

        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]


class DenseFusion(nn.Module):
    def __init__(self, num_points):
        super(DenseFusion, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(128, 256, 1)
        self.conv2_cld = torch.nn.Conv1d(128, 256, 1)

        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, rgb_emb, cld_emb):
        bs, _, n_pts = cld_emb.size()
        feat_1 = torch.cat((rgb_emb, cld_emb), dim=1)
        rgb = F.relu(self.conv2_rgb(rgb_emb))
        cld = F.relu(self.conv2_cld(cld_emb))

        feat_2 = torch.cat((rgb, cld), dim=1)

        rgbd = F.relu(self.conv3(feat_1))
        rgbd = F.relu(self.conv4(rgbd))

        ap_x = self.ap1(rgbd)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([feat_1, feat_2, ap_x], 1) # 256 + 512 + 1024 = 1792


class DenseFusion_1(nn.Module):
    def __init__(self, num_points):
        super(DenseFusion, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(128, 256, 1)
        self.conv2_cld = torch.nn.Conv1d(128, 256, 1)

        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, rgb_emb, cld_emb):
        bs, _, n_pts = cld_emb.size()
        feat_1 = torch.cat((rgb_emb, cld_emb), dim=1)
        rgb = F.relu(self.conv2_rgb(rgb_emb))
        cld = F.relu(self.conv2_cld(cld_emb))

        feat_2 = torch.cat((rgb, cld), dim=1)

        rgbd = F.relu(self.conv3(feat_1))
        rgbd = F.relu(self.conv4(rgbd))

        return torch.cat([feat_1, feat_2, rgbd], 1) # 256 + 512 + 1024 = 1792

class IA_Layer(nn.Module):
    def __init__(self, channels):
        super(IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                    nn.BatchNorm1d(self.pc),
                                    nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)


    def forward(self, img_feas, point_feas):
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
        # print(img_feas)
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        # att = F.sigmoid(self.fc3(F.tanh(ri + rp))) #BNx1
        att = F.gumbel_softmax(self.fc3(F.tanh(ri + rp))) #BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1) #B1N
        # print(img_feas.size(), att.size())

        img_feas_new = self.conv1(img_feas)
        out = img_feas_new * att

        return out

class GC_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(GC_Fusion_Conv, self).__init__()

        self.IA_Layer = IA_Layer(channels = [inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)

        img_features =  self.IA_Layer(img_features, point_features)
        #print("img_features:", img_features.shape)

        #fusion_features = img_features + point_features
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features

class PAGE_Estimator(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        pcld_input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        pcld_use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
        num_kps: int = 8
            Number of keypoints to predict
        num_points: int 8192
            Number of sampled points from point clouds.
    """

    def __init__(
        self, num_classes, pcld_input_channels=6, pcld_use_xyz=True,
        num_kps=3, num_points=8192
    ):
        super(PAGE_Estimator, self).__init__()

        self.num_kps = num_kps
        self.num_classes = num_classes
        self.cnn = ModifiedResnet()
        self.pointnet2 = Pointnet2MSG(
            input_channels=pcld_input_channels,
            use_xyz=pcld_use_xyz
        )
        # self.rgbd_feat = DenseFusion(num_points)
        self.rgbd_feat = GC_Fusion_Conv(128, 128, 128)

        self.SEG_layer = (
            pt_utils.Seq(128)
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(64, bn=True, activation=nn.ReLU())
            .conv1d(num_classes, activation=None)
        )
        self.Kpradiul_3d_layer = (
            pt_utils.Seq(128)
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(num_kps, activation=None)
        )
        # Joint parameters
        self.joint_feature_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            # nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm1d(128),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
        )
        # Joint UNitVec, heatmap, joint_cls
        self.axis_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)
        self.unitvec_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)
        self.heatmap_layer = nn.Conv1d(128, 1, kernel_size=1, padding=0)
        self.joint_cls_layer = nn.Conv1d(128, num_classes, kernel_size=1, padding=0)


    def forward(self, pointcloud, rgb, choose):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
            rgb: Variable(torch.cuda.FloatTensor)
                (B, C, H, W) tensor
            choose: Variable(torch.cuda.LongTensor)
                (B, 1, N) tensor
                indexs of choosen points(pixels).
        """
        out_rgb, rgb_seg = self.cnn(rgb)  #input: batch_size, Channel, H, W

        bs, di, _, _ = out_rgb.size()

        rgb_emb = out_rgb.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)   #choose: B, 1, N
        rgb_emb = torch.gather(rgb_emb, 2, choose).contiguous()

        _, N, _ = pointcloud.size()
        pcld_emb = self.pointnet2(pointcloud)   #input: batch_size, num_points, 9

        rgbd_feature = self.rgbd_feat(rgb_emb, pcld_emb)

        pred_rgbd_seg = self.SEG_layer(rgbd_feature).transpose(1, 2).contiguous()
        pred_kp_radial_3d = self.Kpradiul_3d_layer(rgbd_feature).transpose(1, 2).contiguous()
        pred_kp_radial_3d = F.softplus(pred_kp_radial_3d)

        #joint param
        joint_features = self.joint_feature_layer(rgbd_feature)
        pred_axis = self.axis_layer(joint_features).transpose(1, 2)
        pred_unitvec = self.unitvec_layer(joint_features).transpose(1, 2)
        pred_heatmap = self.heatmap_layer(joint_features).transpose(1, 2)
        pred_joint_cls = self.joint_cls_layer(joint_features).transpose(1, 2)

        pred_heatmap = F.sigmoid(pred_heatmap)
        pred_unitvec = F.tanh(pred_unitvec)
        pred_axis = F.tanh(pred_axis)
        pred_joint_cls = F.softmax(pred_joint_cls, dim=2)

        return pred_rgbd_seg, pred_kp_radial_3d, pred_heatmap, pred_unitvec, pred_axis, pred_joint_cls
