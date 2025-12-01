# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License



import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from opencood.models.sub_modules.lss_submodule import (
    Up,
    CamEncode,
    BevEncode,
    CamEncode_Resnet101,
    CamEncode_Resnet50,
    CamEncode_Resnet34,
    CamEncode_EfficientNet_B1,
    CamEncode_EfficientNet_B2,
)
from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, depth_discretization
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.point_transformer_v3 import PointTransformerV3
from opencood.models.pixor import Bottleneck as PIXORBottlenect, BackBone as PIXORBackBone
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.fax_modules import FAXModule

from opencood.models.voxel_net import CML
from opencood.utils.common_utils import torch_tensor_to_numpy
from torch.autograd import Variable
from torch_scatter import scatter_mean, scatter_max
import math


class PointTransformer(nn.Module):
    def __init__(self, args):
        super(PointTransformer, self).__init__()
        self.transformer = PointTransformerV3(
            stride=(2, ),
            enc_depths=(1, 1),
            enc_channels=(16, 32),
            enc_num_head=(2, 4),
            enc_patch_size=(128, 128),
            dec_depths=(1,),
            dec_channels=(64, ),
            dec_num_head=(2, ),
            dec_patch_size=(128, ),
            mlp_ratio=8,
            in_channels=1,
            enable_flash=False, 
            cls_mode=False)
        
    @staticmethod
    def point_to_bev_mean(point):
        # Extract information
        batch_indices = point['batch'].long()  # Ensure correct data type
        coords = point['coord'].long()
        feats = point['feat']
        
        # Determine dimensions
        batch_size = int(batch_indices.max().item()) + 1
        num_channels = feats.shape[1]
        lidar_range = point['lidar_range']
        
        H, W = point['grid_size'][1].item(), point['grid_size'][0].item()
        
        # Valid coordinate mask
        valid_mask = (
            (coords[:, 0] >= 0) & (coords[:, 0] < W) &
            (coords[:, 1] >= 0) & (coords[:, 1] < H) &
            (coords[:, 2] >= 0) & (coords[:, 2] < point['grid_size'][2].item()) &
            (batch_indices >= 0) & (batch_indices < batch_size)
        )
        
        # Apply mask
        coords = coords[valid_mask]
        feats = feats[valid_mask]
        batch_indices = batch_indices[valid_mask]
        
        # Compute indices
        x_indices = coords[:, 0]
        y_indices = coords[:, 1]
        batch_offset = batch_indices * H * W
        linear_indices = batch_offset + y_indices * W + x_indices
        
        total_elements = batch_size * H * W
        
        # Check indices
        assert linear_indices.min() >= 0 and linear_indices.max() < total_elements, (
            f"Indices out of bounds: min {linear_indices.min()}, max {linear_indices.max()}, "
            f"allowed range [0, {total_elements - 1}]"
        )
        
        # Aggregate features
        bev_feature_map = scatter_mean(feats, linear_indices, dim=0, dim_size=total_elements)
        
        # Reshape
        bev_feature_map = bev_feature_map.view(batch_size, H, W, num_channels)
        bev_feature_map = bev_feature_map.permute(0, 3, 1, 2)
        
        return bev_feature_map
    
    # def point_to_bev_max(point):
    #     # Extract necessary information from the point dictionary
    #     batch_indices = point['batch']  # Shape: [N]
    #     coords = point['coord'].long()  # Shape: [N, 3], assuming integer grid indices
    #     feats = point['feat']           # Shape: [N, C]
    #     grid_size = point['grid_size']  # Tensor of shape [3], e.g., [256, 256, 1]
        
    #     # Determine the batch size and feature dimensions
    #     batch_size = batch_indices.max().item() + 1
    #     num_channels = feats.shape[1]
    #     H, W = grid_size[1].item(), grid_size[0].item()  # Height and Width of the BEV map

    #     # Compute linear indices for scatter operation
    #     x_indices = coords[:, 0]
    #     y_indices = coords[:, 1]
    #     batch_offset = batch_indices * H * W
    #     linear_indices = batch_offset + y_indices * W + x_indices  # Shape: [N]

    #     # Total number of elements in the BEV feature map
    #     total_elements = batch_size * H * W

    #     # Aggregate features into the BEV feature map using max
    #     # Note: scatter_max returns both the max values and the argmax indices
    #     bev_feature_map, _ = scatter_max(feats, linear_indices, dim=0, dim_size=total_elements)

    #     # Handle cells with no points (scatter_max fills with the minimum tensor value)
    #     # Replace them with zeros or an appropriate value
    #     bev_feature_map[bev_feature_map == feats.min()] = 0

    #     # Reshape and permute to get the final BEV feature map of shape [batch_size, num_channels, H, W]
    #     bev_feature_map = bev_feature_map.view(batch_size, H, W, num_channels)
    #     bev_feature_map = bev_feature_map.permute(0, 3, 1, 2)  # Shape: [batch_size, num_channels, H, W]


    def forward(self, data_dict, modality_name, multi_sensor=False):
        point = self.transformer(data_dict[f'inputs_{modality_name}'])
        # convert points to BEV feature map
        # lidar_range = point['lidar_range']
        # voxel_size = point['voxel_size']
        # import pdb; pdb.set_trace()
        
        feature = self.point_to_bev_mean(point)
        
        return feature


class FAX(nn.Module):

    def __init__(self, args):
        super(FAX, self).__init__()
        self.encoder = ResnetEncoder(args["encoder"])

        # channel 3 -> 4 for depth and reuse the old pretrained channels
        if "depth" in args["input_source"]:
            original_conv1 = self.encoder.encoder.conv1

            new_conv1 = nn.Conv2d(
                in_channels=4,
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias is not None,
            )
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight
                new_conv1.weight[:, 3:, :, :] = original_conv1.weight[:, :1, :, :]
            self.encoder.encoder.conv1 = new_conv1
        # cvm params
        fax_params = args["fax"]
        fax_params["backbone_output_shape"] = self.encoder.output_shapes
        self.fax = FAXModule(fax_params)

    def forward(self, data_dict, modality_name, multi_sensor=False):
        if multi_sensor:
            input_data = data_dict[f"inputs_{modality_name}"]["camera"]
        else:
            input_data = data_dict[f"inputs_{modality_name}"]
        image_inputs_dict = input_data
        x, rots, trans, intrins, post_rots, post_trans = (
            image_inputs_dict["imgs"],
            image_inputs_dict["rots"],
            image_inputs_dict["trans"],
            image_inputs_dict["intrins"],
            image_inputs_dict["post_rots"],
            image_inputs_dict["post_trans"],
        )
        bl, m, _, _, _ = x.shape
        x = self.encoder(x)
        input_data.update({"features": x})
        x = self.fax(input_data)

        return x


class PIXOR(nn.Module):

    def __init__(self, args):
        super(PIXOR, self).__init__()
        geom = args["geometry_param"]
        use_bn = args["use_bn"]
        self.backbone = PIXORBackBone(PIXORBottlenect, [3, 6, 6, 3], geom, use_bn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, data_dict, modality_name, multi_sensor=False):
        if multi_sensor:
            input_data = data_dict[f"inputs_{modality_name}"]["lidar"]["bev_input"]
        else:
            input_data = data_dict[f"inputs_{modality_name}"]["bev_input"]
        feature = self.backbone(input_data)

        return feature


class VoxelNet(nn.Module):

    def __init__(self, args):
        super(VoxelNet, self).__init__()
        self.svfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )

        self.cml = CML()

        grid_size = (np.array(args["lidar_range"][3:6]) - np.array(args["lidar_range"][0:3])) / np.array(
            args["voxel_size"]
        )
        grid_size = np.round(grid_size).astype(np.int64)
        self.H = grid_size[1]
        self.W = grid_size[0]
        self.D = grid_size[2]

    def voxel_indexing(self, sparse_features, coords, N):
        dim = sparse_features.shape[-1]

        dense_feature = Variable(torch.zeros(dim, N, self.D, self.H, self.W).cuda())

        dense_feature[:, coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = sparse_features.transpose(0, 1)

        return dense_feature.transpose(0, 1)

    def forward(self, data_dict, modality_name, multi_sensor=False):
        if multi_sensor:
            input_data = data_dict[f"inputs_{modality_name}"]["lidar"]
        else:
            input_data = data_dict[f"inputs_{modality_name}"]

        voxel_features = input_data["voxel_features"]
        voxel_coords = input_data["voxel_coords"]
        voxel_num_points = input_data["voxel_num_points"]

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
        }

        # feature learning network
        vwfs = self.svfe(batch_dict)["pillar_features"]
        voxel_coords = torch_tensor_to_numpy(voxel_coords)

        # for N in data_dict['record_len']:
        N = data_dict["record_len"].sum().item()
        vwfs = self.voxel_indexing(vwfs, voxel_coords, N=N)

        # TODO: 目前voxel net的版本可能不是对的，但我们暂且忽略，之后需要解决, 原因是目前用了D=1，但voxelnet的cml层默认D≥3
        # convolutional middle network
        # vwfs = self.cml(vwfs)

        vwfs = vwfs.view(N, -1, self.H, self.W)

        # # region proposal network

        # # merge the depth and feature dim into one, output probability score
        # # map and regression map
        # psm, rm = self.rpn(vwfs.view(self.N, -1, self.H, self.W))

        # output_dict = {'psm': psm,
        #                'rm': rm}

        # return output_dict
        return vwfs


class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()
        grid_size = (np.array(args["lidar_range"][3:6]) - np.array(args["lidar_range"][0:3])) / np.array(
            args["voxel_size"]
        )
        grid_size = np.round(grid_size).astype(np.int64)
        args["point_pillar_scatter"]["grid_size"] = grid_size

        # PIllar VFE
        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])

    def forward(self, data_dict, modality_name, multi_sensor=False):

        if multi_sensor:
            input_data = data_dict[f"inputs_{modality_name}"]["lidar"]
        else:
            input_data = data_dict[f"inputs_{modality_name}"]

        voxel_features = input_data["voxel_features"]
        voxel_coords = input_data["voxel_coords"]
        voxel_num_points = input_data["voxel_num_points"]

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
        }

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        lidar_feature_2d = batch_dict["spatial_features"]  # H0, W0

        return lidar_feature_2d


class SECOND(nn.Module):
    def __init__(self, args):
        super(SECOND, self).__init__()
        lidar_range = np.array(args["lidar_range"])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) / np.array(args["voxel_size"])).astype(np.int64)
        self.vfe = MeanVFE(args["mean_vfe"], args["mean_vfe"]["num_point_features"])
        self.spconv_block = VoxelBackBone8x(
            args["spconv"], input_channels=args["spconv"]["num_features_in"], grid_size=grid_size
        )
        self.map_to_bev = HeightCompression(args["map2bev"])

    def forward(self, data_dict, modality_name, multi_sensor=False):
        if multi_sensor:
            input_data = data_dict[f"inputs_{modality_name}"]["lidar"]
        else:
            input_data = data_dict[f"inputs_{modality_name}"]

        voxel_features = input_data["voxel_features"]
        voxel_coords = input_data["voxel_coords"]
        voxel_num_points = input_data["voxel_num_points"]
        batch_size = voxel_coords[:, 0].max() + 1

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "batch_size": batch_size,
        }

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        return batch_dict["spatial_features"]


class LiftSplatShoot(nn.Module):
    def __init__(self, args):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = args["grid_conf"]  # 网格配置参数
        self.data_aug_conf = args["data_aug_conf"]  # 数据增强配置参数
        dx, bx, nx = gen_dx_bx(
            self.grid_conf["xbound"],
            self.grid_conf["ybound"],
            self.grid_conf["zbound"],
        )  # 划分网格

        self.dx = dx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [0.4,0.4,20]
        self.bx = bx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [-49.8,-49.8,0]
        self.nx = nx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [250,250,1]
        self.depth_supervision = args["depth_supervision"]
        self.downsample = args["img_downsample"]  # 下采样倍数
        self.camC = args["img_features"]  # 图像特征维度
        self.frustum = (
            self.create_frustum().clone().detach().requires_grad_(False).to(torch.device("cuda"))
        )  # frustum: DxfHxfWx3(41x8x16x3)
        self.use_quickcumsum = True
        self.D, _, _, _ = self.frustum.shape  # D: 41
        self.camera_encoder_type = args["camera_encoder"]
        if self.camera_encoder_type == "EfficientNet":
            self.camencode = CamEncode(
                self.D,
                self.camC,
                self.downsample,
                self.grid_conf["ddiscr"],
                self.grid_conf["mode"],
                args["use_depth_gt"],
                args["depth_supervision"],
            )
        elif self.camera_encoder_type == "EfficientNetB1":
            self.camencode = CamEncode_EfficientNet_B1(
                self.D,
                self.camC,
                self.downsample,
                self.grid_conf["ddiscr"],
                self.grid_conf["mode"],
                args["use_depth_gt"],
                args["depth_supervision"],
            )
        elif self.camera_encoder_type == "EfficientNetB2":
            self.camencode = CamEncode_EfficientNet_B2(
                self.D,
                self.camC,
                self.downsample,
                self.grid_conf["ddiscr"],
                self.grid_conf["mode"],
                args["use_depth_gt"],
                args["depth_supervision"],
            )
        elif self.camera_encoder_type == "Resnet101":
            self.camencode = CamEncode_Resnet101(
                self.D,
                self.camC,
                self.downsample,
                self.grid_conf["ddiscr"],
                self.grid_conf["mode"],
                args["use_depth_gt"],
                args["depth_supervision"],
            )
        elif self.camera_encoder_type == "Resnet50":
            self.camencode = CamEncode_Resnet50(
                self.D,
                self.camC,
                self.downsample,
                self.grid_conf["ddiscr"],
                self.grid_conf["mode"],
                args["use_depth_gt"],
                args["depth_supervision"],
            )
        elif self.camera_encoder_type == "Resnet34":
            self.camencode = CamEncode_Resnet34(
                self.D,
                self.camC,
                self.downsample,
                self.grid_conf["ddiscr"],
                self.grid_conf["mode"],
                args["use_depth_gt"],
                args["depth_supervision"],
            )

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf["final_dim"]  # 原始图片大小  ogfH:128  ogfW:288
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # 下采样16倍后图像大小  fH: 12  fW: 22
        # ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)  # 在深度方向上划分网格 ds: DxfHxfW(41x12x22)
        ds = (
            torch.tensor(depth_discretization(*self.grid_conf["ddiscr"], self.grid_conf["mode"]), dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )

        D, _, _ = ds.shape  # D: 41 表示深度方向上网格的数量
        xs = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        )  # 在0到288上划分18个格子 xs: DxfHxfW(41x12x22)
        ys = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        )  # 在0到127上划分8个格子 ys: DxfHxfW(41x12x22)

        # D x H x W x 3
        frustum = torch.stack(
            (xs, ys, ds), -1
        )  # 堆积起来形成网格坐标, frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标   frustum: DxfHxfWx3
        return frustum

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # B:4(batchsize)    N: 4(相机数目)

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (
                points[:, :, :, :, :, :2]
                * points[:, :, :, :, :, 2:3],  # points[:, :, :, :, :, 2:3] ranges from [4, 45) meters
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )  # 将像素坐标(u,v,d)变成齐次坐标(du,dv,d)
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)  # 将像素坐标d[u,v,1]^T转换到车体坐标系下的[x,y,z]

        return points  # B x N x D x H x W x 3 (4 x 4 x 41 x 16 x 22 x 3)

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C"""
        B, N, C, imH, imW = x.shape  # B: 4  N: 4  C: 3  imH: 256  imW: 352

        x = x.view(B * N, C, imH, imW)  # B和N两个维度合起来  x: 16 x 4 x 256 x 352
        depth_items, x = self.camencode(x)  # 进行图像编码  x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22)
        x = x.view(
            B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample
        )  # 将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)
        x = x.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)

        return x, depth_items

    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B * N * D * H * W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices

        geom_feats = (
            (geom_feats - (self.bx - self.dx / 2.0)) / self.dx
        ).long()  # 将[-48,48] [-10 10]的范围平移到 [0, 240), [0, 1) 计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3)  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3
        batch_ix = torch.cat(
            [torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)]
        )  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4, geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~240  y: 0~240  z: 0
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)
            + geom_feats[:, 1] * (self.nx[2] * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(
                x, geom_feats, ranks
            )  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)  # final: 4 x 64 x Z x Y x X
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = (
            x  # 将x按照栅格坐标放到final中
        )

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维

        return final  # final: 4 x 64 x 240 x 240  # B, C, H, W

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(
            rots, trans, intrins, post_rots, post_trans
        )  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3 (4 x N x 42 x 16 x 22 x 3)
        x_img, depth_items = self.get_cam_feats(
            x
        )  # 提取图像特征并预测深度编码 x: B x N x D x fH x fW x C(4 x N x 42 x 16 x 22 x 64)
        x = self.voxel_pooling(geom, x_img)  # x: 4 x 64 x 240 x 240

        return x, depth_items

    def forward(self, data_dict, modality_name, multi_sensor=False):
        # x: [4,4,3,256, 352]
        # rots: [4,4,3,3]
        # trans: [4,4,3]
        # intrins: [4,4,3,3]
        # post_rots: [4,4,3,3]
        # post_trans: [4,4,3]
        if multi_sensor:
            input_data = data_dict[f"inputs_{modality_name}"]["camera"]
        else:
            input_data = data_dict[f"inputs_{modality_name}"]
        image_inputs_dict = input_data
        x, rots, trans, intrins, post_rots, post_trans = (
            image_inputs_dict["imgs"],
            image_inputs_dict["rots"],
            image_inputs_dict["trans"],
            image_inputs_dict["intrins"],
            image_inputs_dict["post_rots"],
            image_inputs_dict["post_trans"],
        )
        x, depth_items = self.get_voxels(
            x, rots, trans, intrins, post_rots, post_trans
        )  # 将图像转换到BEV下，x: B x C x 240 x 240 (4 x 64 x 240 x 240)

        if self.depth_supervision:
            self.depth_items = depth_items

        return x


class LiftSplatShootVoxel(LiftSplatShoot):
    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B * N * D * H * W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices

        geom_feats = (
            (geom_feats - (self.bx - self.dx / 2.0)) / self.dx
        ).long()  # 将[-48,48] [-10 10]的范围平移到 [0, 240), [0, 1) 计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3)  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3
        batch_ix = torch.cat(
            [torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)]
        )  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4, geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~240  y: 0~240  z: 0
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)
            + geom_feats[:, 1] * (self.nx[2] * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(
                x, geom_feats, ranks
            )  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)  # final: 4 x 64 x Z x Y x X
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = (
            x  # 将x按照栅格坐标放到final中
        )

        # collapse Z
        # final = torch.max(final.unbind(dim=2), 1)[0]  # 消除掉z维
        final = torch.max(final, 2)[0]  # 消除掉z维
        return final  # final: 4 x 64 x 240 x 240  # B, C, H, W
