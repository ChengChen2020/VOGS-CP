# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.fuse_modules.adapter import Adapter, Reverter
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.fuse_modules.fusion_in_one import (
    MaxFusion,
    AttFusion,
    DiscoFusion,
    V2VNetFusion,
    V2XViTFusion,
    CoBEVT,
    Where2commFusion,
    Who2comFusion,
)
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from opencood.models.heter_single import HeterSingle

import importlib
import torchvision


class HeterLateFusion(HeterSingle):

    def __init__(self, args):
        super(HeterLateFusion, self).__init__(args)

    def forward(self, data_dict):

        agent_modality_list = data_dict["agent_modality_list"]
        modality_feature_dict, output_dict = self.forward_features(data_dict)
        for modality_name in modality_feature_dict:
            feature_list = []
            for feat_idx in range(modality_feature_dict[modality_name].shape[0]):
                
                feature = self.forward_fusion(
                    modality_feature_dict[modality_name][feat_idx:feat_idx + 1],
                    data_dict,
                    modality_name,
                    torch.tensor([1], device=data_dict["record_len"].device),
                    agent_modality_list,
                    output_dict[modality_name],
                )
                feature_list.append(feature)
            
            feature = self.forward_shrink(torch.cat(feature_list, 0), modality_name)

            self.forward_head(feature, modality_name, output_dict[modality_name])

        return output_dict

    def forward_fusion(
        self,
        feature,
        data_dict,
        modality_name,
        record_len,
        agent_modality_list,
        output_dict,
    ):
        """
        Forwards the feature through the fusion module.

        Parameters:
        - feature (Tensor): Compressed features.
        - data_dict (dict): Input data dictionary.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - record_len (int): Length of the record.
        - agent_modality_list (list): List of agent modalities.
        - output_dict (dict): Output data dictionary.

        Returns:
        - fused_feature (Tensor): Fused features.
        """

        affine_matrix = normalize_pairwise_tfm(
            data_dict["pairwise_t_matrix"],
            eval(f"self.H_{modality_name}"),
            eval(f"self.W_{modality_name}"),
            self.fake_voxel_size,
        )

        # Since the first affine_matrix (B,L,L,2,3) is always identity matrix (TO BE CONFIRMED). For not applying the
        # transformation, we only take the first element of the affine_matrix and repeat it to the length of record_len.
        affine_matrix = affine_matrix[0:1, 0:1, 0:1].repeat(len(record_len), 1, 1, 1, 1)
        fused_feature, occ_outputs = eval(f"self.pyramid_backbone_{modality_name}").forward_collab(
            feature,
            record_len,
            affine_matrix,
            [modality_name],
            self.cam_crop_info,
            # transform_idx=0,
        )

        output_dict.update({"occ_single_list": occ_outputs})

        return fused_feature
            
        