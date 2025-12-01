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
from opencood.models.heter_base import HeterBase

import importlib
import torchvision


class HeterSingle(HeterBase):

    def __init__(self, args):

        super(HeterSingle, self).__init__(args)

        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            self.build_fusion(modality_name, model_setting)
            self.build_shrink_header(modality_name, model_setting)
            self.build_head(modality_name, model_setting)

        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)

    def forward(self, data_dict):
        record_len = data_dict["record_len"]
        agent_modality_list = data_dict["agent_modality_list"]
        modality_feature_dict, output_dict = self.forward_features(data_dict)
        for modality_name in self.modality_name_list:

            feature = self.forward_fusion(
                modality_feature_dict[modality_name][torch.cumsum(record_len, dim=0) - record_len],
                data_dict,
                modality_name,
                torch.ones(len(record_len), dtype=torch.int64),
                agent_modality_list,
                output_dict[modality_name],
            )
            
            feature = self.forward_shrink(feature, modality_name)

            self.forward_head(feature, modality_name, output_dict[modality_name])

        return output_dict

    def build_shrink_header(self, modality_name, model_setting):
        """
        Builds the shrink header for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up a downsample convolutional layer if the shrink header is specified.
        """

        setattr(self, f"shrink_flag_{modality_name}", False)
        if "shrink_header" in model_setting:
            setattr(self, f"shrink_flag_{modality_name}", True)
            setattr(
                self,
                f"shrink_conv_{modality_name}",
                DownsampleConv(model_setting["shrink_header"]),
            )

    def build_head(self, modality_name, model_setting):
        """
        Builds the head for a given modality.

        Parametersz:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the head network for various head methods such as object detection, segmentation, etc.
        """

        # By default, point pillar pyramid object detection head
        self.head_method = model_setting.get("head_method", "point_pillar_pyramid_object_detection_head")
        self.downsample_rate = model_setting.get("downsample_rate", 1)

        if self.head_method == "point_pillar_pyramid_object_detection_head":

            setattr(
                self,
                f"cls_head_{modality_name}",
                nn.Conv2d(
                    model_setting["in_head"],
                    model_setting["anchor_number"],
                    kernel_size=1,
                ),
            )
            setattr(
                self,
                f"reg_head_{modality_name}",
                nn.Conv2d(
                    model_setting["in_head"],
                    7 * model_setting["anchor_number"],
                    kernel_size=1,
                ),
            )
            if model_setting.get("dir_args", None):
                setattr(
                    self,
                    f"dir_head_{modality_name}",
                    nn.Conv2d(
                        model_setting["in_head"],
                        model_setting["dir_args"]["num_bins"] * model_setting["anchor_number"],
                        kernel_size=1,
                    ),
                )

        elif self.head_method == "point_pillar_object_detection_head":
            setattr(
                self,
                f"cls_head_{modality_name}",
                nn.Conv2d(model_setting["in_head"], 1, kernel_size=1),
            )
            setattr(
                self,
                f"reg_head_{modality_name}",
                nn.Conv2d(model_setting["in_head"], 7, kernel_size=1),
            )
            if model_setting.get("dir_args", None):
                setattr(
                    self,
                    f"dir_head_{modality_name}",
                    nn.Conv2d(
                        model_setting["in_head"],
                        model_setting["dir_args"]["num_bins"],
                        kernel_size=1,
                    ),
                )

        elif self.head_method == "bev_seg_head":
            self.head = nn.Sequential(
                NaiveDecoder(model_setting["decoder_args"]),
                BevSegHead(
                    model_setting["target"],
                    model_setting["seg_head_dim"],
                    model_setting["output_class_dynamic"],
                    model_setting["output_class_static"],
                ),
            )

        elif self.head_method == "pixor_head":

            setattr(
                self,
                f"cls_head_{modality_name}",
                nn.Conv2d(model_setting["in_head"], 1, kernel_size=1),
            )
            setattr(
                self,
                f"reg_head_{modality_name}",
                nn.Conv2d(model_setting["in_head"], 6, kernel_size=1),
            )

        else:
            raise NotImplementedError(f"Head method {self.head_method} not implemented.")

    def build_fusion(self, modality_name, model_setting):
        """
        Builds the fusion module for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the fusion method based on the specified fusion method in the model settings.
        """

        """
        Fusion, by default multiscale fusion:
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """

        if model_setting["fusion_method"] == "max":
            setattr(self, f"pyramid_backbone_{modality_name}", MaxFusion())
        elif model_setting["fusion_method"] == "att":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                AttFusion(model_setting["fusion_backbone"]),
            )
        elif model_setting["fusion_method"] == "disconet":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                DiscoFusion(model_setting["fusion_backbone"]),
            )
        elif model_setting["fusion_method"] == "v2vnet":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                V2VNetFusion(model_setting["fusion_backbone"]),
            )
        elif model_setting["fusion_method"] == "v2xvit":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                V2XViTFusion(model_setting["fusion_backbone"]),
            )
        elif model_setting["fusion_method"] == "cobevt":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                CoBEVT(model_setting["fusion_backbone"]),
            )
        elif model_setting["fusion_method"] == "where2comm":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                Where2commFusion(model_setting["fusion_backbone"]),
            )
        elif model_setting["fusion_method"] == "who2com":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                Who2comFusion(model_setting["fusion_backbone"]),
            )
        elif model_setting["fusion_method"] == "pyramid":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                PyramidFusion(model_setting["fusion_backbone"]),
            )
        else:
            raise NotImplementedError(f"Method {model_setting['fusion_method']} not implemented.")

        if model_setting["fusion_method"] != "pyramid":
            # other method does not have agent_modality_list and cam_crop_info, neither returning occ_single_list
            pyramid_backbone = getattr(self, f"pyramid_backbone_{modality_name}")
            pyramid_backbone.forward_collab = lambda *args: (
                pyramid_backbone.forward(*args[:3]),
                [],
            )

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

        fused_feature, occ_outputs = eval(f"self.pyramid_backbone_{modality_name}").forward_collab(
            feature,
            record_len,
            affine_matrix,
            agent_modality_list,
            self.cam_crop_info,
            # transform_idx=0,
        )

        output_dict.update({"occ_single_list": occ_outputs})

        return fused_feature

    def forward_head(self, feature, modality_name, output_dict):
        """
        Forwards the feature through the head network.

        Parameters:
        - feature (Tensor): Fused features.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - output_dict (dict): Output data dictionary.

        This function handles the forward pass for different head methods such as object detection, segmentation, etc.
        """

        if self.head_method == "bev_seg_head":
            output_dict.update(self.head(feature))
        else:
            cls_preds = eval(f"self.cls_head_{modality_name}")(feature)
            reg_preds = eval(f"self.reg_head_{modality_name}")(feature)
            if hasattr(self, f"dir_head_{modality_name}"):
                dir_preds = eval(f"self.dir_head_{modality_name}")(feature)
            else:
                dir_preds = None

            output_dict.update({"cls_preds": cls_preds, "reg_preds": reg_preds, "dir_preds": dir_preds})

    def forward_shrink(self, feature, modality_name):
        """
        Forwards the feature through the shrink header if available.

        Parameters:
        - feature (Tensor): Aligned features.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').

        Returns:
        - feature (Tensor): Shrunken features.
        """

        if getattr(self, f"shrink_flag_{modality_name}"):
            feature = eval(f"self.shrink_conv_{modality_name}")(feature)
        return feature
