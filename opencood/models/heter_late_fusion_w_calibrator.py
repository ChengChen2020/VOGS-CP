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
from opencood.models.heter_late_fusion import HeterLateFusion
from opencood.models.sub_modules.calibrators import DoublyBoundedScaling, PlattScaling, TemperatureScaling

import importlib
import torchvision


class HeterLateFusionWCalibrator(HeterLateFusion):

    def __init__(self, args):
        super(HeterLateFusionWCalibrator, self).__init__(args)
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            self.build_calibrator(modality_name, model_setting)
        self.model_train_init_calibrator()        

    def build_calibrator(self, modality_name, model_setting):
        assert "calibrator" in model_setting, f"Calibrator is not defined in {modality_name}"
        if model_setting["calibrator"]['core_method'] == 'DBS':
            calib = DoublyBoundedScaling()
        elif model_setting["calibrator"]['core_method'] == 'Platt':
            calib = PlattScaling()
        elif model_setting["calibrator"]['core_method'] == 'Temp':
            calib = TemperatureScaling()
        else:
            raise NotImplementedError(f"Calibrator type {model_setting['calibrator_type']} is not implemented")
        setattr(self, f"calibrator_{modality_name}", calib)
        
        
    def model_train_init_calibrator(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        for modality_name in self.modality_name_list:
            calib = getattr(self, f"calibrator_{modality_name}")
            calib.train()
            for p in calib.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict):
        record_len = data_dict["record_len"]
        ego_idx = torch.cumsum(record_len, 0) - record_len[0]
        assert len(self.modality_name_list) == 1, "Only one modality is allowed for training calibrator"
        ego_modality = self.modality_name_list[0]
        output_dict = super().forward(data_dict)
        calib = getattr(self, f"calibrator_{ego_modality}")
        output_dict[ego_modality]["cls_preds"] = output_dict[ego_modality]["cls_preds"][ego_idx]
        output_dict[ego_modality]["reg_preds"] = output_dict[ego_modality]["reg_preds"][ego_idx]
        output_dict[ego_modality]["dir_preds"] = output_dict[ego_modality]["dir_preds"][ego_idx]
        output_dict[ego_modality]["cls_preds"] = calib(output_dict[ego_modality]["cls_preds"])
        output_dict = output_dict[ego_modality]
        return output_dict

