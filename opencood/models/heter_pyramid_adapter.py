""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
import torchvision
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.fuse_modules.adapter import Adapter, Reverter
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
import importlib
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn

class HeterPyramidAdapter(nn.Module):
    def __init__(self, args):
        super(HeterPyramidAdapter, self).__init__()
        self.ego_modality = args.get('ego_modality', None)
        self.protocol_modality = args.get('protocol_modality', None)
        assert self.ego_modality is not None, "Ego modality should be provided"
        assert self.protocol_modality is not None, "Protocol modality should be provided"
        
        
        self.modality_name_list = [self.ego_modality, self.protocol_modality]
        
        self.sensor_type_dict = OrderedDict()
        self.fix_modules = []
        # self.fix_modules = ['pyramid_backbone', 'cls_head', 'reg_head', 'dir_head']
        
        
        # # setup each modality model
        # for modality_name in self.modality_name_list:
        for modality_name in self.modality_name_list:
            
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            setattr(self, f"cav_range_{modality_name}", model_setting['lidar_range'])
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            # build encoder
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            # depth supervision for camera
            if model_setting['encoder_args'].get("depth_supervision", False) :
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            # setup backbone (very light-weight)
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", 
                        eval(f"self.cav_range_{modality_name}[3]") / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", 
                        eval(f"self.cav_range_{modality_name}[4]") / (camera_mask_args['grid_conf']['ybound'][1]))

            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))

            if args.get("fix_encoder", False):
                self.fix_modules += [f"encoder_{modality_name}", f"backbone_{modality_name}", f"aligner_{modality_name}"]

            """
            Adapter and Reverter
            """
            if modality_name == self.ego_modality:
                setattr(self, f"adapter_{modality_name}", Adapter(model_setting['adapter']))
                setattr(self, f"reverter_{modality_name}", Reverter(model_setting['reverter']))
            
        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)

    def model_train_init(self):
        for module in self.fix_modules:
            for p in eval(f"self.{module}").parameters():
                p.requires_grad_(False)
            eval(f"self.{module}").apply(fix_bn)

    def forward(self, data_dict):
        # output_dict = {'pyramid': 'single'}
        modality_names = [x for x in list(data_dict.keys()) if x.startswith("inputs_")]
        assert len(modality_names) == 2, "Only ego modality and protocol modality should be provided"
        modality_names = [x.lstrip('inputs_') for x in modality_names]
        assert self.ego_modality in modality_names, "Ego modality should be provided"
        assert self.protocol_modality in modality_names, "Protocol modality should be provided"

        # Ego modality
        with torch.no_grad():
            ego_feature = eval(f"self.encoder_{self.ego_modality}")(data_dict, self.ego_modality)
            ego_feature = eval(f"self.backbone_{self.ego_modality}")({"spatial_features": ego_feature})['spatial_features_2d']
            ego_feature = eval(f"self.aligner_{self.ego_modality}")(ego_feature)
            
        # Protocol modality
        with torch.no_grad():
            protocol_feature = eval(f"self.encoder_{self.protocol_modality}")(data_dict, self.protocol_modality)
            protocol_feature = eval(f"self.backbone_{self.protocol_modality}")({"spatial_features": protocol_feature})['spatial_features_2d']
            protocol_feature = eval(f"self.aligner_{self.protocol_modality}")(protocol_feature)

        if self.sensor_type_dict[self.ego_modality] == "camera":
            # should be padding. Instead of masking
            _, _, H, W = protocol_feature.shape
            protocol_feature = torchvision.transforms.CenterCrop(
                    (int(H / eval(f"self.crop_ratio_H_{self.ego_modality}")), int(W / eval(f"self.crop_ratio_W_{self.ego_modality}")))
                )(protocol_feature)
            
        # Adapter
        ego_feature = ego_feature.detach()
        protocol_feature = protocol_feature.detach()
        e2p_feature = eval(f"self.adapter_{self.ego_modality}")(ego_feature)
        p2e_feature = eval(f"self.reverter_{self.ego_modality}")(protocol_feature)
        e2p2e_feature = eval(f"self.reverter_{self.ego_modality}")(e2p_feature.detach())

        return ego_feature, protocol_feature, p2e_feature, e2p2e_feature, e2p_feature
