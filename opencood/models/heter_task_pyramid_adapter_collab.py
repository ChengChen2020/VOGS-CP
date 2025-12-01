""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License

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
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision

class HeterTaskPyramidAdapterCollab(nn.Module):
    def __init__(self, args):
        super(HeterTaskPyramidAdapterCollab, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        self.sensor_type_dict = OrderedDict()
        self.cam_crop_info = {} 

        # setup each modality model
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

            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building 
            """
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))

            """
            Aligner building
            """
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", 
                        eval(f"self.cav_range_{modality_name}[3]") / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", 
                        eval(f"self.cav_range_{modality_name}[4]") / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }


            # TODO: does not currently support more than one lidar range
            """For feature transformation"""
            setattr(self, f"H_{modality_name}", 
                    eval(f"self.cav_range_{modality_name}[4]") - eval(f"self.cav_range_{modality_name}[1]"))
            setattr(self, f"W_{modality_name}",
                    eval(f"self.cav_range_{modality_name}[3]") - eval(f"self.cav_range_{modality_name}[0]"))
            self.fake_voxel_size = 1
            # TODO: does not currently support more than one lidar range
            self.H = eval(f"self.H_{modality_name}")
            self.W = eval(f"self.W_{modality_name}")
            

            """
            Fusion, by default multiscale fusion: 
            Note the input of PyramidFusion has downsampled 2x. (SECOND required)
            """
            setattr(self, f"pyramid_backbone_{modality_name}", PyramidFusion(model_setting['fusion_backbone']))


            """
            Shrink header
            """
            self.shrink_flag = False
            if 'shrink_header' in model_setting:
                self.shrink_flag = True
                setattr(self, f"shrink_conv_{modality_name}", DownsampleConv(model_setting['shrink_header']))

            """
            Shared Heads
            """
            setattr(self, f"cls_head_{modality_name}", 
                    nn.Conv2d(model_setting['in_head'], 
                              model_setting['anchor_number'], 
                              kernel_size=1))
            setattr(self, f"reg_head_{modality_name}", 
                    nn.Conv2d(model_setting['in_head'], 
                              7 * model_setting['anchor_number'], 
                              kernel_size=1))
            setattr(self, f"dir_head_{modality_name}", 
                    nn.Conv2d(model_setting['in_head'], 
                              model_setting['dir_args']['num_bins'] * model_setting['anchor_number'], 
                              kernel_size=1))
            
            # compressor will be only trainable
            self.compress = False
            if 'compressor' in model_setting:
                self.compress = True
                setattr(self, f"compressor_{modality_name}", 
                        NaiveCompressor(model_setting['compressor']['input_dim'],
                                        model_setting['compressor']['compress_ratio']))
                
            setattr(self, f"adapter_{modality_name}", Adapter(model_setting['adapter']))
            setattr(self, f"reverter_{modality_name}", Reverter(model_setting['reverter']))
                
        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)


    def model_train_init(self):
        # if compress, only make compressor trainable
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            for modality_name in self.modality_name_list:
                if f"compressor_{modality_name}" in self.__dict__:
                    eval(f"self.compressor_{modality_name}").train()
                    for p in eval(f"self.compressor_{modality_name}").parameters():
                        p.requires_grad_(True)

    def forward(self, data_dict):
        
        agent_modality_list = data_dict['agent_modality_list'] 
        output_dict = [{'pyramid': 'collab'} for _ in agent_modality_list]
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature
            
            

        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict[modality_name].update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })
                        
        """
        Assemble heter features
        """
        protocol_features = dict()
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                protocol_features[f"e2p_feature_{modality_name}"] = eval(f"self.adapter_{modality_name}")(modality_feature_dict[modality_name])

        cur_feature_dict = dict()
        for cur in self.modality_name_list:
            if cur not in modality_count_dict:
                continue
            cur_feature_dict[cur] = dict()
            for src in self.modality_name_list:
                if src not in modality_count_dict:
                    continue
                if cur == src:
                    cur_feature_dict[cur][src] = modality_feature_dict[cur]
                else:
                    cur_feature_dict[cur][src] = eval(f"self.reverter_{cur}")(protocol_features[f"e2p_feature_{src}"])  
        
        for modality_name in self.modality_name_list:
            
            if modality_name not in modality_count_dict:
                continue
            
            counting_dict = {m:0 for m in self.modality_name_list}
            heter_feature_2d_list_dict = {m:[] for m in self.modality_name_list}
            for m in agent_modality_list:
                feat_idx = counting_dict[m]
                heter_feature_2d_list_dict[modality_name].append(cur_feature_dict[modality_name][m][feat_idx])
                counting_dict[m] += 1
                
            heter_feature_2d_list_dict[modality_name] = torch.stack(heter_feature_2d_list_dict[modality_name])
        

            if self.compress:
                heter_feature_2d_list_dict[modality_name] = eval(f"self.compressor_{modality_name}")(heter_feature_2d_list_dict[modality_name])

            # heter_feature_2d_list_dict[modality_name] is downsampled 2x
            # add croping information to collaboration module
            
            for agent_idx, m in enumerate(agent_modality_list):
                
                if m != modality_name:
                    continue
                
                fused_feature, occ_outputs = eval(f"self.pyramid_backbone_{modality_name}").forward_collab(
                                                        heter_feature_2d_list_dict[modality_name],
                                                        record_len, 
                                                        affine_matrix, 
                                                        agent_modality_list, 
                                                        self.cam_crop_info,
                                                        transform_idx = agent_idx,
                                                    )

                if self.shrink_flag:
                    fused_feature = eval(f"self.shrink_conv_{modality_name}")(fused_feature)

                cls_preds = eval(f"self.cls_head_{modality_name}")(fused_feature)
                reg_preds = eval(f"self.reg_head_{modality_name}")(fused_feature)
                dir_preds = eval(f"self.dir_head_{modality_name}")(fused_feature)

                            

                output_dict[agent_idx].update(
                    {
                        'cls_preds': cls_preds,
                        'reg_preds': reg_preds,
                        'dir_preds': dir_preds
                    })
                
                output_dict[agent_idx].update(
                    {
                        'occ_single_list': occ_outputs
                    })

        return output_dict
    
