""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License

import torch
import cv2
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, CoBEVT, Where2commFusion, Who2comFusion
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn

import importlib
import torchvision

class HomoCollab(nn.Module):
    def __init__(self, args):
        super(HomoCollab, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        self.sensor_type_dict = OrderedDict()
        self.cam_crop_info = {} 

        # setup each modality model
        modality_name = self.modality_name_list[0]
        model_setting = args[modality_name]
        setattr(self, f"cav_range_{modality_name}", model_setting['lidar_range'])
        sensor_name = model_setting['sensor_type']
        self.sensor_type_dict[modality_name] = sensor_name

        # import model
        encoder_filename = "opencood.models.heter_encoders"
        encoder_lib = importlib.import_module(encoder_filename)
        encoder_class = None
        self.multi_sensor = False
        if isinstance(model_setting['core_method'], str):
            self.multi_sensor = False
            target_model_name = model_setting['core_method'].replace('_', '')
            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls
                    
        elif isinstance(model_setting['core_method'], dict):
            self.multi_sensor = True
            target_model_name_camera = model_setting['core_method']['camera'].replace('_', '')
            target_model_name_lidar = model_setting['core_method']['lidar'].replace('_', '')
            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name_camera.lower():
                    encoder_class_camera = cls
                if name.lower() == target_model_name_lidar.lower():
                    encoder_class_lidar = cls

        """
        Encoder building
        """
        
        if self.multi_sensor:
            
            assert model_setting.get('encoder_args_camera', None) and model_setting.get('encoder_args_lidar', None), \
                "for multi_sensor, encoder_args_camera and encoder_args_lidar should be provided"
            setattr(self, f"encoder_{modality_name}_camera", encoder_class_camera(model_setting['encoder_args_camera']))
            setattr(self, f"encoder_{modality_name}_lidar", encoder_class_lidar(model_setting['encoder_args_lidar']))
            if model_setting['encoder_args_camera'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)
                
        else:
            
            assert model_setting.get('encoder_args', None), "encoder_args should be provided"
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)
            

        """
        Backbone building 
        """
        if model_setting.get('backbone_args', None):
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
        else:
            setattr(self, f"backbone_{modality_name}", lambda x: {'spatial_features_2d': nn.Identity()(x["spatial_features"])})

        """
        Aligner building
        """
        if model_setting.get('aligner_args', None):
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
        else:
            setattr(self, f"aligner_{modality_name}", AlignNet({"core_method": "identity"}))
        if "camera" in sensor_name:
            camera_mask_args = model_setting['camera_mask_args']
            setattr(self, f"crop_ratio_W_{modality_name}", 
                    (eval(f"self.cav_range_{modality_name}")[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
            setattr(self, f"crop_ratio_H_{modality_name}", 
                    (eval(f"self.cav_range_{modality_name}")[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
            setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
            setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
            self.cam_crop_info[modality_name] = {
                f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
            }

        """For feature transformation"""
        setattr(self, f"H_{modality_name}", 
                (eval(f"self.cav_range_{modality_name}")[4] - eval(f"self.cav_range_{modality_name}")[1]))
        setattr(self, f"W_{modality_name}",
                (eval(f"self.cav_range_{modality_name}")[3] - eval(f"self.cav_range_{modality_name}")[0]))
        self.fake_voxel_size = 1

        """
        Fusion, by default multiscale fusion: 
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """
        
        if model_setting['fusion_method'] == "max":
            setattr(self, f"pyramid_backbone_{modality_name}", MaxFusion())
        elif model_setting['fusion_method'] == "att":
            setattr(self, f"pyramid_backbone_{modality_name}", AttFusion(model_setting['fusion_backbone']))
        elif model_setting['fusion_method'] == "disconet":
            setattr(self, f"pyramid_backbone_{modality_name}", DiscoFusion(model_setting['fusion_backbone']))
        elif model_setting['fusion_method'] == "v2vnet":
            setattr(self, f"pyramid_backbone_{modality_name}", V2VNetFusion(model_setting['fusion_backbone']))
        elif model_setting['fusion_method'] == 'v2xvit':
            setattr(self, f"pyramid_backbone_{modality_name}", V2XViTFusion(model_setting['fusion_backbone']))
        elif model_setting['fusion_method'] == 'cobevt':
            setattr(self, f"pyramid_backbone_{modality_name}", CoBEVT(model_setting['fusion_backbone']))
        elif model_setting['fusion_method'] == 'where2comm':
            setattr(self, f"pyramid_backbone_{modality_name}", Where2commFusion(model_setting['fusion_backbone']))
        elif model_setting['fusion_method'] == 'who2com':
            setattr(self, f"pyramid_backbone_{modality_name}", Who2comFusion(model_setting['fusion_backbone']))
        elif model_setting['fusion_method'] == 'pyramid':
            setattr(self, f"pyramid_backbone_{modality_name}", PyramidFusion(model_setting['fusion_backbone']))
        else:
            raise NotImplementedError(f"Method {model_setting['fusion_method']} not implemented.")
            
        if model_setting['fusion_method'] != 'pyramid':
            # other method does not have agent_modality_list and cam_crop_info, neither returning occ_single_list
            pyramid_backbone = getattr(self, f"pyramid_backbone_{modality_name}")
            pyramid_backbone.forward_collab = lambda *args: (pyramid_backbone.forward(*args[:3]), [])


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
        # By default, point pillar pyramid object detection head 
        self.head_method = model_setting.get('head_method', "point_pillar_pyramid_object_detection_head")
        self.downsample_rate = model_setting.get('downsample_rate', 1)
        
        if self.head_method == "point_pillar_pyramid_object_detection_head":
            
            setattr(self, f"cls_head_{modality_name}", 
                    nn.Conv2d(model_setting['in_head'], model_setting['anchor_number'], 
                            kernel_size=1))
            setattr(self, f"reg_head_{modality_name}",
                    nn.Conv2d(model_setting['in_head'], 7 * model_setting['anchor_number'],
                            kernel_size=1))
            if model_setting.get('dir_args', None):
                setattr(self, f"dir_head_{modality_name}",
                        nn.Conv2d(model_setting['in_head'], model_setting['dir_args']['num_bins'] * model_setting['anchor_number'],
                                kernel_size=1))            
            
        elif self.head_method == "point_pillar_object_detection_head":
            setattr(self, f"cls_head_{modality_name}", 
                    nn.Conv2d(model_setting['in_head'], 1, 
                            kernel_size=1))
            setattr(self, f"reg_head_{modality_name}",
                    nn.Conv2d(model_setting['in_head'], 7,
                            kernel_size=1))
            if model_setting.get('dir_args', None):
                setattr(self, f"dir_head_{modality_name}",
                        nn.Conv2d(model_setting['in_head'], model_setting['dir_args']['num_bins'],
                                kernel_size=1))   
        
        elif self.head_method == "bev_seg_head":
            setattr(
                self, f"head_{modality_name}", nn.Sequential(
                NaiveDecoder(model_setting['decoder_args']),
                BevSegHead(model_setting['target'],
                            model_setting['seg_head_dim'],
                            model_setting['output_class_dynamic'],
                            model_setting['output_class_static'])
                )
                )
        elif self.head_method == "seg_head":
                setattr(
                self,
                f"head_{modality_name}",
                nn.Sequential(
                BevSegHead(
                    model_setting["target"],
                    model_setting["seg_head_dim"],
                    model_setting["output_class_dynamic"],
                    model_setting["output_class_static"],
                    ),
                ),
            )
        
        elif self.head_method == "pixor_head":
            
            setattr(self, f"cls_head_{modality_name}", 
                    nn.Conv2d(model_setting['in_head'], 1, 
                            kernel_size=1))
            setattr(self, f"reg_head_{modality_name}",
                    nn.Conv2d(model_setting['in_head'], 6,
                            kernel_size=1))
            
        else:
            raise NotImplementedError(f"Head method {self.head_method} not implemented.")
        
        
        # compressor will be only trainable
        self.compress = False
        if 'compressor' in model_setting:
            self.compress = True
            setattr(self, f"compressor_{modality_name}", 
                    NaiveCompressor(model_setting['compressor']['input_dim'],
                                    model_setting['compressor']['compress_ratio']))

        self.model_train_init()
        # check again which module is not fixed.
        # check_trainable_module(self)


    def model_train_init(self):
        # if compress, only make compressor trainable
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict, show_bev=False):
        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list'] 
        record_len = data_dict['record_len'] 
        modality_count_dict = Counter(agent_modality_list)
        assert len(modality_count_dict.keys()) == 1, "Cannot have more than one modality in Homo setting"
        modality_feature_dict = {}

        modality_name = self.modality_name_list[0]
        if modality_name in modality_count_dict:
            
            if self.multi_sensor:
                feature_camera = eval(f"self.encoder_{modality_name}_camera")(data_dict, modality_name, self.multi_sensor)
                """
                Crop/Padd camera feature map.
                """
                if "camera" in self.sensor_type_dict[modality_name]:
                    # should be padding. Instead of masking
                    _, _, H, W = feature_camera.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    feature_camera = crop_func(feature_camera)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}_camera").depth_items
                        })
                        
                
                feature_lidar = eval(f"self.encoder_{modality_name}_lidar")(data_dict, modality_name, self.multi_sensor)
                
                feature = feature_camera + feature_lidar
            else:
                feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name, self.multi_sensor)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            

            modality_feature_dict[modality_name] = feature
            
            if not self.multi_sensor:
                """
                Crop/Padd camera feature map.
                """
                if "camera" in self.sensor_type_dict[modality_name]:
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })
                        
            
           
        """
        Assemble heter features
        """
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1
            
        heter_feature_2d = torch.stack(heter_feature_2d_list)
        
        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], 
                                        eval(f"self.H_{modality_name}"), 
                                        eval(f"self.W_{modality_name}"), 
                                        self.fake_voxel_size)
        
        # feature, occ_outputs = eval(f"self.pyramid_backbone_{modality_name}").forward_collab(
        #                                         data_dict['label_dict']['gt_static'].unsqueeze(1).transpose(-1,-2).flip(-1),
        #                                         record_len, 
        #                                         affine_matrix, 
        #                                         agent_modality_list, 
        #                                         self.cam_crop_info
        #                                     )
        
        # heter_feature_2d_rotated = heter_feature_2d.transpose(-1,-2).flip(-1)
        feature, occ_outputs = eval(f"self.pyramid_backbone_{modality_name}").forward_collab(
                                                heter_feature_2d,
                                                record_len, 
                                                affine_matrix, 
                                                agent_modality_list, 
                                                self.cam_crop_info
                                            )
        # feature = feature.transpose(-1,-2).flip(-1)
        
        if self.shrink_flag:
            feature = eval(f"self.shrink_conv_{modality_name}")(feature)
        
        


        if self.head_method == "bev_seg_head":
            output_dict.update(eval(f"self.head_{modality_name}")(feature))
        elif self.head_method == "seg_head":
            output_dict.update(eval(f"self.head_{modality_name}")(feature))
        else:
            cls_preds = eval(f"self.cls_head_{modality_name}")(feature)
            reg_preds = eval(f"self.reg_head_{modality_name}")(feature)
            if hasattr(self, f"dir_head_{modality_name}"):
                dir_preds = eval(f"self.dir_head_{modality_name}")(feature)
            else:
                dir_preds = None

            output_dict.update({'cls_preds': cls_preds,
                                'reg_preds': reg_preds,
                                'dir_preds': dir_preds})
        
        output_dict.update({'occ_single_list': 
                            occ_outputs})
        
        # import cv2
        # feat = heter_feature_2d
        # cv2.imwrite('debug/dynamic_seg.png', (output_dict['dynamic_seg'][0].softmax(0)[0].cpu().detach().numpy() * 255).astype(np.uint8))
        # cv2.imwrite('debug/dynamic_seg1.png', (output_dict['dynamic_seg'][0].softmax(0).argmax(0).cpu().detach().numpy() * 255).astype(np.uint8))
        # cv2.imwrite('debug/static_seg.png', (output_dict['static_seg'][0].softmax(0)[0].cpu().detach().numpy() * 255).astype(np.uint8))
        # cv2.imwrite('debug/static_seg1.png', ((output_dict['static_seg'][0].softmax(0).argmax(0) == 0).cpu().detach().numpy() * 255).astype(np.uint8))
        # cv2.imwrite('debug/static_seg2.png', ((output_dict['static_seg'][0].softmax(0).argmax(0) == 1).cpu().detach().numpy() * 255).astype(np.uint8))
        # cv2.imwrite('debug/static_seg3.png', ((output_dict['static_seg'][0].softmax(0).argmax(0) == 2).cpu().detach().numpy() * 255).astype(np.uint8))
        # cv2.imwrite('debug/test_gt.png', (data_dict['label_dict']['dynamic_bev'][0].cpu().detach().numpy() * 255).astype(np.uint8))
        # cv2.imwrite('debug/test_gt_static_1.png', ((data_dict['label_dict']['static_bev'][0] == 0).cpu().detach().numpy() * 255).astype(np.uint8))
        # cv2.imwrite('debug/test_gt_static.png', ((data_dict['label_dict']['static_bev'][0] == 1).cpu().detach().numpy() * 255).astype(np.uint8))
        # cv2.imwrite('debug/test_gt_static_2.png', ((data_dict['label_dict']['static_bev'][0] == 2).cpu().detach().numpy() * 255).astype(np.uint8))
        # cv2.imwrite('debug/test_feat.png', (((feature[0].mean(0) - feature[0].mean(0).min()) / (feature[0].mean(0).max() - feature[0].mean(0).min())).cpu().detach().numpy() * 255).astype(np.uint8))
        # cv2.imwrite('debug/test_feat_heter.png', (((heter_feature_2d[0].mean(0) - heter_feature_2d[0].mean(0).min()) / (heter_feature_2d[0].mean(0).max() - heter_feature_2d[0].mean(0).min())).cpu().detach().numpy() * 255).astype(np.uint8))
        # img = data_dict['inputs_m1']['imgs'][0,0,:3].permute(1,2,0)
        # cv2.imwrite('debug/test_img.png', (((img - img.min()) / (img.max() - img.min())).cpu().detach().numpy() * 255).astype(np.uint8))
        # import pdb; pdb.set_trace()
        
        
        
        # import cv2
        # feat = heter_feature_2d
        # cv2.imwrite('debug2/test_feat.png', (((feature[0].mean(0) - feature[0].mean(0).min()) / (feature[0].mean(0).max() - feature[0].mean(0).min())).cpu().detach().numpy() * 255).astype(np.uint8))
        # cv2.imwrite('debug2/test_feat_heter.png', (((heter_feature_2d[0].mean(0) - heter_feature_2d[0].mean(0).min()) / (heter_feature_2d[1].mean(0).max() - heter_feature_2d[1].mean(0).min())).cpu().detach().numpy() * 255).astype(np.uint8))
        # import pdb; pdb.set_trace()
        # from opencood.visualization import vis_utils, my_vis, simple_vis
        # simple_vis.visualize(dict(), data_dict['origin_lidar'][0],[-100, -40, -3, 100, 40, 1],"debug2",method='bev',left_hand=True)
        if show_bev:
            return (output_dict, 
                    feature,
                    heter_feature_2d,
                    None,
                    None,
                    None,
                    None)

        return output_dict
