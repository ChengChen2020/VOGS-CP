# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

"""
Post processing for rgb camera groundtruth
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

from opencood.data_utils.post_processor.base_postprocessor import BasePostprocessor
    


class CameraBevPostprocessor(BasePostprocessor):
    """
    This postprocessor mainly transfer the uint bev maps to float.
    """

    def __init__(self, anchor_params, train):
        super(CameraBevPostprocessor, self).__init__(anchor_params, train)
        self.params = anchor_params
        self.train = train
        self.softmax = nn.Softmax(dim=1)
        


    def collate_batch(self, batch_dict_list):
        """
        Collate the batch dictionary list into a single dictionary.

        Parameters
        ----------
        batch_dict_list : list
            List of batch dictionaries.

        Returns
        -------
        A single batch dictionary.
        """
        collated_dict = dict()
        if batch_dict_list:
            for key in batch_dict_list[0]:
                collated_dict[key] = torch.cat([torch.tensor(batch_dict[key]) for batch_dict in batch_dict_list], 0)
        return collated_dict

    def generate_label(self, **kwargs):
        """
        Convert rgb images to binary output.

        Parameters
        ----------
        bev_map : np.ndarray
            Uint 8 image with 3 channels.
        """
        bev_dict = dict()
        label_dict = dict()

        for key in ["dynamic_bev", "road_bev", "lane_bev"]:
            bev_map = kwargs[key]
            bev_map = cv2.cvtColor(bev_map, cv2.COLOR_BGR2GRAY)
            bev_map = np.array(bev_map, dtype=float) / 255.0
            bev_map[bev_map > 0] = 1
            bev_dict[key] = bev_map

        label_dict["dynamic_bev"] = np.expand_dims(np.rot90(bev_dict["dynamic_bev"], k=3).astype(np.int64), 0)
        label_dict["static_bev"] = np.expand_dims(np.rot90(self.merge_label(bev_dict["road_bev"], bev_dict["lane_bev"]), k=3).astype(np.int64), 0)

        return label_dict

    def merge_label(self, road_map, lane_map):
        """
        Merge lane and road map into one.

        Parameters
        ----------
        static_map :
        lane_map :
        """
        merge_map = np.zeros((road_map.shape[0], road_map.shape[1]))
        merge_map[road_map == 1] = 1
        merge_map[lane_map == 1] = 2

        return merge_map

    def softmax_argmax(self, seg_logits):
        output_prob = self.softmax(seg_logits)
        output_map = torch.argmax(output_prob, dim=1)

        return output_prob, output_map

    def post_process_train(self, output_dict):
        """
        Post process the output of bev map to segmentation mask.
        todo: currently only for single vehicle bev visualization.

        Parameters
        ----------
        output_dict : dict
            The output dictionary that contains the bev softmax.

        Returns
        -------
        The segmentation map. (B, C, H, W) and (B, H, W)
        """
        pred = dict()
        pred_score = dict()
        output_dict = output_dict['ego']
        static_seg = output_dict["static_seg"]
        dynamic_seg = output_dict["dynamic_seg"]

        static_prob, static_map = self.softmax_argmax(static_seg)
        dynamic_prob, dynamic_map = self.softmax_argmax(dynamic_seg)

        pred.update(
            {
                "static_map": static_map,
                "dynamic_map": dynamic_map,
            }
        )
        
        pred_score.update(
            {
                "static_prob": static_prob,
                "dynamic_prob": dynamic_prob,
            }
        )


        return pred, pred_score
    
    def post_process_intermediate(self, cav_content, output_dict, m, **kwargs):
        if "static_seg" not in output_dict[m] or "dynamic_seg" not in output_dict[m]:
            return None
        single_output_dict = {
            "static_seg": output_dict[m]["static_seg"],
            "dynamic_seg": output_dict[m]["dynamic_seg"],
        }
        single_output_dict['ego'] = single_output_dict
        pred_box_tensor, pred_score = self.post_process(cav_content, single_output_dict)
        return pred_box_tensor, pred_score

    def post_process(self, batch_dict, output_dict, **kwargs):
        # todo: rignt now we don't support late fusion (only no fusion)
        pred, pred_score = self.post_process_train(output_dict)

        return pred, pred_score
    
    def generate_gt_intermediate(self, batch_dict, modality, feat_idx, **kwargs):
        
        gt_dict = dict()
        batch_dict = batch_dict['ego']
        gt_dict["dynamic_bev"] = batch_dict[f'label_dict_{modality}']["dynamic_bev"][feat_idx:feat_idx+1]
        gt_dict["static_bev"] = batch_dict[f'label_dict_{modality}']["static_bev"][feat_idx:feat_idx+1]
            
        return gt_dict
    
    def generate_gt(self, batch_dict):
        """
        Generate the ground truth for the bev map.

        Parameters
        ----------
        batch_dict : dict
            The input dictionary that contains the bev map.

        Returns
        -------
        The ground truth dictionary.
        """
        gt_dict = dict()
        batch_dict = batch_dict['ego']
        gt_dict["dynamic_bev"] = batch_dict['label_dict']["dynamic_bev"]
        gt_dict["static_bev"] = batch_dict['label_dict']["static_bev"]

        return gt_dict
