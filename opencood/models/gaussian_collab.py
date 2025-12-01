
import torch
import cv2
import torch.nn as nn
import numpy as np
from collections import Counter

import inspect
import importlib

from opencood.models.gaussian_modules.backbone import GaussianBackbone
from opencood.models.gaussian_modules.lifter import GaussianLifter
from opencood.models.gaussian_modules.encoder import GaussianEncoder
from opencood.models.gaussian_modules.gaussian_fuse import (
    GaussianCollabRefiner,
    transform_neighbor_gaussians,
)

class GaussianCollab(nn.Module):
    def __init__(self, args):
        super(GaussianCollab, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()]
        self.modality_name_list = modality_name_list

        modality_name = self.modality_name_list[0]
        model_setting = args[modality_name]

        self.num_gaussian = model_setting["lifter_args"]["num_anchor"]
        print("num_gaussian =", self.num_gaussian)

        """
        Backbone building
        """
        setattr(self, f"backbone_{modality_name}", GaussianBackbone(**model_setting['backbone_args']))

        """
        Lifter building
        """
        setattr(self, f"lifter_{modality_name}", GaussianLifter(**model_setting['lifter_args']))

        """
        Encoder building
        """
        setattr(self, f"encoder_{modality_name}", GaussianEncoder(**model_setting['encoder_args']))

        """
        Collaboration refiner building
        """
        self.learned_refiner = model_setting.get('learned_refiner', False)
        if self.learned_refiner:
            setattr(self, f"refiner_{modality_name}", GaussianCollabRefiner(**model_setting['refiner_args']))

        """
        Shared Heads
        """
        self.head_method = model_setting.get('head_method', None)

        if self.head_method == "occ_head":
            from opencood.models.gaussian_modules.occ_head import GaussianOccHead
            setattr(self, f"occ_head_{modality_name}", GaussianOccHead(**model_setting['occ_args']))

        elif self.head_method == "det_head":
            from opencood.models.gaussian_modules.det_head import GaussianDetHead
            setattr(self, f"det_head_{modality_name}", GaussianDetHead(**model_setting['det_args']))

        elif self.head_method == "seg_head":
            from opencood.models.gaussian_modules.seg_head import GaussianSegHead
            setattr(self, f"seg_head_{modality_name}", GaussianSegHead(**model_setting['seg_args']))

        else:
            assert False, "unknown head_method"


    def init_weights(self, verbose=True):
        """Universal init for GaussianCollab model."""
        for name, module in self.named_children():
            if hasattr(module, 'init_weights') and callable(module.init_weights):
                if verbose:
                    print(f"[Init] Initializing module: {name}")
                module.init_weights()


    def forward(self, data_dict, show_bev=False):
        output_dict = {'pyramid': 'collab'}
        record_len = data_dict['record_len'] # [2, 2, 4, 5]
        assert len(record_len) == 1, "only support one record_len"

        modality_name = self.modality_name_list[0]

        results = eval(f"self.backbone_{modality_name}")(data_dict, modality_name)

        outs = eval(f"self.lifter_{modality_name}")(**results)
        results.update(outs)

        outs = eval(f"self.encoder_{modality_name}")(**results)
        results.update(outs)

        num_of_gaussian_list = []

        #fuse
        if record_len[0] > 1:
            # Fuse with shared neighbors
            fused_gaussian, num_of_gaussian_list = transform_neighbor_gaussians(
                gaussian_pred=results['representation'][-1]['gaussian'],
                record_len=record_len,
                pairwise_t_matrix=data_dict['pairwise_t_matrix'],
                roi_bounds=(-20, -20, -2.3, 20, 20, 0.9),
                # opacity_thresh=0.05
            )
            results['representation'][-1]['gaussian'] = fused_gaussian

            # Optional learned refinement
            if fused_gaussian.means.shape[1] > self.num_gaussian:
                if self.learned_refiner:
                    refiner = getattr(self, f"refiner_{modality_name}")
                    refined_gaussian = refiner(fused_gaussian)
                    results['representation'][-1]['gaussian'] = refined_gaussian

        # Fused metas
        if self.head_method == "occ_head":
            results['metas'].update({
                'occ_xyz': data_dict['label_dict']['occ_xyz'],
                'occ_label': data_dict['label_dict']['occ_label'],
                'occ_cam_mask': data_dict['label_dict']['occ_cam_mask'],
            })

        output_dict.update({'gaussian': results['representation'][-1]['gaussian']})
        output_dict.update({'gaussians': [r['gaussian'] for r in results['representation']]})
        output_dict.update({'anchor_init': results['anchor_init']})
        output_dict.update({'neighbor_gaussians': num_of_gaussian_list})
        output_dict.update(eval(f"self.occ_head_{modality_name}")(**results))

        return output_dict
