from mmseg.models import build_segmentor, SEGMENTORS
from mmseg.registry import MODELS
from mmengine.model import BaseModule # similar to nn.module

import inspect
@SEGMENTORS.register_module()
class GaussianBackbone(BaseModule):
    def __init__(
        self,
        img_backbone=None,
        img_neck=None,
        img_backbone_out_indices=None,
    ):
        super().__init__()
        if img_backbone is not None:
            self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        self.img_backbone_out_indices = img_backbone_out_indices

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""

        result = {}

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        imgs = imgs[:, :3, :, :]
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        img_feats = self.img_neck(img_feats)
        if isinstance(img_feats, dict):
            secondfpn_out = img_feats["secondfpn_out"][0]
            BN, C, H, W = secondfpn_out.shape
            secondfpn_out = secondfpn_out.view(B, int(BN / B), C, H, W)
            img_feats = img_feats["fpn_out"]
            result.update({"secondfpn_out": secondfpn_out})

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        result.update({'ms_img_feats': img_feats_reshaped})

        return result

    def forward(self, data_dict, modality_name):
        input_data = data_dict[f"inputs_{modality_name}"]
        imgs = input_data.pop("imgs")
        results = {
            'imgs': imgs,
            'metas': input_data,
        }

        outs = self.extract_img_feat(**results)
        results.update(outs)

        return results