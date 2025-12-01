import torch
import torch.nn as nn

from opencood.models.gaussian_modules.gaussian_utils import safe_inverse_sigmoid

import inspect

class GaussianLifter(nn.Module):
    def __init__(
        self,
        type='v1',
        num_anchor=25600,
        embed_dims=128,
        anchor_grad=False,
        feat_grad=True,
        semantics=True,
        semantic_dim=13,
        include_opa=True,
        pts_init=False,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.pts_init = pts_init
        assert not (pts_init and anchor_grad)

        xyz = torch.rand(num_anchor, 3, dtype=torch.float)
        xyz = safe_inverse_sigmoid(xyz)

        scale = torch.rand_like(xyz)
        scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        # Initial opacity 0.5
        if include_opa:
            opacity = safe_inverse_sigmoid(0.5 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)

        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1)

        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(
            anchor.clone().detach().float(),
            requires_grad=anchor_grad
        )
        # fixed copy for reset/visualization
        self.anchor_init = anchor.clone().detach().float()
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )

    def init_weights(self):
        with torch.no_grad():
            self.anchor.copy_(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, imgs, metas, **kwargs):
        batch_size = imgs.shape[0]
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        if self.pts_init:
            if self.xyz_act == "sigmoid":
                xyz = safe_inverse_sigmoid(metas['anchor_points'])
            anchor = torch.cat([
                xyz, torch.tile(self.anchor[None, :, 3:], (batch_size, 1, 1))], dim=-1)
        else:
            anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))
            
        return {
            'rep_features': instance_feature, # Gaussian queries
            'representation': anchor,         # Gaussian properties
            'anchor_init': self.anchor_init.clone()
        }
        