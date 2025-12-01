import torch
import torch.nn as nn

import numpy as np
from opencood.models.gaussian_modules.gaussian_utils import (
    get_rotation_from_quaternion,
    get_meshgrid,
)


class GaussianOccHead(nn.Module):
    def __init__(
        self,
        apply_loss_type='random_1',
        num_classes=14,
        cuda_kwargs=None,
        use_localaggprob=False,
        use_localaggprob_fast=False,
        combine_geosem=False,
        with_empty=False,
        empty_args=None,
        empty_label=13,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_localaggprob = use_localaggprob
        if use_localaggprob:
            if use_localaggprob_fast:
                import local_aggregate_prob_fast
                self.aggregator = local_aggregate_prob_fast.LocalAggregator(**cuda_kwargs)
            else:
                import local_aggregate_prob
                self.aggregator = local_aggregate_prob.LocalAggregator(**cuda_kwargs)
        else:
            import local_aggregate
            self.aggregator = local_aggregate.LocalAggregator(**cuda_kwargs)

        self.combine_geosem = combine_geosem
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0)
            self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, :])
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.empty_label = empty_label

        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
        elif 'random' in apply_loss_type:
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = int(apply_loss_type.split('_')[1])
        elif 'fixed' in apply_loss_type:
            self.apply_loss_type = 'fixed'
            self.fixed_apply_loss_layers = [int(item) for item in apply_loss_type.split('_')[1:]]
            print(f"Supervised fixed layers: {self.fixed_apply_loss_layers}")
        else:
            raise NotImplementedError

        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def prepare_gaussian_args(self, gaussians):

        if gaussians.mask is not None:

            valid_mask = gaussians.mask.bool()  # [B,N]
            valid_idx = valid_mask.nonzero(as_tuple=False)  # [M,2]

            if valid_idx.shape[0] == 0:
                raise ValueError("No valid Gaussians after masking.")

            batch_idx = valid_idx[:, 0]  # [M]
            gauss_idx = valid_idx[:, 1]  # [M]

            means = gaussians.means[batch_idx, gauss_idx]  # [M,3]
            scales = gaussians.scales[batch_idx, gauss_idx]  # [M,3]
            rotations = gaussians.rotations[batch_idx, gauss_idx]  # [M,4]
            opacities = gaussians.opacities[batch_idx, gauss_idx]  # [M,1]
            semantics = gaussians.semantics[batch_idx, gauss_idx]  # [M,C]
        else:
            means = gaussians.means[0]
            scales = gaussians.scales[0]
            rotations = gaussians.rotations[0]
            opacities = gaussians.opacities[0]
            semantics = gaussians.semantics[0]

        if opacities.numel() == 0:
            opacities = torch.ones_like(opacities[..., :1], requires_grad=False)
        if self.with_emtpy:
            assert semantics.shape[-1] == self.num_classes - 1
            semantics = torch.cat([semantics, torch.zeros_like(opacities[..., :1])], dim=-1)
            means = torch.cat([means, self.empty_mean], dim=0)
            scales = torch.cat([scales, self.empty_scale], dim=0)
            rotations = torch.cat([rotations, self.empty_rot], dim=0)
            empty_sem = self.empty_sem.clone()
            empty_sem[..., self.empty_label] += self.empty_scalar
            semantics = torch.cat([semantics, empty_sem], dim=0)
            opacities = torch.cat([opacities, self.empty_opa], dim=0)

        bsg, _ = means.shape
        S = torch.zeros(bsg, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = get_rotation_from_quaternion(rotations)  # b, g, 3, 3
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        CovInv = torch.inverse(Cov)
        return means, opacities, semantics, scales, CovInv

    def _sampling(self, gt_xyz, gt_label, gt_mask=None):
        if gt_mask is None:
            gt_label = gt_label.flatten(1)
            gt_xyz = gt_xyz.flatten(1, 3)
        else:
            assert gt_label.shape[0] == 1, "OccLoss does not support bs > 1"
            gt_label = gt_label[gt_mask].reshape(1, -1)
            gt_xyz = gt_xyz[gt_mask].reshape(1, -1, 3)
        return gt_xyz, gt_label

    def forward(
        self,
        representation,
        metas=None,
        **kwargs
    ):
        num_decoder = len(representation)
        if not self.training:
            apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == "all":
            apply_loss_layers = list(range(num_decoder))
        elif self.apply_loss_type == "random":
            if self.random_apply_loss_layers > 1:
                apply_loss_layers = np.random.choice(num_decoder - 1, self.random_apply_loss_layers - 1, False)
                apply_loss_layers = apply_loss_layers.tolist() + [num_decoder - 1]
            else:
                apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == 'fixed':
            apply_loss_layers = self.fixed_apply_loss_layers
        else:
            raise NotImplementedError

        assert len(apply_loss_layers) == 1
        assert apply_loss_layers[0] == num_decoder - 1

        prediction = []
        bin_logits = []
        density = []
        occ_xyz = metas['occ_xyz'].to(self.zero_tensor.device)
        occ_label = metas['occ_label'].to(self.zero_tensor.device)
        occ_cam_mask = metas['occ_cam_mask'].to(self.zero_tensor.device)
        sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)
        
        for idx in apply_loss_layers:
            gaussians = representation[idx]['gaussian']


            means, opacities, semantics, scales, CovInv = self.prepare_gaussian_args(gaussians)
            bs, g = means.shape[:2]


            semantics = self.aggregator(
                sampled_xyz.clone().float(),
                means.unsqueeze(0),
                opacities.squeeze().unsqueeze(0),
                semantics.unsqueeze(0),
                scales.unsqueeze(0),
                CovInv.unsqueeze(0)
            )

            if self.use_localaggprob:
                if self.combine_geosem:
                    sem = semantics[0][:, :-1] * semantics[1].unsqueeze(-1)
                    geo = 1 - semantics[1].unsqueeze(-1)
                    geosem = torch.cat([sem, geo], dim=-1)
                else:
                    geosem = semantics[0]

                prediction.append(geosem[None].transpose(1, 2))
                bin_logits.append(semantics[1][None])
                density.append(semantics[2][None])
            else:
                prediction.append(semantics[None].transpose(1, 2))

        if self.use_localaggprob and not self.combine_geosem:
            threshold = kwargs.get("sigmoid_thresh", 0.5)
            final_semantics = prediction[-1].argmax(dim=1)
            final_occupancy = bin_logits[-1] > threshold
            final_prediction = torch.ones_like(final_semantics) * self.empty_label
            final_prediction[final_occupancy] = final_semantics[final_occupancy]
        else:
            final_prediction = prediction[-1].argmax(dim=1)


        return {
            'pred_occ': prediction,
            'bin_logits': bin_logits,
            'density': density,
            'sampled_label': sampled_label,
            'sampled_xyz': sampled_xyz,
            'occ_mask': occ_cam_mask,
            'final_occ': final_prediction,
            'gaussian': representation[-1]['gaussian'],
            'gaussians': [r['gaussian'] for r in representation]
        }

if __name__ == '__main__':
    GaussianOccHead()
    