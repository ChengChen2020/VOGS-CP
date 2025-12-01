import math
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torch_cluster import radius
from typing import List, Optional, Tuple
from torch_scatter import scatter, scatter_mean, scatter_max, scatter_sum


class NeighborAttention(nn.Module):
    """
    Query from ego features, key from relation features. Multi-head dot-product.
    Returns a scalar weight per pair in [0,1] after per-ego softmax.
    """
    def __init__(self, ego_dim, rel_dim, heads=2, d_k=64, temp=None, dropout=0.0, use_bias_with_rel=True):
        super().__init__()
        self.heads = heads
        self.d_k = d_k
        self.q = nn.Linear(ego_dim, heads * d_k, bias=False)
        self.k = nn.Linear(rel_dim, heads * d_k, bias=False)
        self.bias = nn.Linear(rel_dim, heads, bias=False) if use_bias_with_rel else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.temp = (1.0 / math.sqrt(d_k)) if temp is None else float(temp)
        self.ln_q = nn.LayerNorm(ego_dim)
        self.ln_k = nn.LayerNorm(rel_dim)

    def forward(self, ego_feat_pairs, rel_feat_pairs, ego_idx, num_ego):
        # ego_feat_pairs: [E, ego_dim] (already repeated per neighbor)
        # rel_feat_pairs: [E, rel_dim]
        E = ego_feat_pairs.size(0)
        q = self.q(self.ln_q(ego_feat_pairs)).view(E, self.heads, self.d_k)
        k = self.k(self.ln_k(rel_feat_pairs)).view(E, self.heads, self.d_k)
        logits = (q * k).sum(-1) * self.temp  # [E, H]
        if self.bias is not None:
            logits = logits + self.bias(rel_feat_pairs)  # [E, H]

        # per-ego softmax (segment softmax), numerically stable
        max_per_ego, _ = scatter_max(logits, ego_idx, dim=0, dim_size=num_ego)             # [N_ego, H]
        logits = logits - max_per_ego[ego_idx]                                             # [E, H]
        w = torch.exp(logits)
        denom = scatter_sum(w, ego_idx, dim=0, dim_size=num_ego)                           # [N_ego, H]
        alpha = w / (denom[ego_idx] + 1e-6)                                                # [E, H]
        alpha = self.dropout(alpha)
        # collapse heads to one scalar weight per pair (simple average); keeps your reducer shapes unchanged
        alpha = alpha.mean(dim=-1, keepdim=True)                                           # [E, 1]
        return alpha


from opencood.models.gaussian_modules.gaussian_utils import (
    get_rotation_from_quaternion,
    get_quaternion_from_rotation,

    scale_rotation_to_covariance,
    covariance_to_scale_rotation,

    quaternion_multiply,

    get_meshgrid,

    cartesian,
    reverse_cartesian,

    safe_sigmoid,
    safe_inverse_sigmoid,

    linear_relu_ln,

    GaussianPrediction,
    Scale,
)


class GaussianCollabRefiner(nn.Module):
    def __init__(
        self,
        embed_dims=128,
        ego_gaussian_num=25600,
        semantic_dim=13,
        nei_radius=0.4,
        max_num_neighbors=64,
        pc_range=[-20.0, -20.0, -2.3, 20.0, 20.0, 0.9],
        scale_range=[0.01, 1.8],
        unit_xyz=[2.0, 2.0, 0.5],
        semantic_input='softmax',
        semantic_strategy='replace',
        in_loops=2,
        rel='only',
        avg_opa=True,
        use_attention=False,
        attention_args=dict(
            attn_heads=2,
            attn_dk=64,
            attn_temp=None,
            attn_dropout=0.0,
            attn_bias_with_rel=True,
            attn_blend=1.0,
            attn_detach=True,
        )
    ):
        super(GaussianCollabRefiner, self).__init__()
        self.embed_dims = embed_dims
        self.ego_gaussian_num = ego_gaussian_num
        self.semantic_dim = semantic_dim
        self.nei_radius = nei_radius
        self.max_num_neighbors = max_num_neighbors

        self.semantic_input = semantic_input
        self.semantic_strategy = semantic_strategy
        self.avg_opa = avg_opa

        self.pc_range = pc_range
        self.scale_range = scale_range

        self.register_buffer("unit_xyz_1", torch.tensor(unit_xyz, dtype=torch.float), persistent=False)

        self.get_xyz = partial(cartesian, pc_range=pc_range, use_sigmoid=True)
        self.reverse_xyz = partial(reverse_cartesian, pc_range=pc_range, use_sigmoid=True)

        # Construction of relative feature
        input_dim = 3 + 3 + 4 + 1 + semantic_dim + 3 + 3 + 1 + 1 + semantic_dim
        output_dim = 3 + 3 + 4 + 1 + semantic_dim

        self.use_attention = use_attention
        self.attention_args = attention_args
        self.rel = rel
        if self.use_attention:
            self.neighbor_attn = NeighborAttention(
                ego_dim = 3 + 3 + 4 + 1 + semantic_dim,
                rel_dim = 3 + 3 + 1 + 1 + semantic_dim,
                heads = attention_args["attn_heads"],
                d_k = attention_args["attn_dk"],
                temp = attention_args["attn_temp"],
                dropout = attention_args["attn_dropout"],
                use_bias_with_rel = attention_args["attn_bias_with_rel"],
            )

        self.layers = nn.Sequential(
            *linear_relu_ln(self.embed_dims, in_loops=in_loops, out_loops=2, input_dims=input_dim),
            nn.Linear(self.embed_dims, output_dim),
            Scale([1.0] * output_dim)
        )

    def forward(self, gaussian_pred):
        device = gaussian_pred.means.device
        ego_mean = gaussian_pred.means[0, :self.ego_gaussian_num]
        shared_mean = gaussian_pred.means[0, self.ego_gaussian_num:]

        ego_scale = gaussian_pred.scales[0, :self.ego_gaussian_num]
        shared_scale = gaussian_pred.scales[0, self.ego_gaussian_num:]

        ego_rot = gaussian_pred.rotations[0, :self.ego_gaussian_num]
        assert torch.allclose(ego_rot.norm(dim=-1), torch.ones_like(ego_rot[..., 0]), atol=1e-4), \
            "ego_rot is not normalized"
        shared_rot = gaussian_pred.rotations[0, self.ego_gaussian_num:]
        assert torch.allclose(shared_rot.norm(dim=-1), torch.ones_like(shared_rot[..., 0]), atol=1e-4), \
            "ego_rot is not normalized"

        ego_opacity = gaussian_pred.opacities[0, :self.ego_gaussian_num]
        assert torch.all((ego_opacity >= 0) & (ego_opacity <= 1)), "Opacity out of bounds: not in [0, 1]"
        shared_opacity = gaussian_pred.opacities[0, self.ego_gaussian_num:]
        assert torch.all((shared_opacity >= 0) & (shared_opacity <= 1)), "Opacity out of bounds: not in [0, 1]"

        ego_semantic = gaussian_pred.semantics[0, :self.ego_gaussian_num]
        assert torch.all(ego_semantic > 0), "Semantics have non-positive values (should be softplused)"
        shared_semantic = gaussian_pred.semantics[0, self.ego_gaussian_num:]
        assert torch.all(shared_semantic > 0), "Semantics have non-positive values (should be softplused)"

        ego_batch = torch.zeros(ego_mean.shape[0], dtype=torch.long, device=device)
        shared_batch = torch.zeros(shared_mean.shape[0], dtype=torch.long, device=device)

        ego_idx, neighbor_idx = radius(
            x=shared_mean,
            y=ego_mean,
            r=self.nei_radius,
            batch_x=shared_batch,
            batch_y=ego_batch,
            # max_num_neighbors=self.max_num_neighbors,
        )

        has_neighbor = torch.zeros(self.ego_gaussian_num, dtype=torch.bool, device=device)
        has_neighbor[ego_idx] = True

        # TO 0-1 SIGMOIDED
        anchor_ego_mean = reverse_cartesian(ego_mean, self.pc_range, use_sigmoid=False)
        anchor_shared_mean = reverse_cartesian(shared_mean, self.pc_range, use_sigmoid=False)

        # TO 0-1 SIGMOIDED
        anchor_ego_scale = (ego_scale - self.scale_range[0]) / (self.scale_range[1] - self.scale_range[0])
        anchor_shared_scale = (shared_scale - self.scale_range[0]) / (self.scale_range[1] - self.scale_range[0])

        # TO 0-1
        if self.semantic_input == 'softmax':
            anchor_ego_sem = ego_semantic.softmax(dim=-1)
            anchor_shared_sem = shared_semantic.softmax(dim=-1)
        elif self.semantic_input == 'hardmax':
            anchor_ego_sem = ego_semantic / ego_semantic.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
            anchor_shared_sem = shared_semantic / shared_semantic.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        else:
            assert False, "unknown semantic input"

        em, sm = anchor_ego_mean[ego_idx], anchor_shared_mean[neighbor_idx]
        es, ss = anchor_ego_scale[ego_idx], anchor_shared_scale[neighbor_idx]
        er, sr = ego_rot[ego_idx], shared_rot[neighbor_idx]
        ese, sse = anchor_ego_sem[ego_idx], anchor_shared_sem[neighbor_idx]
        eo, so = ego_opacity[ego_idx], shared_opacity[neighbor_idx]

        rel_mean = sm - em
        rel_scale = ss - es
        rot_dot = torch.sum(er * sr, dim=-1, keepdim=True)
        rel_sem = sse - ese
        rel_opa = so - eo


        ego_feat = torch.cat([em, es, er, ese, eo], dim=-1)
        if self.rel == 'partial':
            rel_feat = torch.cat([rel_mean, rel_scale, rot_dot, sse, so], dim=-1)
        elif self.rel == 'less':
            rel_feat = torch.cat([rel_mean, ss, rot_dot, sse, so], dim=-1)
        elif self.rel == 'more':
            rel_feat = torch.cat([rel_mean, rel_scale, rot_dot, rel_sem, rel_opa], dim=-1)
        else:
            assert False, "unknown relation input"
        pair_feat = torch.cat([ego_feat, rel_feat], dim=-1)
        pair_out = self.layers(pair_feat)

        # print(pair_out.shape)

        # Aggregate per ego
        delta_xyz = pair_out[:, :3]
        delta_scale = pair_out[:, 3:6]
        rot_update = pair_out[:, 6:10]
        opa_update = torch.sigmoid(pair_out[:, 10:11])
        sem_update = F.softplus(pair_out[:, 11:])

        if self.use_attention:
            alpha = self.neighbor_attn(ego_feat, rel_feat, ego_idx, self.ego_gaussian_num)
            if self.attention_args["attn_detach"]:
                alpha = alpha.detach()

            # attention-weighted sums
            a_xyz = scatter_sum(alpha * delta_xyz, ego_idx, dim=0, dim_size=self.ego_gaussian_num)
            a_scale = scatter_sum(alpha * delta_scale, ego_idx, dim=0, dim_size=self.ego_gaussian_num)
            a_rot = scatter_sum(alpha * rot_update, ego_idx, dim=0, dim_size=self.ego_gaussian_num)
            a_sem = scatter_sum(alpha * sem_update, ego_idx, dim=0, dim_size=self.ego_gaussian_num)

            # mean baselines (for blending)
            m_xyz = scatter_mean(delta_xyz, ego_idx, dim=0, dim_size=self.ego_gaussian_num)
            m_scale = scatter_mean(delta_scale, ego_idx, dim=0, dim_size=self.ego_gaussian_num)
            m_rot = scatter_mean(rot_update, ego_idx, dim=0, dim_size=self.ego_gaussian_num)
            m_sem = scatter_mean(sem_update, ego_idx, dim=0, dim_size=self.ego_gaussian_num)

            lam = float(self.attention_args["attn_blend"])
            agg_xyz = lam * a_xyz + (1.0 - lam) * m_xyz
            agg_scale = lam * a_scale + (1.0 - lam) * m_scale
            agg_rot = lam * a_rot + (1.0 - lam) * m_rot
            agg_sem = lam * a_sem + (1.0 - lam) * m_sem

            # opacity: keep your existing path (mean or max). You can also switch to attention if you want:
            if self.avg_opa:
                agg_opa = scatter_mean(opa_update, ego_idx, dim=0, dim_size=self.ego_gaussian_num)
            else:
                agg_opa, _ = scatter_max(opa_update, ego_idx, dim=0, dim_size=self.ego_gaussian_num)

        else:
            agg_xyz = scatter_mean(delta_xyz, ego_idx, dim=0, dim_size=self.ego_gaussian_num)
            agg_scale = scatter_mean(delta_scale, ego_idx, dim=0, dim_size=self.ego_gaussian_num)
            agg_rot = scatter_mean(rot_update, ego_idx, dim=0, dim_size=self.ego_gaussian_num)
            if self.avg_opa:
                agg_opa = scatter_mean(opa_update, ego_idx, dim=0, dim_size=self.ego_gaussian_num)
            else:
                agg_opa, _ = scatter_max(opa_update, ego_idx, dim=0, dim_size=self.ego_gaussian_num)
            agg_sem = scatter_mean(sem_update, ego_idx, dim=0, dim_size=self.ego_gaussian_num)

        # print(ego_idx.shape, agg_xyz.shape)

        # Apply only to those with neighbors
        refine_mean = ego_mean.clone()
        refine_mean[has_neighbor] += (2 * safe_sigmoid(agg_xyz[has_neighbor]) - 1.) * self.unit_xyz_1[None]
        refine_mean = self.get_xyz(self.reverse_xyz(refine_mean))

        refine_scale = ego_scale.clone()
        scale_sig = safe_sigmoid(agg_scale[has_neighbor])
        refine_scale[has_neighbor] = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * scale_sig

        refine_rot = ego_rot.clone()
        refine_rot[has_neighbor] = F.normalize(agg_rot[has_neighbor], p=2, dim=-1, eps=1e-6)

        refine_opa = ego_opacity.clone()
        if self.avg_opa:
            refine_opa[has_neighbor] = agg_opa[has_neighbor]
        else:
            refine_opa[has_neighbor] = torch.max(ego_opacity[has_neighbor], agg_opa[has_neighbor])


        refine_sem = ego_semantic.clone()
        if self.semantic_strategy == 'replace':
            refine_sem[has_neighbor] = agg_sem[has_neighbor]
        elif self.semantic_strategy == 'confv1':
            ego_conf = anchor_ego_sem.max(dim=-1, keepdim=True)[0]
            agg_conf = agg_sem.max(dim=-1, keepdim=True)[0]
            alpha = ego_conf / (ego_conf + agg_conf + 1e-6)
            refine_sem[has_neighbor] = alpha[has_neighbor] * ego_semantic[has_neighbor] + \
                                       (1.0 - alpha[has_neighbor]) * agg_sem[has_neighbor]
        elif self.semantic_strategy == 'confv2':
            # Normalize both sides only to compute confidences (prob-like); do not use these for blending
            ego_prob = ego_semantic / (ego_semantic.sum(dim=-1, keepdim=True) + 1e-6)  # [N_ego,C]
            agg_prob = agg_sem / (agg_sem.sum(dim=-1, keepdim=True) + 1e-6)  # [N_ego,C]

            ego_conf = ego_prob.max(dim=-1, keepdim=True)[0]  # scalar in [0,1]
            agg_conf = agg_prob.max(dim=-1, keepdim=True)[0]

            alpha = ego_conf / (ego_conf + agg_conf + 1e-6)  # [N_ego,1]
            alpha_used = alpha.detach()

            # Blend in the SAME space as the inputs used elsewhere (your softplus/unnormalized space)
            refine_sem[has_neighbor] = alpha_used[has_neighbor] * ego_semantic[has_neighbor] + \
                                       (1.0 - alpha_used[has_neighbor]) * agg_sem[has_neighbor]
        else:
            raise ValueError(f"Unknown semantic strategy: {self.semantic_strategy}")

        return GaussianPrediction(
            means=refine_mean[None],
            scales=refine_scale[None],
            rotations=refine_rot[None],
            opacities=refine_opa[None],
            semantics=refine_sem[None],
        )


def transform_neighbor_gaussians(
    gaussian_pred,
    pairwise_t_matrix,
    record_len,
    roi_bounds,
    opacity_thresh=0.01,
):
    """
    Fuse ego Gaussians with shared Gaussians after transforming them to ego frame and filtering by ROI.
    Skip fusion if no valid shared neighbors.

    Inputs:
    - gaussian_pred: GaussianPrediction with batch=1
    - pairwise_t_matrix: transforms from other agents to ego frame (batch=1)
    - record_len: number of CAVs in batch=1
    - roi_bounds: (x_min, y_min, z_min, x_max, y_max, z_max)
    """

    b = 0
    n_cav = record_len[b]
    pairwise_t_matrix = pairwise_t_matrix.to(torch.float32)

    device = gaussian_pred.means.device

    # Extract ego Gaussians (agent 0)
    ego_mean = gaussian_pred.means[0]
    ego_scale = gaussian_pred.scales[0]
    ego_rot = gaussian_pred.rotations[0]
    assert torch.allclose(ego_rot.norm(dim=-1), torch.ones_like(ego_rot[..., 0]), atol=1e-4), \
        "ego_rot is not normalized"
    ego_opacity = gaussian_pred.opacities[0]
    ego_semantic = gaussian_pred.semantics[0]

    # Collect all shared Gaussians transformed and filtered by ROI
    shared_means_list = []
    shared_scales_list = []
    shared_rotations_list = []
    shared_opacities_list = []
    shared_semantics_list = []

    num_of_gaussian_list = []

    # means_f = gaussian_pred.means.reshape(-1, gaussian_pred.means.shape[-1])
    # scales_f = gaussian_pred.scales.reshape(-1, gaussian_pred.scales.shape[-1])
    # rots_f = gaussian_pred.rotations.reshape(-1, gaussian_pred.rotations.shape[-1])
    #
    # opas = gaussian_pred.opacities
    # opas = opas if opas.ndim == 3 else opas.unsqueeze(-1)  # ensure (..., 1)
    # opas_f = opas.reshape(-1, opas.shape[-1])
    #
    # sems_f = gaussian_pred.semantics.reshape(-1, gaussian_pred.semantics.shape[-1])
    # # FOR VISUALIZATION
    # return GaussianPrediction(
    #     means=means_f.unsqueeze(0),
    #     scales=scales_f.unsqueeze(0),
    #     rotations=rots_f.unsqueeze(0),
    #     opacities=opas_f.unsqueeze(0),
    #     semantics=sems_f.unsqueeze(0),
    # ), num_of_gaussian_list

    for j in range(1, n_cav):
        pairwise_t = pairwise_t_matrix[b, j, 0]

        mean = gaussian_pred.means[j]
        scale = gaussian_pred.scales[j]
        rot = gaussian_pred.rotations[j]
        assert torch.allclose(rot.norm(dim=-1), torch.ones_like(rot[..., 0]), atol=1e-4), \
            "rot is not normalized"
        opacity = gaussian_pred.opacities[j]
        semantic = gaussian_pred.semantics[j]

        # Transform mean to ego frame
        ones = torch.ones(mean.shape[0], 1, device=device)
        mean_hom = torch.cat([mean, ones], dim=-1)  # (25600,4)
        mean_ego = (pairwise_t @ mean_hom.T).T[:, :3]

        R_mat = pairwise_t[:3, :3]
        identity = torch.eye(3, device=R_mat.device)
        assert torch.allclose(R_mat.T @ R_mat, identity, atol=1e-3)
        assert torch.allclose(torch.det(R_mat), torch.tensor(1.0, device=R_mat.device), atol=1e-3)
        q_rot = get_quaternion_from_rotation(R_mat).unsqueeze(0).expand(rot.shape[0], -1)
        rot_ego = quaternion_multiply(q_rot, rot)

        # ROI filter
        x_min, y_min, z_min, x_max, y_max, z_max = roi_bounds
        inside_roi = (
            (mean_ego[:, 0] > x_min) & (mean_ego[:, 0] < x_max - 1e-4) &
            (mean_ego[:, 1] > y_min) & (mean_ego[:, 1] < y_max - 1e-4) &
            (mean_ego[:, 2] > z_min) & (mean_ego[:, 2] < z_max - 1e-4)
        )

        mask_opacity = opacity[:, 0] >= opacity_thresh

        valid_mask = inside_roi & mask_opacity

        if valid_mask.sum() == 0:
            continue  # no Gaussians in ROI for this agent, skip

        shared_means_list.append(mean_ego[valid_mask])
        shared_scales_list.append(scale[valid_mask])
        shared_rotations_list.append(rot_ego[valid_mask])
        shared_opacities_list.append(opacity[valid_mask])
        shared_semantics_list.append(semantic[valid_mask])

        num_of_gaussian_list.append(valid_mask.sum())

    if len(shared_means_list) == 0:
        # No shared Gaussians in ROI from any agent
        # Return original ego Gaussians unchanged
        return GaussianPrediction(
            means=ego_mean.unsqueeze(0),
            scales=ego_scale.unsqueeze(0),
            rotations=ego_rot.unsqueeze(0),
            opacities=ego_opacity.unsqueeze(0),
            semantics=ego_semantic.unsqueeze(0),
        ), []

    # Concatenate all shared Gaussians
    shared_mean = torch.cat(shared_means_list, dim=0)
    shared_scale = torch.cat(shared_scales_list, dim=0)
    shared_rot = torch.cat(shared_rotations_list, dim=0)
    # print(shared_rot.norm(dim=-1))
    shared_rot = F.normalize(shared_rot, 2, -1)
    shared_opacity = torch.cat(shared_opacities_list, dim=0)
    shared_semantic = torch.cat(shared_semantics_list, dim=0)

    assert torch.allclose(shared_rot.norm(dim=-1), torch.ones_like(shared_rot[..., 0]), atol=1e-4), \
        "shared_rot is not normalized after transformation"

    # NAIVE MERGE
    return GaussianPrediction(
        means=torch.cat([ego_mean, shared_mean], dim=0).unsqueeze(0),
        scales=torch.cat([ego_scale, shared_scale], dim=0).unsqueeze(0),
        rotations=torch.cat([ego_rot, shared_rot], dim=0).unsqueeze(0),
        opacities=torch.cat([ego_opacity, shared_opacity], dim=0).unsqueeze(0),
        semantics=torch.cat([ego_semantic, shared_semantic], dim=0).unsqueeze(0),
    ), num_of_gaussian_list
