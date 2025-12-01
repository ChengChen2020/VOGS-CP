import torch, torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np

from .lovasz_softmax import lovasz_softmax

# Surround train
nusc_class_frequencies = np.array([
    8921462,
    4077636,
    3336744,
    27356228,
    657075,
    89139632,
    22676500,
    9267481,
    13632227,
    4691101,
    3470079,
    57787,
    73495,
    # 1503762553
])
# [0.9622, 1.0117, 1.0253, 0.8993, 1.1496, 0.8413, 0.9092, 0.9600, 0.9374, 1.0025, 1.0226, 1.4045, 1.3744, 0.5]

class OccupancyLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.loss_func = self.loss_voxel

        self.empty_label = args.get('empty_label', 13)
        self.num_classes = args.get('num_classes', 14)
        self.classes = list(range(self.num_classes))
        self.use_sem_geo_scal_loss = args.get('use_sem_geo_scal_loss', False)
        self.use_lovasz_loss = args.get('use_lovasz_loss', True)
        self.lovasz_ignore = args.get('lovasz_ignore', 13)
        self.ignore_empty = args.get('ignore_empty', False)
        self.lovasz_use_softmax = args.get('lovasz_use_softmax', False)

        balance_cls_weight = args.get('balance_cls_weight', False)
        manual_class_weight = args.get('manual_class_weight', None)
        self.loss_voxel_ce_weight = args['multi_loss_weights'].get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = args['multi_loss_weights'].get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = args['multi_loss_weights'].get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = args['multi_loss_weights'].get('loss_voxel_lovasz_weight', 1.0)

        if balance_cls_weight:
            if manual_class_weight is not None:
                self.class_weights = torch.tensor(manual_class_weight)
            else:
                class_freqs = nusc_class_frequencies
                self.class_weights = torch.from_numpy(1 / np.log(class_freqs[:self.num_classes] + 0.001))
                raise ValueError("manual_class_weight must be provided when balance_cls_weight=True")
            self.class_weights = self.num_classes * F.normalize(self.class_weights, 1, -1)
        else:
            self.class_weights = torch.ones(self.num_classes)
            assert False, "Need balance_cls_weight=True"

        # self.loss_dict_global = {}
        self.use_focal_loss = args.get('use_focal_loss', False)
        if self.use_focal_loss:
            self.focal_loss = CustomFocalLoss(**focal_loss_args)

        self.use_dice_loss = args.get('use_dice_loss', False)
        if self.use_dice_loss:
            self.dice_loss = DiceLoss(
                class_weight=self.class_weights,
                loss_weight=2.0
            )

    def loss_voxel(self, pred_occ, sampled_xyz, sampled_label, occ_mask=None):
        tot_loss = 0.

        aggregated_loss_dict = {}

        if self.ignore_empty:
            empty_mask = sampled_label != self.empty_label
            occ_mask = empty_mask if occ_mask is None else empty_mask & occ_mask.flatten(1)

        if occ_mask is not None:
            occ_mask = occ_mask.flatten(1)
            sampled_label = sampled_label[occ_mask][None]

        for semantics in pred_occ:
            if occ_mask is not None:
                semantics = semantics.transpose(1, 2)[occ_mask][None].transpose(1, 2)

            loss_dict = {}

            if self.use_focal_loss:
                loss_dict['loss_voxel_ce'] = self.loss_voxel_ce_weight * self.focal_loss(
                    semantics, sampled_label, sampled_xyz,
                    self.class_weights.type_as(semantics),
                    ignore_index=255
                )
            else:
                if self.lovasz_use_softmax:
                    loss_dict['loss_voxel_ce'] = self.loss_voxel_ce_weight * CE_ssc_loss(
                        semantics, sampled_label,
                        self.class_weights.type_as(semantics),
                        ignore_index=255
                    )
                else:
                    loss_dict['loss_voxel_ce'] = self.loss_voxel_ce_weight * CE_wo_softmax(
                        semantics, sampled_label,
                        self.class_weights.type_as(semantics),
                        ignore_index=255
                    )

            if self.use_sem_geo_scal_loss:
                scal_input = torch.softmax(semantics, dim=1) if self.lovasz_use_softmax else semantics
                loss_dict['loss_voxel_sem_scal'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(
                    scal_input.clone(), sampled_label, ignore_index=255
                )
                loss_dict['loss_voxel_geo_scal'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(
                    scal_input.clone(), sampled_label, ignore_index=255, non_empty_idx=self.empty_label
                )

            if self.use_lovasz_loss:
                lovasz_input = torch.softmax(semantics, dim=1) if self.lovasz_use_softmax else semantics
                loss_dict['loss_voxel_lovasz'] = self.loss_voxel_lovasz_weight * lovasz_softmax(
                    lovasz_input.transpose(1, 2).flatten(0, 1), sampled_label.flatten(), ignore=self.lovasz_ignore
                )

            if self.use_dice_loss:
                loss_dict['loss_voxel_dice'] = self.dice_loss(semantics, sampled_label)

            loss = sum(loss_dict.values())
            tot_loss += loss

            # Accumulate component-wise loss
            for k, v in loss_dict.items():
                if k not in aggregated_loss_dict:
                    aggregated_loss_dict[k] = v.clone()
                else:
                    aggregated_loss_dict[k] += v

        # Average over number of predictions
        num_preds = len(pred_occ)
        avg_total_loss = tot_loss / num_preds
        avg_loss_dict = {k: v / num_preds for k, v in aggregated_loss_dict.items()}
        avg_loss_dict['total_loss'] = avg_total_loss

        return avg_total_loss, avg_loss_dict


    def forward(self, output_dict, gt_dict):

        loss, loss_dict = self.loss_func(
            output_dict['pred_occ'],
            output_dict['sampled_xyz'],
            output_dict['sampled_label'],
            output_dict['occ_mask'],
        )
        self.loss_dict = loss_dict
        return loss


    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print and log all keys in self.loss_dict dynamically.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        batch_id : int
            Index of the current batch.
        batch_len : int
            Number of total batches in this epoch.
        writer : SummaryWriter
            TensorBoard writer.
        pbar : tqdm.tqdm, optional
            Progress bar for CLI logging.
        """
        # Prepare formatted loss string
        log_items = [f"[epoch {epoch}][{batch_id + 1}/{batch_len}]"]
        for k, v in self.loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().item()
            log_items.append(f"{k}: {v:.4f}")

        log_str = " || ".join(log_items)

        if pbar is not None:
            pbar.set_description(log_str)
        else:
            print(log_str)


        # Write to TensorBoard
        if writer is not None:
            global_step = epoch * batch_len + batch_id
            for k, v in self.loss_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                writer.add_scalar(k, v, global_step)

def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    with autocast('cuda', enabled=False):
        loss = criterion(pred, target.long())

    return loss

