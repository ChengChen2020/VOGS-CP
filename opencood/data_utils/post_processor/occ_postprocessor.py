import torch
import numpy as np
import numba as nb

from opencood.models.gaussian_modules.gaussian_utils import get_meshgrid
from opencood.data_utils.post_processor.base_postprocessor import BasePostprocessor

class OccPostprocessor(BasePostprocessor):
    def __init__(self,
        anchor_params,
        train,
    ):
        self.params = anchor_params
        self.train = train
        self.cav_lidar_range = self.params['gt_range']
        self.min_bound = np.array(self.cav_lidar_range[:3])
        self.max_bound = np.array(self.cav_lidar_range[3:])
        self.grid_size = np.array(self.params['grid_size'])
        self.intervals = (self.max_bound - self.min_bound) / self.grid_size

    def generate_anchor_box(self):
        return None

    def generate_label(self, **kwargs):

        # -----------------------------
        # Get 3D center coordinates of each voxel
        # -----------------------------
        occ_xyz = get_meshgrid(self.cav_lidar_range, self.grid_size, self.intervals)

        if kwargs['voxel_label_20'] is not None:
            if isinstance(kwargs['voxel_label_20'], str):
                # construct from raw
                semantic_xyz = kwargs["semantic_xyz"]
                semantic_class = kwargs["semantic_class"]

                cav_label, cav_grid_ind = point_cut(
                    semantic_class, semantic_xyz, self.min_bound, self.max_bound, self.intervals,
                    self.grid_size
                )

                voxel_label = np.ones(self.grid_size, dtype=np.uint8) * 13 # empty

                # Merge voxel indices and labels
                voxel_pair = np.concatenate([cav_grid_ind, cav_label[:, None]], axis=1)
                voxel_pair = voxel_pair[np.lexsort((cav_grid_ind[:, 0], cav_grid_ind[:, 1], cav_grid_ind[:, 2]))]

                # Apply fast label assignment to voxel grid
                voxel_label = nb_process_label(np.copy(voxel_label), voxel_pair)

                # np.save(kwargs['voxel_label_20'], voxel_label)
                return {
                    "occ_label": voxel_label,
                    "occ_xyz": occ_xyz,
                    "occ_cam_mask": voxel_label != 0
                }
            else:
                assert False
                voxel_label = kwargs['voxel_label_20']
                return {
                    "occ_label": voxel_label,
                    "occ_xyz": occ_xyz,
                    "occ_cam_mask": voxel_label != 0
                }

        return {}


    def post_process(self, data_dict, output_dict, **kwargs):
        return {
            'final_occ': output_dict['ego']['final_occ'],
            'neighbor_gaussians': output_dict['ego'].get('neighbor_gaussians', []),
            'gaussian': output_dict['ego']['gaussian'],
            'gaussians': output_dict['ego']['gaussians'],
            'anchor_init': output_dict['ego']['anchor_init'],
        }, None


    def generate_gt(self, data_dict, **kwargs):
        return {
            'sampled_label': data_dict['ego']['label_dict']['occ_label'].flatten(1),
            'occ_mask': data_dict['ego']['label_dict']['occ_cam_mask'],
        }


    def collate_batch(self, label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            List of dictionaries containing occupancy-related labels for each frame.

        Returns
        -------
        target_batch : dict
            Reformatted labels as torch tensors.
        """
        occ_xyz = []
        occ_label = []
        occ_cam_mask = []

        for label_dict in label_batch_list:
            occ_xyz.append(torch.tensor(label_dict["occ_xyz"]))  # (X, Y, Z, 3)
            occ_label.append(torch.tensor(label_dict["occ_label"]))  # (X, Y, Z)
            occ_cam_mask.append(torch.tensor(label_dict["occ_cam_mask"]))  # (X, Y, Z)

        if occ_xyz:
            occ_xyz = torch.stack(occ_xyz, dim=0)  # (B, X, Y, Z, 3)
        else:
            occ_xyz = torch.empty(0)

        if occ_label:
            occ_label = torch.stack(occ_label, dim=0)  # (B, X, Y, Z)
        else:
            occ_label = torch.empty(0)

        if occ_cam_mask:
            occ_cam_mask = torch.stack(occ_cam_mask, dim=0)  # (B, X, Y, Z)
        else:
            occ_cam_mask = torch.empty(0)

        return {
            "occ_label": occ_label,  # torch.Tensor, (B, X, Y, Z)
            "occ_xyz": occ_xyz,  # torch.Tensor, (B, X, Y, Z, 3)
            "occ_cam_mask": occ_cam_mask  # torch.Tensor, (B, X, Y, Z)
        }


def point_cut(lidar_label, lidar_xyz, min_bound, max_bound, intervals, cur_grid_size):
    """
    Filter points that lie within the specified 3D bounding box and compute their voxel indices.

    Args:
        lidar_label (np.ndarray): Labels for each point.
        lidar_xyz (np.ndarray): Coordinates of each point.
        min_bound (np.ndarray): Lower bound of 3D grid (x, y, z).
        max_bound (np.ndarray): Upper bound of 3D grid (x, y, z).
        intervals (np.ndarray): Size of each voxel cell.
        cur_grid_size (np.ndarray): Total number of voxels in each dimension.

    Returns:
        tuple: Filtered labels and voxel grid indices.
    """
    mask = np.all((lidar_xyz >= min_bound) & (lidar_xyz <= max_bound), axis=1)
    lidar_xyz = lidar_xyz[mask]
    lidar_label = lidar_label[mask]
    grid_ind = np.floor((lidar_xyz - min_bound) / intervals).astype(int)
    grid_ind = np.clip(grid_ind, 0, cur_grid_size - 1)
    return lidar_label, grid_ind


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    """
    Assigns voxel labels by majority vote using sorted (voxel index, class) pairs.

    Args:
        processed_label (np.ndarray): (H, W, D) initialized with unknown label.
        sorted_label_voxel_pair (np.ndarray): (N, 4) array: (x, y, z, class_id).

    Returns:
        np.ndarray: Updated voxel label grid.
    """
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    # Initialize first voxel
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_voxel = sorted_label_voxel_pair[0, :3]

    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        class_id = sorted_label_voxel_pair[i, 3]
        if not np.all(cur_ind == cur_voxel):
            # Assign label by majority vote
            processed_label[cur_voxel[0], cur_voxel[1], cur_voxel[2]] = np.argmax(counter)
            counter.fill(0)
            cur_voxel = cur_ind

        counter[class_id] += 1

    processed_label[cur_voxel[0], cur_voxel[1], cur_voxel[2]] = np.argmax(counter)
    return processed_label
