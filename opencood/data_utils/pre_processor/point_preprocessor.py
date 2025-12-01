# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

# -*- coding: utf-8 -*-

import numpy as np

from opencood.utils import pcd_utils
from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor
import torch

class PointPreprocessor(BasePreprocessor):
    """
    Lidar Point pre-processor.

    Parameters
    ----------
    preprocess_params : dict
        The dictionary containing all parameters of the preprocessing.

    train : bool
        Train or test mode.
    """

    def __init__(self, preprocess_params, train):
        super(PointPreprocessor, self).__init__(preprocess_params,
                                            train)
        self.lidar_range = self.params['cav_lidar_range']
        self.voxel_size = self.params['args']['voxel_size']

        grid_size = (np.array(self.lidar_range[3:6]) -
                     np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

    def preprocess(self, pcd_np):
        """
        Preprocess the lidar points by simple sampling.

        Parameters
        ----------
        pcd_np : np.ndarray
            The raw lidar.

        Returns
        -------
        data_dict : the output dictionary.
        """
        # data_dict = {}
        # sample_num = self.params['args']['sample_num']

        # pcd_np = pcd_utils.downsample_lidar(pcd_np, sample_num)
        # data_dict['downsample_lidar'] = pcd_np
        
        normalized_coords = (pcd_np[:, :3] - np.array(self.lidar_range[:3])) / np.array(self.voxel_size)
        
        data_dict = {
            'feat': torch.from_numpy(pcd_np[:, 3:]),
            "coord": torch.from_numpy(normalized_coords),
        }

        return data_dict
    
    def collate_batch(self, batch):
        """
        Collate the batch data.
        """
        
        coords = []
        feats = []
        batch_idx = []
        
        for i in range(len(batch['feat'])):
            coords.append(batch['coord'][i])
            feats.append(batch['feat'][i])
            batch_idx.extend([i] * batch['coord'][i].shape[0])
            
        batch_idx = torch.tensor(batch_idx)
        coords = torch.concat(coords, axis=0)
        feats = torch.concat(feats, axis=0)
        
        return {
            'coord': coords,
            'feat': feats,
            'grid_size': torch.from_numpy(self.grid_size),
            'batch': batch_idx,
            'lidar_range': torch.tensor(self.lidar_range),
            'voxel_size': torch.tensor(self.voxel_size)
        }