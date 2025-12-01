# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbog@umich.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
from collections import OrderedDict
import cv2
import h5py
import cv2
from PIL import Image
import json
from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset
from opencood.utils.common_utils import read_json
from collections import defaultdict, OrderedDict
import opencood.utils.pcd_utils as pcd_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.common_utils import read_json

# All the same as OPV2V
class V2V4REALBaseDataset(OPV2VBaseDataset):
    def __init__(self, params, visulize, train=True):

        super().__init__(params, visulize, train)
        self.modality_assignment = (
            defaultdict(lambda: {
                '0': "m1",
                '1': "m2",
            })
            if ("assignment_path" not in params["heter"] or params["heter"]["assignment_path"] is None)
            else read_json(params["heter"]["assignment_path"])
        )


    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']
            
            # load param file: json is faster than yaml
            json_file = cav_content[timestamp_key]['yaml'].replace("yaml", "json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data[cav_id]['params'] = json.load(f)
            else:
                data[cav_id]['params'] = \
                    load_yaml(cav_content[timestamp_key]['yaml'])
                    
            for key in data[cav_id]['params']['vehicles'].keys():
                data[cav_id]['params']['vehicles'][key]['location'] = \
                    data[cav_id]['params']['vehicles'][key]['location'] + data[cav_id]['params']['lidar_pose'][:3]
            # load camera file: hdf5 is faster than png
            hdf5_file = cav_content[timestamp_key]['cameras'][0].replace("camera0.png", "imgs.hdf5")

            if self.use_hdf5 and os.path.exists(hdf5_file):
                with h5py.File(hdf5_file, "r") as f:
                    data[cav_id]['camera_data'] = []
                    data[cav_id]['depth_data'] = []
                    for i in range(4):
                        if self.load_camera_file:
                            data[cav_id]['camera_data'].append(Image.fromarray(f[f'camera{i}'][()]))
                        if self.load_depth_file:
                            data[cav_id]['depth_data'].append(Image.fromarray(f[f'depth{i}'][()]))
            else:
                if self.load_camera_file:
                    data[cav_id]['camera_data'] = \
                        load_camera_data(cav_content[timestamp_key]['cameras'])
                if self.load_depth_file:
                    data[cav_id]['depth_data'] = \
                        load_camera_data(cav_content[timestamp_key]['depths']) 

            # load lidar file
            if self.load_lidar_file or self.visualize:
                data[cav_id]['lidar_np'] = \
                    pcd_utils.pcd_to_np(cav_content[timestamp_key]['lidar'])

            if getattr(self, "heterogeneous", False):
                data[cav_id]['modality_name'] = cav_content[timestamp_key]['modality_name']

            for file_extension in self.add_data_extension:
                # if not find in the current directory
                # go to additional folder
                if not os.path.exists(cav_content[timestamp_key][file_extension]):
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("train","additional/train")
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("validate","additional/validate")
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("test","additional/test")
                    
                if '.yaml' in file_extension:
                    data[cav_id][file_extension] = \
                        load_yaml(cav_content[timestamp_key][file_extension])
                else:
                    data[cav_id][file_extension] = \
                        cv2.imread(cav_content[timestamp_key][file_extension])

        return data

