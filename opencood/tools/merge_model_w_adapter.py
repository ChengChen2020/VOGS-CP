# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

import argparse
import os
from opencood.tools import train_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
import torch

parser = argparse.ArgumentParser(description="synthetic data generation")
parser.add_argument('--model_dir', type=str, required=True,
                    help='Continued training path. Example: opencood/logs/OURS_2Agents_SameArch_Lidar')
parser.add_argument('--sub_dir', type=str, default='local_adapter',
                    help='local_adapter, local_calibrator, etc.')
parser.add_argument('--adapter_dir', type=str, required=True,
                    help='Examples: "conv", "fc", "convnext"')
parser.add_argument('--with_protocol', action='store_true', help='Use protocol modality')
opt = parser.parse_args()


infer_path = os.path.join(opt.model_dir, 'final_infer', opt.adapter_dir)
model_config = os.path.join(infer_path, 'config.yaml')
hypes = yaml_utils.load_yaml(model_config)
model = train_utils.create_model(hypes)

adapter_path = os.path.join(opt.model_dir, opt.sub_dir, opt.adapter_dir)
agent_path = os.path.join(opt.model_dir, 'local')
for agent in os.listdir(adapter_path):
    local_agent_path = os.path.join(agent_path, agent)
    resume_epoch, model = train_utils.load_saved_model(local_agent_path, model)
for agent in os.listdir(adapter_path):
    adapter_agent_path = os.path.join(adapter_path, agent)
    resume_epoch, model = train_utils.load_saved_model(adapter_agent_path, model)
    # assert resume_epoch == 0, "No best model found in {}. Make sure model is trained".format(adapter_agent_path)

protocol_path = os.path.join(opt.model_dir, 'protocol')
if opt.with_protocol:
    resume_epoch, model = train_utils.load_saved_model(protocol_path, model)

torch.save(model.state_dict(),
           os.path.join(infer_path, 'net_epoch1.pth'))
