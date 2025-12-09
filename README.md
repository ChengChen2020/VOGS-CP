# VOGS-CP

[AAAI 2026 Oral] Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction

[Paper](https://arxiv.org/abs/2508.10936) | [Project page](https://chengchen2020.github.io/VOGS-CP/)

## Prepare Datasets

- OPV2V: Download from [HERE](https://mobility-lab.seas.ucla.edu/opv2v/).
- Semantic-OPV2V: Please refer to [CoHFF](https://github.com/rruisong/CoHFF) `4LidarSurround` version.

Organize as follows:
```
VOGS-CP/dataset

. 
├── OPV2V
│   ├── surround
│   ├── test
│   ├── train
│   └── validate
```

or
`ln -s /path/to/opv2v dataset/OPV2V`

## Prepare Environments

- OpenCOOD: Please refer to [HEAL](https://github.com/yifanlu0227/HEAL).

Additionally,

```bash
pip install openmim
mim install mmcv==2.1.0
mim install mmdet==3.3.0
mim install mmsegmentation==1.2.2

# deformable attention & gaussian-to-voxel splatting
(cd opencood/models/gaussian_modules/ops && pip install -e .)
(cd opencood/models/gaussian_modules/localagg && pip install -e .)
```

## Quick Start

Please refer to [HEAL](https://github.com/yifanlu0227/HEAL) to get familiar with Basic Train / Test Command.

### Example Commands

Download the pretrained weights for the image backbone [HERE](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth) and put it inside `ckpts`.

```bash
# Single-agent training
mkdir opencood/logs/m4_single_25600
cp opencood/hypes_yaml/opv2v/gaussian/m4_single_25600/config.yaml opencood/logs/m4_single_25600
python opencood/tools/train.py -y None --model_dir opencood/logs/m4_single_25600

# Collaborative training
mkdir opencood/logs/m4_collab_25600_0.4
cp opencood/hypes_yaml/opv2v/gaussian/m4_collab_25600_0.4/config.yaml opencood/logs/m4_collab_25600_0.4
cp opencood/logs/m4_single_25600/your/bestval/checkpoint opencood/logs/m4_collab_25600_0.4/net_epoch1.pth
python opencood/tools/train.py -y None --model_dir opencood/logs/m4_collab_25600_0.4

# Collaborative inference
python opencood/tools/inference_heter_task.py --model_dir opencood/logs/m4_collab_25600_0.4 --task occupancy --range 20,20
```

## Acknowledgements

Our implementation benefits from a lot of awesome previous works, such as: [STAMP](https://github.com/taco-group/STAMP), [HEAL](https://github.com/yifanlu0227/HEAL), [GaussianFormer](https://github.com/huang-yh/GaussianFormer), [CoHFF](https://github.com/rruisong/CoHFF).

## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{chen2025vision,
  title={Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction},
  author={Chen, Cheng and Huang, Hao and Bagchi, Saurabh},
  journal={arXiv preprint arXiv:2508.10936},
  year={2025}
}
```
