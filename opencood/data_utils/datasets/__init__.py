# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License

from opencood.data_utils.datasets.late_fusion_dataset import getLateFusionDataset
from opencood.data_utils.datasets.late_heter_fusion_dataset import getLateheterFusionDataset
from opencood.data_utils.datasets.early_fusion_dataset import getEarlyFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset import getIntermediateFusionDataset
from opencood.data_utils.datasets.intermediate_heter_fusion_dataset import getIntermediateheterFusionDataset
from opencood.data_utils.datasets.intermediate_heter_adapter_fusion_dataset import getIntermediateheteradapterFusionDataset

from opencood.data_utils.datasets.heter_infer.intermediate_heter_infer_fusion_dataset import getIntermediateheterinferFusionDataset

from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset
from opencood.data_utils.datasets.basedataset.v2xsim_basedataset import V2XSIMBaseDataset
from opencood.data_utils.datasets.basedataset.dairv2x_basedataset import DAIRV2XBaseDataset
from opencood.data_utils.datasets.basedataset.v2xset_basedataset import V2XSETBaseDataset
from opencood.data_utils.datasets.basedataset.v2v4real_basedataset import V2V4REALBaseDataset


from opencood.data_utils.datasets.basedataset.base_camera_dataset import BaseCameraDataset
from opencood.data_utils.datasets.basedataset.basedataset import BaseDataset

def build_dataset(dataset_cfg, visualize=False, train=True):
    fusion_name = dataset_cfg['fusion']['core_method']
    dataset_name = dataset_cfg['fusion']['dataset']
    assert fusion_name in ['late', 'lateheter', 'intermediate', 'intermediateheterinfer', 
                           'intermediate2stage', 'intermediateheter', 
                           'early', 'intermediatehetertask', 'intermediateheterseg',
                           'intermediateheteradapter']
    assert dataset_name in ['opv2v', 'v2xsim', 'dairv2x', 'v2xset', 'v2v4real']
    
    
    base_dataset_cls = dataset_name.upper() + "BaseDataset"
    base_dataset_cls = eval(base_dataset_cls)
    
    if fusion_name == 'intermediatehetertask':
        fusion_dataset_func = "getIntermediateheter" + fusion_name.capitalize() + "FusionDataset"
        assert dataset_cfg['fusion']['core_method'][fusion_name].get('sub_method', None), "sub_method is required for intermediatehetertask"
        fusion_dataset_sub_func_dict = dict()
        for sub_method in fusion_name['sub_method']:
            assert sub_method in ['dect', 'seg'], "sub_method should be either dect or seg"
            fusion_dataset_sub_func = "getIntermediateheter" + sub_method.capitalize() + "FusionDataset"
            fusion_dataset_sub_func = eval(fusion_dataset_sub_func)
            fusion_dataset_sub_func_dict[sub_method] = fusion_dataset_sub_func()(
                    params=dataset_cfg,
                    visualize=visualize,
                    train=train
                )
            
        fusion_dataset_func = eval(fusion_dataset_func)
        dataset = fusion_dataset_func(base_dataset_cls)(
            fusion_dataset_sub_func_dict
        )
    else:
        fusion_dataset_func = "get" + fusion_name.capitalize() + "FusionDataset"
        fusion_dataset_func = eval(fusion_dataset_func)
        dataset = fusion_dataset_func(base_dataset_cls)(
            params=dataset_cfg,
            visualize=visualize,
            train=train
        )

    return dataset
