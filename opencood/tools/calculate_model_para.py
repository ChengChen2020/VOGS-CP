
import torch
import argparse
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from types import SimpleNamespace
import os

parser = argparse.ArgumentParser(description="synthetic data generation")
parser.add_argument('--model_dir', type=str,
                    default='OURS_4Agents_Heter_Simple/local_adapter/convnext_crop_w_output_wholemap/m1',
                    help='Continued training path')

# parser.add_argument('--folder_dir', type=str,
#                     default='efficiency_comparsion/OURS')
parser.add_argument('--folder_dir', type=str,
                    default='')

opt = parser.parse_args()

opt.model_dir = os.path.join("opencood/logs", opt.model_dir)
if opt.folder_dir:
    opt.folder_dir = os.path.join("opencood/logs", opt.folder_dir)
    # import pdb; pdb.set_trace()
    hypers_list = []
    for file in sorted(os.listdir(opt.folder_dir)):
        f_dir = os.path.join(opt.folder_dir, file)
        # import pdb; pdb.set_trace()
        if file.endswith('.yaml'):
            hypes = yaml_utils.load_yaml(None, config=f_dir)
            hypers_list.append((file, hypes))
else:
    hypers_list = [(opt.model_dir, yaml_utils.load_yaml(None, opt))]
        


# import pdb; pdb.set_trace()

# total_params = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
# encoder_params = sum(p.numel() * p.element_size() for p in model.encoder_m2.parameters()) / (1024 * 1024)
# backbone_params = sum(p.numel() * p.element_size() for p in model.backbone_m2.parameters()) / (1024 * 1024)
# aligner_m2 = sum(p.numel() * p.element_size() for p in model.aligner_m2.parameters()) / (1024 * 1024)
# adapter_m2 = sum(p.numel() * p.element_size() for p in model.adapter_m2.parameters()) / (1024 * 1024)
# reverter_m2 = sum(p.numel() * p.element_size() for p in model.reverter_m2.parameters()) / (1024 * 1024)
# # pyramid_backbone_m2 = sum(p.numel() * p.element_size() for p in model.pyramid_backbone_m2.parameters()) / (1024 * 1024)
# # shrink_conv_m2 = sum(p.numel() * p.element_size() for p in model.shrink_conv_m2.parameters()) / (1024 * 1024)
# # cls_head_m2 = sum(p.numel() * p.element_size() for p in model.cls_head_m2.parameters()) / (1024 * 1024)
# # reg_head_m2 = sum(p.numel() * p.element_size() for p in model.reg_head_m2.parameters()) / (1024 * 1024)
# # dir_head_m2 = sum(p.numel() * p.element_size() for p in model.dir_head_m2.parameters()) / (1024 * 1024)
# print(f"Total params: {total_params} M")
# print(f"Encoder params: {encoder_params} M")
# print(f"Backbone params: {backbone_params} M")
# print(f"Aligner params: {aligner_m2} M")
# print(f"Adapter params: {adapter_m2} M")
# print(f"Reverter params: {reverter_m2} M")
# # print(f"Pyramid Backbone params: {pyramid_backbone_m2} M")
# # print(f"Shrink Conv params: {shrink_conv_m2} M")
# # print(f"Cls Head params: {cls_head_m2} M")
# # print(f"Reg Head params: {reg_head_m2} M")
# # print(f"Dir Head params: {dir_head_m2} M")

# total_params = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

# torch.save(model.state_dict(), 'debug.pth')

# import pdb; pdb.set_trace()

for name, hypes in hypers_list:
    
    model = train_utils.create_model(hypes)

    encoder_keys = ['aligner', 'encoder', 'backbone']
    adaptation_keys = ['adapter', 'reverter']

    total_params_all = 0
    for modality_name in ['m0', 'm1', 'm2', 'm3', 'm4']:
        item_list = [item for item in model.__dir__() if (item[-2:] == modality_name) and isinstance(getattr(model, item), torch.nn.Module) ]
        total_params = 0
        encoder_total_params = 0
        adaptation_total_params = 0
        for item in item_list:
            try:
                item_params = sum(p.numel() * p.element_size() for p in getattr(model, item).parameters()) / (1024 * 1024)
            except:
                import pdb; pdb.set_trace()
            print(f"{item}\t{item_params:5f} M")
            total_params += item_params
            if item.split('_')[0] in encoder_keys:
                encoder_total_params += item_params
            if item.split('_')[0] in adaptation_keys:
                adaptation_total_params += item_params
        if total_params != 0:
            print(name)
            # print(modality_name)
            print(f"Total params: {total_params} M")
            print(f"Encoder params: {encoder_total_params} M")
            if adaptation_total_params != 0:
                print(f"Adaptation params: {adaptation_total_params} M")
            print()
        total_params_all += total_params
    # print(f"Total params: {total_params_all} M")
            
        
        