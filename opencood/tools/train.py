# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License


import random
import torch

# seeding
random.seed(3)
torch.manual_seed(3)
torch.cuda.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# from fvcore.nn import FlopCountAnalysis
# from fvcore.nn import flop_count_table


import argparse
import os
import statistics


from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.misc.checkpoint_util import refine_load_from_sd

from icecream import ic


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True, help="data generation yaml file needed ")
    parser.add_argument("--model_dir", default="", help="Continued training path")
    parser.add_argument("--fusion_method", "-f", default="intermediate", help="passed to inference.")
    parser.add_argument("--flop_count", action="store_true")
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print("Dataset Building")
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    train_loader = DataLoader(
        opencood_train_dataset,
        batch_size=hypes["train_params"]["batch_size"],
        num_workers=24,
        collate_fn=opencood_train_dataset.collate_batch_train,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        opencood_validate_dataset,
        batch_size=hypes["train_params"]["batch_size"],
        num_workers=24,
        collate_fn=opencood_train_dataset.collate_batch_train,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
    )

    print("Creating Model")
    model = train_utils.create_model(hypes)
    if hasattr(model, 'init_weights') and callable(model.init_weights):
        model.init_weights()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(
            hypes, optimizer, init_epoch=init_epoch, n_iter_per_epoch=len(train_loader)
        )
        print(f"resume from {init_epoch} epoch.")

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, n_iter_per_epoch=len(train_loader))

    ## Load pretrained model
    for m_key in ['m1', 'm2', 'm3', 'm4']:
        load_path = hypes['model']['args'].get(m_key, {}).get('load_from', None)
        if load_path is not None:
            print(f"Loading pretrained weights for {m_key} from {load_path}")
            ckpt = torch.load(load_path, map_location='cpu')
            state_dict = ckpt['state_dict']
            # print("Checkpoint keys:", state_dict.keys())

            backbone_attr = f'backbone_{m_key}'
            try:
                load_result = getattr(model, backbone_attr).load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Refining state_dict for {m_key} due to error: {e}")
                refined_state_dict = refine_load_from_sd(state_dict)
                load_result = getattr(model, backbone_attr).load_state_dict(refined_state_dict, strict=False)

            print(f"{m_key} load result:", load_result)
            print(f"Pretrained Backbone {m_key} Loaded.")

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")

    # record training
    writer = SummaryWriter(saved_path)

    print("Training start")
    epoches = hypes["train_params"]["epoches"]
    grad_accumulation = hypes["train_params"].get("grad_accumulation", 1)
    grad_max_norm = hypes["train_params"].get("grad_max_norm", float('inf'))
    global_iter = 0
    supervise_single_flag = (
        False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    )
    # used to help schedule learning rate

    for epoch in [0] if opt.flop_count else range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print("learning rate %f" % param_group["lr"])
        # the model will be evaluation mode during validation
        model.train()
        try:  # heter_model stage2
            model.model_train_init()
        except:
            print("No model_train_init function")
        total_flops_sum = 0
        grad_flops_sum = 0
        for i, batch_data in enumerate(train_loader):
            # if batch_data is None or (
            #     "object_bbx_mask" in batch_data["ego"] and batch_data["ego"]["object_bbx_mask"].sum() == 0
            # ):
            #     continue
            model.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)
            batch_data["ego"]["epoch"] = epoch

            if opt.flop_count:
                flops = FlopCountAnalysis(model, batch_data["ego"])
                flops_counter = flops.by_module()
                for key, value in flops_counter.items():

                    if getattr(model, key, None) is not None:
                        requires_grad = False
                        for param in getattr(model, key).parameters():
                            if param.requires_grad == True:
                                requires_grad = True
                        if requires_grad:
                            grad_flops_sum += value
                        total_flops_sum += value
                print(i, "/", len(train_loader), "Grad FLOPS: ", grad_flops_sum)
                print(i, "/", len(train_loader), "Total FLOPS: ", total_flops_sum)
                del flops
                torch.cuda.empty_cache()
                continue

                # flop_count_table(flops)
            ouput_dict = model(batch_data["ego"])
            final_loss = criterion(ouput_dict, batch_data["ego"]["label_dict"])
            # if i % 100 == 0:
            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data["ego"]["label_dict_single"], suffix="_single") * hypes[
                    "train_params"
                ].get("single_weight", 1)
                # if i % 100 == 0:
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            # back-propagation
            loss = final_loss / grad_accumulation
            loss.backward()


            if (global_iter + 1) % grad_accumulation == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)
                optimizer.step()
                optimizer.zero_grad()

                lr = max([p['lr'] for p in optimizer.param_groups])
                print(f"[epoch {epoch}][{i + 1}/{len(train_loader)}] || lr: {lr:.7f} || grad_norm: {grad_norm:.4f}")

            global_iter += 1

            if hasattr(scheduler, 'step_update'):
                scheduler.step_update(global_iter)

        if opt.flop_count:
            print("Total FLOPS: ", total_flops)
            return

            # torch.cuda.empty_cache()  # it will destroy memory buffer

        if epoch % hypes["train_params"]["save_freq"] == 0:
            torch.save(model.state_dict(), os.path.join(saved_path, "net_epoch%d.pth" % (epoch + 1)))

        if epoch % hypes["train_params"]["eval_freq"] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data["ego"]["epoch"] = epoch
                    ouput_dict = model(batch_data["ego"])

                    final_loss = criterion(ouput_dict, batch_data["ego"]["label_dict"])
                    # print(f'val loss {final_loss:.3f}')
                    criterion.logging(epoch, i, len(val_loader), writer)
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print("At epoch %d, the validation loss is %f" % (epoch, valid_ave_loss))
            writer.add_scalar("Validate_Loss", valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(), os.path.join(saved_path, "net_epoch_bestval_at%d.pth" % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(
                    os.path.join(saved_path, "net_epoch_bestval_at%d.pth" % (lowest_val_epoch))
                ):
                    os.remove(os.path.join(saved_path, "net_epoch_bestval_at%d.pth" % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        if not hasattr(scheduler, 'step_update'):
            scheduler.step()

        opencood_train_dataset.reinitialize()

    print("Training Finished, checkpoints saved to %s" % saved_path)
    if opt.flop_count:
        print("Total FLOPS: ", total_flops)
        return

    # run_test = True
    # if run_test:
    #     fusion_method = opt.fusion_method
    #     cmd = f"python opencood/tools/inference_heter_task.py --model_dir {saved_path} --fusion_method {fusion_method}"
    #     print(f"Running command: {cmd}")
    #     os.system(cmd)


if __name__ == "__main__":
    main()
