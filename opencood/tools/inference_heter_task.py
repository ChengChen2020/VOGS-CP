# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License

import argparse
import os
import json
import pickle
import statistics
import time
from typing import OrderedDict
import importlib
import torch
import torchvision
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
from opencood.utils.seg_iou import mean_IU
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd

torch.multiprocessing.set_sharing_strategy("file_system")

def to_numpy_cpu(x):
    """Return a NumPy array on CPU, detaching if x is a torch tensor."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x)
    return np.ascontiguousarray(x)

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--model_dir", type=str, required=True, help="Continued training path")
    parser.add_argument(
        "--fusion_method", type=str, default="intermediate", help="no, no_w_uncertainty, late, early or intermediate"
    )
    parser.add_argument("--save_vis_interval", type=int, default=40, help="interval of saving visualization")
    parser.add_argument(
        "--save_npy", action="store_true", help="whether to save prediction and gt result" "in npy file"
    )
    parser.add_argument(
        "--range", type=str, default="102.4,102.4", help="detection range is [-102.4, +102.4, -102.4, +102.4]"
    )
    parser.add_argument("--no_score", action="store_true", help="whether print the score of prediction")
    parser.add_argument("--note", default="", type=str, help="any other thing?")
    parser.add_argument("--noise", type=float, default=0.0, help="add noise to pose")
    parser.add_argument("--all", action="store_true", help="evaluate all the agents instead of the first one.")
    parser.add_argument("--show_bev", action="store_true", help="Visualize the BEV feature")
    parser.add_argument(
        "--protocol_result", action="store_true", help="plot the protocol result instead of the ego result."
    )
    parser.add_argument("--data_only", action="store_true", help="Only visualize the data")
    parser.add_argument("--score_threshold", type=float, default=0.2, help="score threshold for visualization")
    parser.add_argument("--aggregation", default="", choices=["", "nms", "psa"], help="post process method")
    parser.add_argument("--task", default="detection", choices=["detection", "segmentation", "occupancy"], help="task type")

    opt = parser.parse_args()

    # if opt.protocol_result:
    #     # No need to plot BEV feature when plotting protocol result, the BEV feature is plotted in the ego mode.
    #     opt.show_bev = False
    #     return opt

    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ["late", "late_heter", "early", "intermediate", "no", "no_w_uncertainty", "single"]
    # if opt.all:
    #     assert not opt.show_bev

    hypes = yaml_utils.load_yaml(None, opt)

    hypes = update_dict(
        hypes,
        {
            "score_threshold": opt.score_threshold,
        },
    )

    if "heter" in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(",")[0]), eval(opt.range.split(",")[0])
        y_min, y_max = -eval(opt.range.split(",")[1]), eval(opt.range.split(",")[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes["cav_lidar_range"][2], x_max, y_max, hypes["cav_lidar_range"][5]]
        # replace all appearance
        hypes = update_dict(
            hypes, {"cav_lidar_range": new_cav_range, "lidar_range": new_cav_range, "gt_range": new_cav_range}
        )

        # reload anchor
        hypes = yaml_utils.update_yaml(hypes, opt)

    if opt.aggregation:
        hypes = update_dict(hypes, {"aggretation": opt.aggregation})

    hypes["validate_dir"] = hypes["test_dir"]
    # if "OPV2V" in hypes["test_dir"] or "v2xsim" in hypes["test_dir"]:
    #     assert "test" in hypes["validate_dir"]

    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    opt.left_hand = True if ("OPV2V" in hypes["test_dir"] or "V2XSET" in hypes["test_dir"]) else False

    print(f"Left hand visualizing: {opt.left_hand}")

    if "box_align" in hypes.keys():
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]

    print("Creating Model")
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Model from checkpoint")
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if opt.noise:
        # add noise to pose.
        pos_std = opt.noise
        rot_std = opt.noise
        pos_mean = 0
        rot_mean = 0

        # setting noise
        np.random.seed(303)
        noise_setting = OrderedDict()
        noise_args = {"pos_std": pos_std, "rot_std": rot_std, "pos_mean": pos_mean, "rot_mean": rot_mean}

        noise_setting["add_noise"] = True
        noise_setting["args"] = noise_args

        # build dataset for each noise setting
        print("Dataset Building")
        print(f"Noise Added: {pos_std}/{rot_std}/{pos_mean}/{rot_mean}.")
        hypes.update({"noise_setting": noise_setting})

    # build dataset for each noise setting
    print("Dataset Building")
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=24,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    modality_list = opencood_dataset.modality_name_list
    # Create the dictionary for evaluation
    if opt.all:
        result_stat = dict() # for detection
        ave_ious = dict() # for segmentation
        
        for modality_name in modality_list:
            assert 'task' in hypes['heter']['modality_setting'][modality_name]
            if hypes['heter']['modality_setting'][modality_name]['task'] == 'detection':
                result_stat[modality_name] = {
                    0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                    0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                    0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
                }
            elif hypes['heter']['modality_setting'][modality_name]['task'] == 'segmentation':
                ave_ious[modality_name] = {
                    'static_ave_iou': [],
                    'dynamic_ave_iou': [],
                    'lane_ave_iou': []
                }
            else:
                raise NotImplementedError("Only detection and segmentation task is supported.")
    else:
        if opt.task == "detection":
            result_stat = {
                0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
            }
        elif opt.task == "segmentation":
            ave_ious = {
                'static_ave_iou': [],
                'dynamic_ave_iou': [],
                'lane_ave_iou': []
            }
        elif opt.task == "occupancy":
            # from opencood.models.gaussian_modules.gaussian_vis import save_occ, save_gaussian, save_gaussian_topdown
            from opencood.misc.metric_util import MeanIoU
            miou_metric = MeanIoU(
                list(range(1, 13)), 13,
                ['Building', 'Fence', 'Terrain', 'Pole', 'Road', 'SideWalk',
                 'Vegetation', 'Vehicles', 'Wall', 'GuardRail', 'TrafficSign', 'Bridge'],
                True, 13, filter_minmax=False
            )
            miou_metric.reset()
            ave_ious = {
                'road_ave_iou': [],
                'vehicle_ave_iou': [],
                'other_ave_iou': []
            }
            ids = []
            lists_ = []
        else:
            raise NotImplementedError("Only detection and segmentation task is supported.")

    opt.infer_info = opt.fusion_method + opt.note + ("_all" if opt.all else "") + "_noise" + str(opt.noise)

    pbar = tqdm(enumerate(data_loader))
    for i, batch_data in pbar:
        pbar.set_description(f"{opt.infer_info}_{i}")
        if batch_data is None:
            continue

        if opt.data_only:
            os.makedirs(os.path.join(opt.model_dir, "data"), exist_ok=True)
            simple_vis.visualize(
                None,
                batch_data["ego"]["origin_lidar"][0],
                new_cav_range,
                os.path.join(opt.model_dir, "data", f"lidar_{i}.png"),
                method="bev",
                left_hand=opt.left_hand,
            )
            continue

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == "late":
                infer_result = inference_utils.inference_late_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == "early":
                infer_result = inference_utils.inference_early_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == "intermediate":
                infer_result = inference_utils.inference_intermediate_fusion(
                    batch_data,
                    model,
                    opencood_dataset,
                    infer_all=opt.all,
                    show_bev=opt.show_bev,
                    protocol_result=opt.protocol_result,
                )
            elif opt.fusion_method == "no":
                infer_result = inference_utils.inference_no_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == "no_w_uncertainty":
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data, model, opencood_dataset)
            elif opt.fusion_method == "single":
                infer_result = inference_utils.inference_no_fusion(batch_data, model, opencood_dataset, single_gt=True)
            elif opt.fusion_method == "late_heter":
                infer_result = inference_utils.inference_heter_late(
                    batch_data,
                    model,
                    opencood_dataset,
                    show_bev=opt.show_bev,
                    infer_all=opt.all,
                )
            else:
                raise NotImplementedError(
                    "Only single, no, no_w_uncertainty, early, late and intermediate" "fusion is supported."
                )

            agent_modality_list = batch_data["ego"]["agent_modality_list"] if opt.all else []
            if not opt.all:
                infer_result = [infer_result]

            for idx, infer_result_single in enumerate(infer_result):
                if opt.all:
                    work_dir = os.path.join(opt.model_dir, f"modality_{agent_modality_list[idx]}")
                    os.makedirs(work_dir, exist_ok=True)
                    if hypes['heter']['modality_setting'][agent_modality_list[idx]]['task'] == 'detection':
                        eval_detection_result(
                            opt,
                            agent_modality_list,
                            opencood_dataset,
                            infer_result_single,
                            result_stat,
                            batch_data,
                            idx,
                            work_dir,
                            hypes,
                            i,
                        )
                    elif hypes['heter']['modality_setting'][agent_modality_list[idx]]['task'] == "segmentation":
                        iou_static, iou_dynamic = eval_segmentation_result(opt, infer_result_single, idx, work_dir, i)
                        if iou_static is not None:
                            ave_ious[agent_modality_list[idx]]["static_ave_iou"].append(iou_static[1])
                            ave_ious[agent_modality_list[idx]]["lane_ave_iou"].append(iou_static[2])
                        if iou_dynamic is not None:
                            ave_ious[agent_modality_list[idx]]["dynamic_ave_iou"].append(iou_dynamic[1])
                        
                    else:
                        raise NotImplementedError("Only detection and segmentation task is supported.")
                else:
                    work_dir = opt.model_dir
                    if opt.task == 'detection':
                        eval_detection_result(
                            opt,
                            agent_modality_list,
                            opencood_dataset,
                            infer_result_single,
                            result_stat,
                            batch_data,
                            idx,
                            work_dir,
                            hypes,
                            i,
                        )
                    elif opt.task == "segmentation":
                        iou_static, iou_dynamic = eval_segmentation_result(opt, infer_result_single, idx, work_dir, i)
                        if iou_static is not None:
                            ave_ious["static_ave_iou"].append(iou_static[1])
                            ave_ious["lane_ave_iou"].append(iou_static[2])
                        if iou_dynamic is not None:
                            ave_ious["dynamic_ave_iou"].append(iou_dynamic[1])
                    elif opt.task == "occupancy":
                        pred_dict = infer_result_single["pred_box_tensor"]
                        gt_dict = infer_result_single["gt_box_tensor"]
                        if 'final_occ' in pred_dict and pred_dict['final_occ'] is not None:
                            for j, pred in enumerate(pred_dict['final_occ']):
                                pred_occ = pred
                                gt_occ = gt_dict['sampled_label'][j]
                                occ_mask = gt_dict['occ_mask'][j].flatten()
                                miou_metric._after_step(pred_occ, gt_occ, occ_mask)

                                origin = np.array([-20.0, -20.0, -2.3])
                                occshape = (100, 100, 8)

                                iou_road, iou_vehicle, iou_other = eval_occseg_result(pred_occ, gt_occ, occshape)
                                ave_ious["road_ave_iou"].append(iou_road)
                                ave_ious["vehicle_ave_iou"].append(iou_vehicle)
                                ave_ious["other_ave_iou"].append(iou_other)

                                ids.append(i)
                                lists_.append(pred_dict['neighbor_gaussians'])

                                vis_save_path_root = os.path.join(work_dir, f'vis_{opt.infer_info}')
                                os.makedirs(vis_save_path_root, exist_ok=True)

                                SHAPE = (100, 100, 8)

                                # save gt occ
                                gt_arr = to_numpy_cpu(gt_occ).reshape(SHAPE)
                                with open(os.path.join(vis_save_path_root, f"{i:05d}_gt_occ.pkl"), "wb") as f:
                                    pickle.dump(gt_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

                                # save pred occ
                                pred_arr = to_numpy_cpu(pred_occ).reshape(SHAPE)
                                with open(os.path.join(vis_save_path_root, f"{i:05d}_pred_occ.pkl"), "wb") as f:
                                    pickle.dump(pred_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

                                # save pred gaussian
                                torch.save(pred_dict['gaussian'], os.path.join(vis_save_path_root, f'{i:05d}_gaussian_attr.pth'))

                    else:
                        raise NotImplementedError("Only detection, segmentation and occupancy task is supported.")

        torch.cuda.empty_cache()
    if opt.all:
        # detection
        result_stat_all = {
            0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
        }
        for modality_name in result_stat:
            for iou in [0.3, 0.5, 0.7]:
                result_stat_all[iou]["tp"] += result_stat[modality_name][iou]["tp"]
                result_stat_all[iou]["fp"] += result_stat[modality_name][iou]["fp"]
                result_stat_all[iou]["gt"] += result_stat[modality_name][iou]["gt"]
                result_stat_all[iou]["score"] += result_stat[modality_name][iou]["score"]
            if result_stat[modality_name][iou]["tp"]:
                os.makedirs(f"{opt.model_dir}/{modality_name}", exist_ok=True)
                _, ap50, ap70 = eval_utils.eval_final_results(
                    result_stat[modality_name], f"{opt.model_dir}/{modality_name}", opt.infer_info
                )
        _, ap50, ap70 = eval_utils.eval_final_results(result_stat_all, opt.model_dir, opt.infer_info)
        
        # segmentation
        for modality in ave_ious:
            if not ave_ious[modality]["static_ave_iou"] or not ave_ious[modality]["dynamic_ave_iou"]:
                continue
            static_ave_iou = statistics.mean(ave_ious[modality]["static_ave_iou"])
            dynamic_ave_iou = statistics.mean(ave_ious[modality]["dynamic_ave_iou"])
            lane_ave_iou = statistics.mean(ave_ious[modality]["lane_ave_iou"])

            print(f"Modality: {modality}")
            print("Road IoU: %f" % static_ave_iou)
            print("Lane IoU: %f" % lane_ave_iou)
            print("Dynamic IoU: %f" % dynamic_ave_iou)
            if not os.path.exists(os.path.join(opt.model_dir, modality)):
                os.mkdir(os.path.join(opt.model_dir, modality))
                
            with open(os.path.join(opt.model_dir, modality, f"{opt.infer_info}_ave_iou.json"), "w") as f:
                json.dump(
                    {"static_ave_iou": static_ave_iou, "dynamic_ave_iou": dynamic_ave_iou, "lane_ave_iou": lane_ave_iou}, f)
    else:
        if opt.task == "detection":
            _, ap50, ap70 = eval_utils.eval_final_results(result_stat, opt.model_dir, opt.infer_info)
        elif opt.task == "segmentation":
            static_ave_iou = statistics.mean(static_ave_iou)
            dynamic_ave_iou = statistics.mean(dynamic_ave_iou)
            lane_ave_iou = statistics.mean(lane_ave_iou)

            print("Road IoU: %f" % static_ave_iou)
            print("Lane IoU: %f" % lane_ave_iou)
            print("Dynamic IoU: %f" % dynamic_ave_iou)
            with open(os.path.join(opt.model_dir, f"{opt.infer_info}_ave_iou.json"), "w") as f:
                json.dump(
                    {"static_ave_iou": static_ave_iou, "dynamic_ave_iou": dynamic_ave_iou, "lane_ave_iou": lane_ave_iou}, f
                )
        elif opt.task == "occupancy":
            vehicle_ave_iou = statistics.mean(ave_ious["vehicle_ave_iou"])
            road_ave_iou = statistics.mean(ave_ious["road_ave_iou"])
            other_ave_iou = statistics.mean(ave_ious["other_ave_iou"])
            print("Road IoU: %f" % road_ave_iou)
            print("Vehicle IoU: %f" % vehicle_ave_iou)
            print("Other IoU: %f" % other_ave_iou)

            miou, iou2, per_class_iou = miou_metric._after_epoch()
            print(f'mIoU: {miou}, iou2: {iou2}')
            # print('Current val loss is %.3f' % (np.mean(val_loss_list)))
            miou_metric.reset()
            # Dump to JSON
            save_path = os.path.join(opt.model_dir, f"{opt.infer_info}_ave_iou.json")
            with open(save_path, "w") as f:
                json.dump(
                    {
                        "mIoU": float(miou),
                        "iou2": float(iou2),
                        "per_class_iou": per_class_iou,
                        "vehicle_ave_iou": vehicle_ave_iou,
                        "road_ave_iou": road_ave_iou,
                        "other_ave_iou": other_ave_iou
                    },
                    f,
                    indent=2
                )

            # COMMUNICATION VOLUME
            df0 = pd.DataFrame({"sample_id": ids, "vals": lists_})

            # Record original length
            df0["length"] = df0["vals"].apply(len)

            # Pad/truncate to 7 and expand to columns pred_0..pred_6
            padded = df0["vals"].apply(lambda v: (list(map(float, v)) + [np.nan] * 7)[:7])
            pred_df = pd.DataFrame(padded.tolist(), columns=[f"pred_{i}" for i in range(7)])

            # Final DataFrame and CSV
            df = pd.concat([df0[["sample_id", "length"]], pred_df], axis=1)
            df.to_csv(os.path.join(opt.model_dir, f"{opt.infer_info}_inference.csv"), index=False)


        


def eval_detection_result(
    opt, agent_modality_list, opencood_dataset, infer_result_single, result_stat, batch_data, idx, work_dir, hypes, i
):

    pred_box_tensor = infer_result_single["pred_box_tensor"]
    gt_box_tensor = infer_result_single["gt_box_tensor"]
    pred_score = infer_result_single["pred_score"]
    if pred_box_tensor is None or gt_box_tensor is None or pred_score is None:
        return
    eval_utils.caluclate_tp_fp(
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        result_stat[agent_modality_list[idx]] if opt.all else result_stat,
        0.3,
    )
    eval_utils.caluclate_tp_fp(
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        result_stat[agent_modality_list[idx]] if opt.all else result_stat,
        0.5,
    )
    eval_utils.caluclate_tp_fp(
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        result_stat[agent_modality_list[idx]] if opt.all else result_stat,
        0.7,
    )
    if opt.save_npy:
        npy_save_path = os.path.join(work_dir, "npy")
        if not os.path.exists(npy_save_path):
            os.makedirs(npy_save_path)
        inference_utils.save_prediction_gt(
            pred_box_tensor, gt_box_tensor, batch_data["ego"]["origin_lidar"][0], i, npy_save_path
        )

    if not opt.no_score:
        infer_result_single.update({"score_tensor": pred_score})

    if getattr(opencood_dataset, "heterogeneous", False):
        cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
        infer_result_single.update({"cav_box_np": cav_box_np, "agent_modality_list": agent_modality_list})

    if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
        vis_save_path_root = os.path.join(work_dir, f'vis_{opt.infer_info}{"_protocol" if opt.protocol_result else ""}')
        if not os.path.exists(vis_save_path_root):
            os.makedirs(vis_save_path_root)

        # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
        # simple_vis.visualize(infer_result_single,
        #                     batch_data['ego'][
        #                         'origin_lidar'][0],
        #                     hypes['postprocess']['gt_range'],
        #                     vis_save_path,
        #                     method='3d',
        #                     left_hand=left_hand)
        vis_save_path = os.path.join(vis_save_path_root, "bev_%05d.png" % i)
        try:
            # new version considering various gt ranges
            gt_range = hypes["heter"]["modality_setting"][infer_result_single["ego_modality"]]["postprocess"][
                "gt_range"
            ]
        except:
            gt_range = hypes["postprocess"]["gt_range"]
        simple_vis.visualize(
            infer_result_single,
            batch_data["ego"]["origin_lidar"][0],
            gt_range,
            vis_save_path,
            method="bev",
            transformation_matrix_clean=torch.inverse(batch_data["ego"]["transformation_matrix_clean"][idx]),
            transformation_matrix=torch.inverse(batch_data["ego"]["transformation_matrix"][idx]),
            left_hand=opt.left_hand,
            show_bev=opt.show_bev,
            pcd_modality=batch_data["ego"]["origin_lidar_modality"][0],
        )


def eval_segmentation_result(opt, infer_result_single, idx, work_dir, i):
    """
    Calculate IoU during training.

    Parameters
    ----------
    batch_dict: dict
        The data that contains the gt.

    output_dict : dict
        The output directory with predictions.

    Returns
    -------
    The iou for static and dynamic bev map.
    """
    pred_dict = infer_result_single["pred_box_tensor"]
    gt_dict = infer_result_single["gt_box_tensor"]
    if pred_dict is None or gt_dict is None:
        return None, None
    # score_dict = infer_result_single['pred_score']
    batch_size = gt_dict["static_bev"].shape[0]
    assert batch_size == 1, "Only support batch size 1 for now."

    gt_static = gt_dict["static_bev"].detach().cpu().data.numpy()[0]
    gt_static = np.array(gt_static, dtype=int)

    gt_dynamic = gt_dict["dynamic_bev"].detach().cpu().data.numpy()[0]
    gt_dynamic = np.array(gt_dynamic, dtype=int)

    pred_static = pred_dict["static_map"]
    pred_static = torchvision.transforms.CenterCrop(gt_static.shape)(pred_static[0]).detach().cpu().data.numpy()
    pred_static = np.array(pred_static, dtype=int)

    pred_dynamic = pred_dict["dynamic_map"]
    pred_dynamic = torchvision.transforms.CenterCrop(gt_dynamic.shape)(pred_dynamic[0]).detach().cpu().data.numpy()
    pred_dynamic = np.array(pred_dynamic, dtype=int)
    
    iou_dynamic = mean_IU(pred_dynamic, gt_dynamic)
    iou_static = mean_IU(pred_static, gt_static)

    if i % opt.save_vis_interval == 0:
        vis_save_path_root = os.path.join(work_dir, f'vis_{opt.infer_info}{"_protocol" if opt.protocol_result else ""}')
        if not os.path.exists(vis_save_path_root):
            os.makedirs(vis_save_path_root)

        save_path = os.path.join(vis_save_path_root, "%05d_bev_seg.png" % i)
        static_save_path = os.path.join(vis_save_path_root, "%05d_bev_static.png" % i)
        dynamic_save_path = os.path.join(vis_save_path_root, "%05d_bev_dynamic.png" % i)

        static_gt_save_path = os.path.join(vis_save_path_root, "%05d_gt_static.png" % i)
        dynamic_gt_save_path = os.path.join(vis_save_path_root, "%05d_gt_dynamic.png" % i)

        colors = [(255, 255, 255), (255, 200, 200), (20, 20, 220), (80, 40, 40)]
        seg_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        dynamic_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        static_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        static_gt = np.ones((256, 256, 3), dtype=np.uint8) * 255
        dynamic_gt = np.ones((256, 256, 3), dtype=np.uint8) * 255

        for j in range(3):
            seg_image[pred_static == j] = colors[j]
            static_image[pred_static == j] = colors[j]
            static_gt[gt_static == j] = colors[j]
        seg_image[pred_dynamic == 1] = colors[3]
        dynamic_image[pred_dynamic == 1] = colors[3]
        dynamic_gt[gt_dynamic == 1] = colors[3]
        cv2.imwrite(save_path, seg_image)
        cv2.imwrite(static_save_path, static_image)
        cv2.imwrite(dynamic_save_path, dynamic_image)
        cv2.imwrite(static_gt_save_path, static_gt)
        cv2.imwrite(dynamic_gt_save_path, dynamic_gt)

        if opt.show_bev:
            simple_vis.visualize_bev(infer_result_single, os.path.join(vis_save_path_root, "%05d_bev.png" % i))

    return iou_static, iou_dynamic


def compute_bev_from_voxels(
    voxel_grid,
    vehicle_label=8,
    road_label=5,
    other_label_value=14
):
    """
    voxel_grid: (H, W, Z) integer labels
    Returns BEV of shape (H, W) with:
      0 = unknown or empty
      road_label = road
      vehicle_label = vehicle
      other_label_value = all other semantic classes grouped
    """
    # Vehicle occupies a voxel column if any voxel == vehicle_label
    vehicle_mask = (voxel_grid == vehicle_label).any(dim=2)
    # Road mask excludes vehicles (priority to vehicles)
    road_mask = (~vehicle_mask) & (voxel_grid == road_label).any(dim=2)

    # Known labels (excluding unknown=0 and empty=13)
    known_mask = (voxel_grid != 0) & (voxel_grid != 13)
    # Mask for any other labels (priority after road/vehicle)
    other_mask = (~vehicle_mask) & (~road_mask) & known_mask.any(dim=2)

    bev = torch.zeros(voxel_grid.shape[:2], dtype=torch.uint8, device=voxel_grid.device)
    bev[other_mask] = other_label_value
    bev[road_mask] = road_label
    bev[vehicle_mask] = vehicle_label
    return bev  # (H, W)



def compute_iou(pred_bev, gt_bev, class_id):
    pred_mask = (pred_bev == class_id)
    gt_mask = (gt_bev == class_id)

    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    if union > 0:
        return (intersection.float() / union.float()).item()  # return float
    else:
        return 0.0


def eval_occseg_result(pred_occ, gt_occ, occshape):
    # print(pred_occ.shape, gt_occ.shape)
    pred_bev = compute_bev_from_voxels(pred_occ.reshape(occshape))
    gt_bev = compute_bev_from_voxels(gt_occ.reshape(occshape))

    # --- Compute IoUs for each class ---
    vehicle_iou = compute_iou(pred_bev, gt_bev, 8)
    road_iou = compute_iou(pred_bev, gt_bev, 5)
    other_iou = compute_iou(pred_bev, gt_bev, 14)
    return road_iou, vehicle_iou, other_iou


if __name__ == "__main__":
    main()
