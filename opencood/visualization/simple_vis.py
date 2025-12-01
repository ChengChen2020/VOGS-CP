# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License


from matplotlib import pyplot as plt
import numpy as np
import copy
import cv2
from collections import defaultdict

from opencood.tools.inference_utils import get_cav_box
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev
from opencood.visualization.vis_utils import rotate_2d


def visualize(infer_result, 
              pcd, 
              pc_range, 
              save_path, 
              method='3d', 
              left_hand=False,
              show_bev=False,
              pcd_modality=None,
              transformation_matrix_clean=None,
              transformation_matrix=None,
              angle=-90
              ):
        """
        Visualize the prediction, ground truth with point cloud together.
        They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

        Parameters
        ----------
        infer_result:
            pred_box_tensor : torch.Tensor
                (N, 8, 3) prediction.

            gt_tensor : torch.Tensor
                (N, 8, 3) groundtruth bbx
            
            uncertainty_tensor : optional, torch.Tensor
                (N, ?)

            lidar_agent_record: optional, torch.Tensor
                (N_agnet, )


        pcd : torch.Tensor
            PointCloud, (N, 4).

        pc_range : list
            [xmin, ymin, zmin, xmax, ymax, zmax]

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        method: str, 'bev' or '3d'

        """
        plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
        pc_range = [int(i) for i in pc_range]
        
        pcd_np = pcd.cpu().numpy()
        transformation_matrix_clean = transformation_matrix_clean.cpu().numpy()
        if transformation_matrix_clean is not None:
            pcd_np = np.dot(pcd_np[:,:3], transformation_matrix_clean[:3,:3].T) + transformation_matrix_clean[:3,3]
        
        if infer_result is not None:

            pred_box_tensor = infer_result.get("pred_box_tensor", None)
            gt_box_tensor = infer_result.get("gt_box_tensor", None)

            if pred_box_tensor is not None:
                pred_box_np = pred_box_tensor.cpu().numpy()
                pred_name = ['pred'] * pred_box_np.shape[0]

                score = infer_result.get("score_tensor", None)
                if score is not None:
                    score_np = score.cpu().numpy()
                    pred_name = [f'score:{score_np[i]:.3f}' for i in range(score_np.shape[0])]

                uncertainty = infer_result.get("uncertainty_tensor", None)
                if uncertainty is not None:
                    uncertainty_np = uncertainty.cpu().numpy()
                    uncertainty_np = np.exp(uncertainty_np)
                    d_a_square = 1.6**2 + 3.9**2
                    
                    if uncertainty_np.shape[1] == 3:
                        uncertainty_np[:,:2] *= d_a_square
                        uncertainty_np = np.sqrt(uncertainty_np) 
                        # yaw angle is in radian, it's the same in g2o SE2's setting.

                        pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:.3f} a_u:{uncertainty_np[i,2]:.3f}' \
                                        for i in range(uncertainty_np.shape[0])]

                    elif uncertainty_np.shape[1] == 2:
                        uncertainty_np[:,:2] *= d_a_square
                        uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                        pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f}' \
                                        for i in range(uncertainty_np.shape[0])]

                    elif uncertainty_np.shape[1] == 7:
                        uncertainty_np[:,:2] *= d_a_square
                        uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                        pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f} a_u:{uncertainty_np[i,6]:3f}' \
                                        for i in range(uncertainty_np.shape[0])]                    

            if gt_box_tensor is not None:
                gt_box_np = gt_box_tensor.cpu().numpy()
                gt_name = ['gt'] * gt_box_np.shape[0]

        if method == 'bev':
            canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=left_hand) 
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
            
            pcd_colors = np.ones((canvas_xy.shape[0], 3))
            if pcd_modality is not None:
                pcd_colors[:] = np.array([66,66,66])
                color_map = {1: np.array([165, 59, 158]),
                                2: np.array([38, 158, 2]),
                                3: np.array([180, 123, 51]),
                                4: np.array([30, 88, 145])}
                for modality, color in color_map.items():
                    pcd_colors[pcd_modality.detach().cpu().numpy() == modality] = color
            
            gt_box_np[:, :4, :2] = rotate_2d(gt_box_np[:, :4, :2], angle)
            pred_box_np[:, :4, :2] = rotate_2d(pred_box_np[:, :4, :2], angle)
            canvas_xy = rotate_2d(canvas_xy, angle).astype(np.int32)
                        
            canvas.draw_canvas_points(canvas_xy[valid_mask], 
                                      radius=-1, 
                                      colors=pcd_colors[valid_mask],
                                      ) # Only draw valid points
            if infer_result is not None:
                if gt_box_tensor is not None:
                    canvas.draw_boxes(gt_box_np,colors=(0,185,0), 
                                    #   texts=gt_name
                                    texts=None
                    )
                    # canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=['']*len(gt_name), box_line_thickness=4) # paper visualization
                if pred_box_tensor is not None:
                    canvas.draw_boxes(pred_box_np, colors=(255,0,0), 
                                    #   texts=pred_name
                                    texts=None
                                      )
                    # canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=['']*len(pred_name), box_line_thickness=4) # paper visualization

                # heterogeneous
                agent_modality_list = infer_result.get("agent_modality_list", None)
                cav_box_np = infer_result.get("cav_box_np", None)
                
                # transform cav_box_np to the same coordinate system as the pcd
                if transformation_matrix_clean is not None:
                    cav_box_np = np.dot(cav_box_np, transformation_matrix_clean[:3,:3].T) + transformation_matrix_clean[:3,3]
                
                cav_box_np[:, :4, :2] = rotate_2d(cav_box_np[:, :4, :2], angle)
                if agent_modality_list is not None:
                    cav_box_np = copy.deepcopy(cav_box_np)
                    for i, modality_name in enumerate(agent_modality_list):
                        # if modality_name == "m1":
                        #     color = (0,191,255)
                        # elif modality_name == "m2":
                        #     color = (255,185,15)
                        # elif modality_name == "m3":
                        #     color = (138,211,222)
                        # elif modality_name == 'm4':
                        #     color = (32, 60, 160)
                        # else:
                        #     color = (66,66,66)
                        
                        if modality_name == "m1":
                            color = (165, 59, 158)
                        elif modality_name == "m2":
                            color = (38, 158, 2)
                        elif modality_name == "m3":
                            color = (180, 123, 51)
                        elif modality_name == 'm4':
                            color = (30, 88, 145)
                        else:
                            color = (66,66,66)
                        
                        

                        # render_dict = {'m1': 'L1', 'm2':"C1", 'm3':'L2', 'm4':'C2'}
                        # canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=[render_dict[modality_name]], box_text_size=1.5, box_line_thickness=5) # paper visualization
                        
                        # canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=[modality_name])
                        render_dict = {'m1': 'A1', 'm2':"A2", 'm3':'A3', 'm4':'A4'}
                        canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=[render_dict[modality_name]], box_text_size=1)
                        


        elif method == '3d':
            canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
            canvas.draw_canvas_points(canvas_xy[valid_mask], colors=pcd_modality)
            if infer_result is not None:
                if gt_box_tensor is not None:
                    canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
                if pred_box_tensor is not None:
                    canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)

                # heterogeneous
                agent_modality_list = infer_result.get("agent_modality_list", None)
                cav_box_np = infer_result.get("cav_box_np", None)
                if agent_modality_list is not None:
                    cav_box_np = copy.deepcopy(cav_box_np)
                    for i, modality_name in enumerate(agent_modality_list):
                        if modality_name == "m1":
                            color = (0,191,255)
                        elif modality_name == "m2":
                            color = (255,185,15)
                        elif modality_name == "m3":
                            color = (123,0,70)
                        else:
                            color = (66,66,66)
                        canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=[modality_name])

        else:
            raise(f"Not Completed for f{method} visualization.")

        plt.axis("off")

        plt.imshow(canvas.canvas)
        plt.tight_layout()
        plt.savefig(save_path, transparent=False, dpi=500)
        plt.clf()
        plt.close()
        
        
        if show_bev:
            ori_feat_dict = infer_result.get("ori_feat_dict", None)
            feat_tensor = infer_result.get("feat_tensor", None)
            fused_feat = infer_result.get("fused_feat", None)
            feat_single = infer_result.get("feat_single", None)
            protocol_feat = infer_result.get("protocol_feat", None)
            protocol_fused_feature = infer_result.get("protocol_fused_feature", None)
            
            ##############################
            
            
            
            # if ori_feat_dict is not None:
            
            #     # for modality_name in ori_feat_dict.keys():
            #     #     ori_feat = ori_feat_dict[modality_name]
            #     #     max_value = -np.inf
            #     #     min_value = np.inf
            #     #     for i in ori_feat.keys():
            #     #         ori_feat[i] = ori_feat[i].abs().mean(0)
            #     #         max_value = max(max_value, ori_feat[i].max())
            #     #         min_value = min(min_value, ori_feat[i].min())
                
            #     for modality_name in ori_feat_dict.keys():
            #         ori_feat = ori_feat_dict[modality_name]
            #         max_value = -np.inf
            #         min_value = np.inf
            #         for i in ori_feat.keys():
            #             ori_feat[i] = ori_feat[i].abs().mean(0)
            #             max_value = max(max_value, ori_feat[i].max())
            #             min_value = min(min_value, ori_feat[i].min())
            #         for i, feat in ori_feat.items():
            #             canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
            #                                                 value_range=(min_value, max_value))
            #             canvas.visualize(feat)
            #             cv2.imwrite(save_path.replace('.png', f'M_{modality_name}_{i}.png'), canvas.get_canvas())
            
            max_value = -np.inf
            min_value = np.inf
            
            if ori_feat_dict is not None:
        
                for modality_name in ori_feat_dict.keys():
                    ori_feat = ori_feat_dict[modality_name]
                    for i in ori_feat.keys():
                        ori_feat[i] = ori_feat[i].abs().mean(0)
                        max_value = max(max_value, ori_feat[i].max())
                        min_value = min(min_value, ori_feat[i].min())
                
                for modality_name in ori_feat_dict.keys():
                    ori_feat = ori_feat_dict[modality_name]
                    for i, feat in ori_feat.items():
                        canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
                                                            value_range=(min_value, max_value))
                        canvas.visualize(feat)
                        cv2.imwrite(save_path.replace('.png', f'M_{modality_name}_{i}.png'), canvas.get_canvas())
                    
            ##############################
                    
            
                
            if feat_tensor is not None:
                
                if isinstance(feat_tensor, dict):
                    feat_dict = defaultdict(list)
                    for modality in feat_tensor.keys():
                        max_value = -np.inf
                        min_value = np.inf
                        
                        for i in range(len(feat_tensor[modality])):
                            feat_tensor_i = feat_tensor[modality][i].abs().mean(0)
                            max_value = max(max_value, feat_tensor_i.max())
                            min_value = min(min_value, feat_tensor_i.min())
                            feat_dict[modality].append(feat_tensor_i)
                            
                        for i, feat in enumerate(feat_dict[modality]):
                            
                            canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
                                                                value_range=(min_value, max_value))
                            canvas.visualize(feat)
                            cv2.imwrite(save_path.replace('.png', f'M2P2M_{modality}_{i}.png'), canvas.get_canvas())
                else:
                    feat_list = []
                    for i in range(len(feat_tensor)):
                        feat_list.append(feat_tensor[i].abs().mean(0))
                        max_value = max(max_value, feat_list[i].max())
                        min_value = min(min_value, feat_list[i].min())
                        
                    for i, feat in enumerate(feat_list):
                        canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
                                                            value_range=(min_value, max_value))
                        canvas.visualize(feat)
                        cv2.imwrite(save_path.replace('.png', f'M2P2M_{i}.png'), canvas.get_canvas())
                
            ##############################
                
            max_value = -np.inf
            min_value = np.inf
            
            if feat_single is not None:
                
                for i in range(len(feat_single)):
                    feat_single[i] = feat_single[i].abs().mean(0)
                    max_value = max(max_value, feat_single[i].max())
                    min_value = min(min_value, feat_single[i].min())
                    
                for i, feat in enumerate(feat_single):
                    canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
                                                        value_range=(min_value, max_value))
                    canvas.visualize(feat)
                    cv2.imwrite(save_path.replace('.png', f'FM_single_{i}.png'), canvas.get_canvas())
                
            ##############################
            
            # max_value = -np.inf
            # min_value = np.inf
            
            if fused_feat is not None:
                if isinstance(fused_feat, dict):
                    for modality in fused_feat.keys():
                        max_value = -np.inf
                        min_value = np.inf
                        if len(fused_feat[modality]) == 0:
                            continue
                        fused_feat[modality] = fused_feat[modality][0].abs().mean(0)
                        
                        max_value = max(max_value, fused_feat[modality].max())
                        min_value = min(min_value, fused_feat[modality].min())
                        
                        canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=fused_feat[modality].shape[-2:],     
                                                            value_range=(min_value, max_value))
                        canvas.visualize(fused_feat[modality])
                        cv2.imwrite(save_path.replace('.png', f'FM_{modality}.png'), canvas.get_canvas())
                        
                else:

                    fused_feat = fused_feat[0].abs().mean(0)
                    max_value = max(max_value, fused_feat.max())
                    min_value = min(min_value, fused_feat.min())
                    
                    canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=fused_feat.shape[-2:], 
                                                            value_range=(min_value, max_value))
                    canvas.visualize(fused_feat)
                    cv2.imwrite(save_path.replace('.png', f'FM.png'), canvas.get_canvas())
                
            ##############################
            
            max_value = -np.inf
            min_value = np.inf
            
            if protocol_feat is not None:
                
                protocol_feat_list = []
                for i in range(len(protocol_feat)):
                    protocol_feat_list.append(protocol_feat[i].abs().mean(0))
                    max_value = max(max_value, protocol_feat_list[i].max())
                    min_value = min(min_value, protocol_feat_list[i].min())
                    
                for i, feat in enumerate(protocol_feat_list):
                    canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
                                                        value_range=(min_value, max_value))
                    canvas.visualize(feat)
                    cv2.imwrite(save_path.replace('.png', f'M2P_{i}.png'), canvas.get_canvas())  
            
            ##############################
            
            max_value = -np.inf
            min_value = np.inf
            
            if protocol_fused_feature is not None:
                    
                protocol_fused_feature = protocol_fused_feature[0].abs().mean(0)
                max_value = max(max_value, protocol_fused_feature.max())
                min_value = min(min_value, protocol_fused_feature.min())
                
                canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=protocol_fused_feature.shape[-2:], 
                                                        value_range=(min_value, max_value))
                canvas.visualize(protocol_fused_feature)
                cv2.imwrite(save_path.replace('.png', f'FM2P_.png'), canvas.get_canvas())




def visualize_bev(infer_result, save_path):
    # ori_feat_dict = infer_result.get("ori_feat_dict", None)
    # feat_tensor = infer_result.get("feat_tensor", None)
    # fused_feat = infer_result.get("fused_feat", None)
    # feat_single = infer_result.get("feat_single", None)
    # protocol_feat = infer_result.get("protocol_feat", None)
    # protocol_fused_feature = infer_result.get("protocol_fused_feature", None)

    
    # ##############################
    
    
    
    # # if ori_feat_dict is not None:
    
    # #     # for modality_name in ori_feat_dict.keys():
    # #     #     ori_feat = ori_feat_dict[modality_name]
    # #     #     max_value = -np.inf
    # #     #     min_value = np.inf
    # #     #     for i in ori_feat.keys():
    # #     #         ori_feat[i] = ori_feat[i].abs().mean(0)
    # #     #         max_value = max(max_value, ori_feat[i].max())
    # #     #         min_value = min(min_value, ori_feat[i].min())
        
    # #     for modality_name in ori_feat_dict.keys():
    # #         ori_feat = ori_feat_dict[modality_name]
    # #         max_value = -np.inf
    # #         min_value = np.inf
    # #         for i in ori_feat.keys():
    # #             ori_feat[i] = ori_feat[i].abs().mean(0)
    # #             max_value = max(max_value, ori_feat[i].max())
    # #             min_value = min(min_value, ori_feat[i].min())
    # #         for i, feat in ori_feat.items():
    # #             canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
    # #                                                 value_range=(min_value, max_value))
    # #             canvas.visualize(feat)
    # #             cv2.imwrite(save_path.replace('.png', f'M_{modality_name}_{i}.png'), canvas.get_canvas())
    
    # max_value = -np.inf
    # min_value = np.inf
    
    # if ori_feat_dict is not None:
        
    #     if isinstance(ori_feat_dict, dict):

    #         for modality_name in ori_feat_dict.keys():
    #             ori_feat = ori_feat_dict[modality_name]
    #             for i in ori_feat.keys():
    #                 ori_feat[i] = ori_feat[i].abs().mean(0)
    #                 max_value = max(max_value, ori_feat[i].max())
    #                 min_value = min(min_value, ori_feat[i].min())
            
    #         for modality_name in ori_feat_dict.keys():
    #             ori_feat = ori_feat_dict[modality_name]
    #             for i, feat in ori_feat.items():
    #                 canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
    #                                                     value_range=(min_value, max_value))
    #                 canvas.visualize(feat)
    #                 cv2.imwrite(save_path.replace('.png', f'M_{modality_name}_{i}.png'), canvas.get_canvas())
    #     else:
    #         ori_feat = ori_feat_dict[0].abs().mean(0)
    #         canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=ori_feat.shape[-2:], 
    #                                             value_range=(ori_feat.min(), ori_feat.max()))
    #         # import pdb; pdb.set_trace()
    #         canvas.visualize(ori_feat)
            
    #         cv2.imwrite(save_path.replace('.png', f'M.png'), canvas.get_canvas())
            
    # ##############################
            
    # max_value = -np.inf
    # min_value = np.inf
        
    # if feat_tensor is not None:
            
    #     feat_list = []
    #     for i in range(len(feat_tensor)):
    #         feat_list.append(feat_tensor[i].mean(0))
    #         max_value = max(max_value, feat_list[i].max())
    #         min_value = min(min_value, feat_list[i].min())
            
    #     for i, feat in enumerate(feat_list):
    #         canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
    #                                             value_range=(min_value, max_value))
    #         canvas.visualize(feat)
    #         cv2.imwrite(save_path.replace('.png', f'M2P2M_{i}.png'), canvas.get_canvas())
        
    # ##############################
        
    # max_value = -np.inf
    # min_value = np.inf
    
    # if feat_single is not None:
        
    #     for i in range(len(feat_single)):
    #         feat_single[i] = feat_single[i].abs().mean(0)
    #         max_value = max(max_value, feat_single[i].max())
    #         min_value = min(min_value, feat_single[i].min())
            
    #     for i, feat in enumerate(feat_single):
    #         canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
    #                                             value_range=(min_value, max_value))
    #         canvas.visualize(feat)
    #         cv2.imwrite(save_path.replace('.png', f'FM_single_{i}.png'), canvas.get_canvas())
        
    # ##############################
    
    # max_value = -np.inf
    # min_value = np.inf
    
    # if fused_feat is not None:

    #     fused_feat = fused_feat[0].abs().mean(0)
    #     max_value = max(max_value, fused_feat.max())
    #     min_value = min(min_value, fused_feat.min())
        
    #     canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=fused_feat.shape[-2:], 
    #                                             value_range=(min_value, max_value))
    #     canvas.visualize(fused_feat)
    #     cv2.imwrite(save_path.replace('.png', f'FM.png'), canvas.get_canvas())
        
    # ##############################
    
    # max_value = -np.inf
    # min_value = np.inf
    
    # if protocol_feat is not None:
        
    #     protocol_feat_list = []
    #     for i in range(len(protocol_feat)):
    #         protocol_feat_list.append(protocol_feat[i].abs().mean(0))
    #         max_value = max(max_value, protocol_feat_list[i].max())
    #         min_value = min(min_value, protocol_feat_list[i].min())
            
    #     for i, feat in enumerate(protocol_feat_list):
    #         canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
    #                                             value_range=(min_value, max_value))
    #         canvas.visualize(feat)
    #         cv2.imwrite(save_path.replace('.png', f'M2P_{i}.png'), canvas.get_canvas())  
    
    # ##############################
    
    # max_value = -np.inf
    # min_value = np.inf
    
    # if protocol_fused_feature is not None:
            
    #     protocol_fused_feature = protocol_fused_feature[0].abs().mean(0)
    #     max_value = max(max_value, protocol_fused_feature.max())
    #     min_value = min(min_value, protocol_fused_feature.min())
        
    #     canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=protocol_fused_feature.shape[-2:], 
    #                                             value_range=(min_value, max_value))
    #     canvas.visualize(protocol_fused_feature)
    #     cv2.imwrite(save_path.replace('.png', f'FM2P_.png'), canvas.get_canvas())




    ori_feat_dict = infer_result.get("ori_feat_dict", None)
    feat_tensor = infer_result.get("feat_tensor", None)
    fused_feat = infer_result.get("fused_feat", None)
    feat_single = infer_result.get("feat_single", None)
    protocol_feat = infer_result.get("protocol_feat", None)
    protocol_fused_feature = infer_result.get("protocol_fused_feature", None)
    
    ##############################
    
    
    
    # if ori_feat_dict is not None:
    
    #     # for modality_name in ori_feat_dict.keys():
    #     #     ori_feat = ori_feat_dict[modality_name]
    #     #     max_value = -np.inf
    #     #     min_value = np.inf
    #     #     for i in ori_feat.keys():
    #     #         ori_feat[i] = ori_feat[i].abs().mean(0)
    #     #         max_value = max(max_value, ori_feat[i].max())
    #     #         min_value = min(min_value, ori_feat[i].min())
        
    #     for modality_name in ori_feat_dict.keys():
    #         ori_feat = ori_feat_dict[modality_name]
    #         max_value = -np.inf
    #         min_value = np.inf
    #         for i in ori_feat.keys():
    #             ori_feat[i] = ori_feat[i].abs().mean(0)
    #             max_value = max(max_value, ori_feat[i].max())
    #             min_value = min(min_value, ori_feat[i].min())
    #         for i, feat in ori_feat.items():
    #             canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
    #                                                 value_range=(min_value, max_value))
    #             canvas.visualize(feat)
    #             cv2.imwrite(save_path.replace('.png', f'M_{modality_name}_{i}.png'), canvas.get_canvas())
    
    max_value = -np.inf
    min_value = np.inf
    
    if ori_feat_dict is not None:

        for modality_name in ori_feat_dict.keys():
            ori_feat = ori_feat_dict[modality_name]
            for i in ori_feat.keys():
                ori_feat[i] = ori_feat[i].abs().mean(0)
                max_value = max(max_value, ori_feat[i].max())
                min_value = min(min_value, ori_feat[i].min())
        
        for modality_name in ori_feat_dict.keys():
            ori_feat = ori_feat_dict[modality_name]
            for i, feat in ori_feat.items():
                canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
                                                    value_range=(min_value, max_value))
                canvas.visualize(feat)
                cv2.imwrite(save_path.replace('.png', f'M_{modality_name}_{i}.png'), canvas.get_canvas())
            
    ##############################
            
    
        
    if feat_tensor is not None:
        
        if isinstance(feat_tensor, dict):
            feat_dict = defaultdict(list)
            for modality in feat_tensor.keys():
                max_value = -np.inf
                min_value = np.inf
                
                for i in range(len(feat_tensor[modality])):
                    feat_tensor_i = feat_tensor[modality][i].abs().mean(0)
                    max_value = max(max_value, feat_tensor_i.max())
                    min_value = min(min_value, feat_tensor_i.min())
                    feat_dict[modality].append(feat_tensor_i)
                    
                for i, feat in enumerate(feat_dict[modality]):
                    
                    canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
                                                        value_range=(min_value, max_value))
                    canvas.visualize(feat)
                    cv2.imwrite(save_path.replace('.png', f'M2P2M_{modality}_{i}.png'), canvas.get_canvas())
        else:
            feat_list = []
            for i in range(len(feat_tensor)):
                feat_list.append(feat_tensor[i].abs().mean(0))
                max_value = max(max_value, feat_list[i].max())
                min_value = min(min_value, feat_list[i].min())
                
            for i, feat in enumerate(feat_list):
                canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
                                                    value_range=(min_value, max_value))
                canvas.visualize(feat)
                cv2.imwrite(save_path.replace('.png', f'M2P2M_{i}.png'), canvas.get_canvas())
        
    ##############################
        
    max_value = -np.inf
    min_value = np.inf
    
    if feat_single is not None:
        
        for i in range(len(feat_single)):
            feat_single[i] = feat_single[i].abs().mean(0)
            max_value = max(max_value, feat_single[i].max())
            min_value = min(min_value, feat_single[i].min())
            
        for i, feat in enumerate(feat_single):
            canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
                                                value_range=(min_value, max_value))
            canvas.visualize(feat)
            cv2.imwrite(save_path.replace('.png', f'FM_single_{i}.png'), canvas.get_canvas())
        
    ##############################
    
    # max_value = -np.inf
    # min_value = np.inf
    
    if fused_feat is not None:
        if isinstance(fused_feat, dict):
            for modality in fused_feat.keys():
                max_value = -np.inf
                min_value = np.inf
                if len(fused_feat[modality]) == 0:
                    continue
                fused_feat[modality] = fused_feat[modality][0].abs().mean(0)
                max_value = max(max_value, fused_feat[modality].max())
                min_value = min(min_value, fused_feat[modality].min())
                
                canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=fused_feat[modality].shape[-2:],     
                                                    value_range=(min_value, max_value))
                canvas.visualize(fused_feat[modality])
                cv2.imwrite(save_path.replace('.png', f'FM_{modality}.png'), canvas.get_canvas())
                
        else:

            fused_feat = fused_feat[0].abs().mean(0)
            max_value = max(max_value, fused_feat.max())
            min_value = min(min_value, fused_feat.min())
            
            canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=fused_feat.shape[-2:], 
                                                    value_range=(min_value, max_value))
            canvas.visualize(fused_feat)
            cv2.imwrite(save_path.replace('.png', f'FM.png'), canvas.get_canvas())
        
    ##############################
    
    max_value = -np.inf
    min_value = np.inf
    
    if protocol_feat is not None:
        
        protocol_feat_list = []
        for i in range(len(protocol_feat)):
            protocol_feat_list.append(protocol_feat[i].abs().mean(0))
            max_value = max(max_value, protocol_feat_list[i].max())
            min_value = min(min_value, protocol_feat_list[i].min())
            
        for i, feat in enumerate(protocol_feat_list):
            canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=feat.shape[-2:], 
                                                value_range=(min_value, max_value))
            canvas.visualize(feat)
            cv2.imwrite(save_path.replace('.png', f'M2P_{i}.png'), canvas.get_canvas())  
    
    ##############################
    
    max_value = -np.inf
    min_value = np.inf
    
    if protocol_fused_feature is not None:
            
        protocol_fused_feature = protocol_fused_feature[0].abs().mean(0)
        max_value = max(max_value, protocol_fused_feature.max())
        min_value = min(min_value, protocol_fused_feature.min())
        
        canvas = canvas_bev.Canvas_BEV_feature(canvas_shape=protocol_fused_feature.shape[-2:], 
                                                value_range=(min_value, max_value))
        canvas.visualize(protocol_fused_feature)
        cv2.imwrite(save_path.replace('.png', f'FM2P_.png'), canvas.get_canvas())