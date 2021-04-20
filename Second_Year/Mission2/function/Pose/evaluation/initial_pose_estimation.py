import os
import glob
import cv2
import pickle
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import DataLoader

from ..bop_toolkit_lib.misc import *
from ..models.correspondence_block_model import CorrespondenceBlockModel
from ..models.part4_model import Part4DirModel
from ..utils.pose_gt import *

sys.path.append('./function/utilities')
from function.utilities.utils import *

class Color():
    def __init__(self):
        """color palette (BGR) """
        self.colors = {
            0 : (0, 0, 0), # bg : black
            1 : (0, 0, 255), # part1 : red
            2 : (0, 165, 255), # part2 : orange 
            3 : (0, 255, 255), # part3 : yellow
            4 : (0, 128, 0), # part4 : green
            5 : (255, 0, 0), # part5 : blue
            6 : (130, 0, 75), # part6 : purple
            7 : (255, 0, 255), # part7 (part2 + bracket) : magenta
            8 : (255, 255, 0), # part8 (part3 + bracket) : cyan
            11 : (217, 179, 255) # part5 or part6 : light purple
        }
    def __call__(self, x):
        return self.colors[x]

class Part4DirEstimation():
    def __init__(self, args):
        self.args = args
        refresh_folder(args.opt.initial_pose_estimation_part4dir_path)
        # load weights (part4)
        part4dir_model_adr = args.opt.part4dir_model_path + '/part4dir_block.pt'
        with args.graph_part4dir.as_default():
            sess = args.sess_part4dir
            self.model = Part4DirModel()
            print('PART4 DIRECTION MODEL : Loading saved model from', part4dir_model_adr)
            checkpoint = torch.load(part4dir_model_adr, map_location='cuda:0') 
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model.cuda(0)

    def test(self, images):
        """estimate direction (0, 1) of part4 images
        
        Args:
            images: list of images resized to (224, 224) (containes images that are not part4)
        
        Returns:
            pred_directions : list of predicted directions
        """
        with torch.no_grad():
            images = [transforms.ToTensor()(image) for image in images]
            images = [torch.unsqueeze(image, 0) for image in images]
            images_tensor = None
            for image in images:
                images_tensor = torch.cat((images_tensor, image), 0) if images_tensor is not None else image
            images_tensor = images_tensor.cuda(0)
            logit = self.model(images_tensor)
            pred_directions = torch.argmax(logit, dim=1).squeeze().cpu().detach().numpy()
            pred_directions = pred_directions.tolist()
            if isinstance(pred_directions, int):
                pred_directions = [pred_directions]
        return pred_directions

    def save_part4dir(self, step_num, directions, pred_directions, part_imgs):
        """ save part4 direction predictions of Pnp and part4dir model
        
        border : inner = pnp, outer = part4dir
        color : yellow = 0, blue = 1
        
        Args:
            step_num: step number
            directions: pnp predictions
            pred_directions: part4dir predictions
            part_imgs: input images of  part4dir network
        """
        
        args = self.args
        if args.opt.save_part4dir:
            cad_models = args.cad_models[step_num]
            part_imgs_bboxed = args.parts_bboxed[step_num]
            for i, (direction, pred_direction, cad_model, part_img, part_img_bboxed) in enumerate(zip(directions, pred_directions, cad_models, part_imgs, part_imgs_bboxed)):
                if cad_model == 'part4':
                    plt.imsave(args.opt.initial_pose_estimation_part4dir_path + '/STEP{}_part{}_bbox'.format(step_num, i), part_img_bboxed)
                    border = 20
                    # inner
                    color = [0, 0, 255] if direction == 1 else [255, 255, 0]
                    part_img = cv2.copyMakeBorder(part_img, border, border, border, border, cv2.BORDER_CONSTANT, None, color)
                    # outer
                    color = [0, 0, 255] if pred_direction == 1 else [255, 255, 0]
                    part_img = cv2.copyMakeBorder(part_img, border, border, border, border, cv2.BORDER_CONSTANT, None, color)
                    plt.clf()
                    plt.figure(figsize=(4, 4))
                    plt.axis('off')
                    part4dir_img = part_img
                    plt.imshow(part4dir_img)
                    plt.title('pnp : {}, part4dir : {}'.format(direction, pred_direction))
                    plt.savefig(args.opt.initial_pose_estimation_part4dir_path + '/STEP{}_part{}_part4dir'.format(step_num, i))
                    plt.close()


class InitialPoseEstimation():
    def __init__(self, args):
        self.args = args.opt
        self.pose_function_path = './function/Pose'
        self.pose_data_path = self.pose_function_path + '/data'
        if args.opt.pose_model_adr:
            pose_model_adr = args.opt.pose_model_adr
        else:
            pose_model_adr = args.opt.pose_model_path + '/correspondence_block_finetuned_1.pt'

        # load weights
        with args.graph_pose.as_default():
            sess = args.sess_pose
            self.model = CorrespondenceBlockModel(3, 9, 256) # in channels, id channels, uvw channels
            print('POSE MODEL : Loading saved model from', pose_model_adr)
            checkpoint = torch.load(pose_model_adr, map_location='cuda:0') 
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model.cuda(0)

        # load data
        with open(self.pose_data_path + '/uvw_xyz_correspondences.pkl', 'rb') as f:
            self.correspondence_dicts = pickle.load(f)
        with open(self.pose_data_path + '/point_cloud.pkl', 'rb') as f:
            self.point_cloud_radii = pickle.load(f)
            self.point_clouds = pickle.load(f)
        for i in range(len(self.point_clouds)):
            sampleNum = 500 if i==0 or i==4 or i==5 else 300
            self.point_clouds[i] = self.point_clouds[i][np.random.choice(self.point_clouds[i].shape[0], sampleNum, replace=False)]
        self.point_clouds = [x / 100 for x in self.point_clouds]
        refresh_folder(args.opt.initial_pose_estimation_prediction_maps_path)
        refresh_folder(args.opt.initial_pose_estimation_visualization_path)
        refresh_folder(args.opt.part_id_pose_path)


    def test(self, args, step_num):
        """predict pose

        update self.return_dict

        Args:
            args: self of Assembly class
            step_num: step number
        """
        color = Color()
        image = args.steps[step_num].copy()
        self.H, self.W, _ = image.shape
        with torch.no_grad():
            # STEP0
            # load image on GPU
            image = cv2.resize(image.astype(np.uint8), (240, 320), interpolation=cv2.INTER_NEAREST)
            image_tensor = transforms.ToTensor()(image)
            image_tensor = torch.unsqueeze(image_tensor, 0)
            image_tensor = image_tensor.cuda(0)

            # feed forward
            idmask_pred, edgemask_pred, umask_pred, vmask_pred, wmask_pred = self.model(image_tensor)

            idmask_pred = torch.argmax(idmask_pred, dim=1).squeeze().cpu().detach().numpy()  # leave maximum value
            edgemask_pred = torch.argmax(edgemask_pred, dim=1).squeeze().cpu().detach().numpy()  # leave maximum value
            umask_pred = torch.argmax(umask_pred, dim=1).squeeze().cpu().detach().numpy()  # leave maximum value
            vmask_pred = torch.argmax(vmask_pred, dim=1).squeeze().cpu().detach().numpy()  # leave maximum value
            wmask_pred = torch.argmax(wmask_pred, dim=1).squeeze().cpu().detach().numpy()  # leave maximum value

            ID = idmask_pred.astype(np.uint8)

            
            # remove small components
            ID_binarized = np.where(ID > 0, 255, 0).astype(np.uint8)
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(ID_binarized, connectivity=4)
            remain_indices = []
            for i, stat in enumerate(stats):
                remain = True
                x, y, w, h, px = stat
                if px < 300:
                    remain = False
                if remain:
                    remain_indices.append(i)
            mask = (np.isin(labels, remain_indices) * 255).astype(np.uint8)
            ID = np.where(mask > 0, ID, 0)
            
            _H, _W = ID.shape
            ID_color = np.zeros((_H, _W, 3)).astype(np.uint8)
            for h in range(_H):
                for w in range(_W):
                    ID_color[h, w, :] = color(ID[h, w])

            EDGE = (255 * edgemask_pred).astype(np.uint8)
            U = umask_pred.astype(np.uint8)
            V = vmask_pred.astype(np.uint8)
            W = wmask_pred.astype(np.uint8)
            UVW = np.stack((U, V, W), axis=-1)
            if args.opt.save_pose_prediction_maps:
                save_path = args.opt.initial_pose_estimation_prediction_maps_path
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path + '/STEP{}_ID.png'.format(step_num), ID_color)
                cv2.imwrite(save_path + '/STEP{}_EDGE.png'.format(step_num), EDGE)
                cv2.imwrite(save_path + '/STEP{}_U.png'.format(step_num), U)
                cv2.imwrite(save_path + '/STEP{}_V.png'.format(step_num), V)
                cv2.imwrite(save_path + '/STEP{}_W.png'.format(step_num), W)
                cv2.imwrite(save_path + '/STEP{}_UVW.png'.format(step_num), UVW)

            # STEP1 : post-process ID-map with detection classes (part2, part3, part7, part8)
            parts_loc = args.parts_loc[step_num]
            cad_models = args.cad_models[step_num]
            
            detected_models = list({'part2', 'part3', 'part7', 'part8'} & set(cad_models))
            undetected_models = list({'part2', 'part3', 'part7', 'part8'} - set(cad_models))
            for undetected_model in undetected_models:
                if undetected_model == 'part2':
                    # 2 -> 7 -> 3 -> 8
                    if 'part7' in detected_models:
                        ID = np.where(ID == 2, 7, ID)
                    elif 'part3' in detected_models:
                        ID = np.where(ID == 2, 3, ID)
                    elif 'part8' in detected_models:
                        ID = np.where(ID == 2, 8, ID) 
                if undetected_model == 'part3':
                    # 3 -> 8 -> 2 -> 7
                    if 'part8' in detected_models:
                        ID = np.where(ID == 3, 8, ID)
                    elif 'part2' in detected_models:
                        ID = np.where(ID == 3, 2, ID)
                    elif 'part7' in detected_models:
                        ID = np.where(ID == 3, 7, ID)
                if undetected_model == 'part7':
                    # 7 -> 2 -> 8 -> 3
                    if 'part2' in detected_models:
                        ID = np.where(ID == 7, 2, ID)
                    elif 'part8' in detected_models:
                        ID = np.where(ID == 7, 8, ID)
                    elif 'part3' in detected_models:
                        ID = np.where(ID == 7, 3, ID)
                if undetected_model == 'part8':
                    # 8 -> 3 -> 7 -> 2
                    if 'part3' in detected_models:
                        ID = np.where(ID == 8, 3, ID)
                    elif 'part7' in detected_models:
                        ID = np.where(ID == 8, 7, ID)
                    elif 'part2' in detected_models:
                        ID = np.where(ID == 8, 2, ID)

            ID_step1 = ID
            ID_step1_color = np.zeros((_H, _W, 3)).astype(np.uint8)
            for h in range(_H):
                for w in range(_W):
                    ID_step1_color[h, w, :] = color(ID_step1[h, w])

            if args.opt.save_pose_prediction_maps:
                cv2.imwrite(save_path + '/STEP{}_ID_1.png'.format(step_num), ID_step1_color)
            

            # STEP2 : if no New part is found, infer from pose segmentation map
            max_pixels = 0
            
            new_part_should_be_found_condition = True
            if not len(args.circles_loc[step_num]):
                # no circles are detected
                new_part_should_be_found_condition = False
            grouped_serial_OCRs = args.connectors_serial_OCR[step_num]
            for grouped_serial_OCR in grouped_serial_OCRs:
                if '122925' in grouped_serial_OCR:
                    # short screw
                    new_part_should_be_found_condition = False

            # search detection results history to figure out whether new part should be found.
            if new_part_should_be_found_condition:
                detected_parts = args.used_parts_nonunique[step_num].copy() # current detection results
                prev_detected_parts = args.used_parts_nonunique_cumulative[step_num - 1].copy() # detection results until previous step
                new_parts = []
                for detected_part in detected_parts:
                    if detected_part in prev_detected_parts:
                        prev_detected_parts.remove(detected_part)
                    else:
                        new_parts.append(detected_part)
                if not len(new_parts):
                    # no New part found

                    # divide ID to ID_parts
                    ID_parts = []
                    part_ids = [0]
                    part_ids.extend(args.unused_parts[1])
                    for part_id in part_ids:
                        ID_part = np.where(ID == part_id, part_id, 0).astype(np.uint8)
                        ID_parts.append(ID_part)

                    # find connected components
                    num_comps, indexed_maps, stats, part_id_area_sums, centers = [], [], [], [], []
                    for ID_part in ID_parts:
                        num_comp, indexed_map, stat, center = cv2.connectedComponentsWithStats(ID_part, connectivity=8)

                        """
                        num_comp : number of components, including background
                        indexed_map : indexed map
                        stat : left(x), top(y), w, h, area(pixel count)
                        center : center positions of each component
                        """
                        num_comps.append(num_comp)
                        indexed_maps.append(indexed_map)
                        stat[0, -1] = 0 # set background pixel count to 0
                        stats.append(stat)
                        part_id_area_sums.append(np.sum(stat[:, -1]))
                        centers.append(center)

                    # choose part id from unused parts, which has largest pixel counts
                    look_up_part_ids = args.unused_parts[step_num]
                    max_pixels_look_up_part_id = 0
                    max_pixels = 0
                    for look_up_part_id in look_up_part_ids:
                        if part_id_area_sums[look_up_part_id] > max_pixels:
                            max_pixels = part_id_area_sums[look_up_part_id]
                            max_pixels_look_up_part_id = look_up_part_id
                    if max_pixels < 100:
                        # too small
                        max_pixels = 0
                        max_pixels_look_up_part_id = 0

                    # choose largest component of that part id
                    index = np.argmax(stats[max_pixels_look_up_part_id][:, -1])
                    print('Added "New" at step {} : part{} (part0 means none added)'.format(step_num, max_pixels_look_up_part_id))

                    ID_step2 = ID_step1
                    ID_step2_color = ID_step1_color

                    # add bbox info, u
                    ID_part_to_add = np.where(indexed_maps[max_pixels_look_up_part_id] == index, max_pixels_look_up_part_id, 0).astype(np.uint8)
                    nonzero_range_y, nonzero_range_x = np.nonzero(ID_part_to_add)
                    try:
                        y_min = np.min(nonzero_range_y)
                        y_max = np.max(nonzero_range_y)
                        x_min = np.min(nonzero_range_x)
                        x_max = np.max(nonzero_range_x)
                        ID_step2_color = cv2.rectangle(ID_step2_color, (x_min, y_min), (x_max, y_max), color=(255, 255, 255), thickness=2)
                        y_min = int(y_min * self.H / _H)
                        y_max = int(y_max * self.H / _H)
                        x_min = int(x_min * self.W / _W)
                        x_max = int(x_max * self.W / _W)
                        step_part_image_bboxed = args.steps[step_num].copy()
                        step_part_image_bboxed = cv2.rectangle(step_part_image_bboxed, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
                        args.parts_bboxed[step_num].append(step_part_image_bboxed)
                        args.parts_loc[step_num].append([x_min, y_min, x_max - x_min, y_max - y_min, 1.0])
                    except:
                        pass

                    # update self.used_parts, self.unused_parts, self.cad_models, self.used_parts_nonunique_cumulative
                    if max_pixels_look_up_part_id != 0:
                        args.used_parts[step_num].append(max_pixels_look_up_part_id)
                        args.used_parts[step_num].sort()
                        args.used_parts_nonunique_cumulative[step_num].append(max_pixels_look_up_part_id)
                        args.used_parts_nonunique_cumulative[step_num].sort()
                        try:
                            args.unused_parts[step_num + 1].remove(max_pixels_look_up_part_id)
                        except:
                            # incase of multiple parts
                            pass
                        args.cad_models[step_num].append('part' + str(max_pixels_look_up_part_id))

                    # finish
                    ID_step2 = ID_step1
            if max_pixels == 0:
                ID_step2 = ID_step1
                ID_step2_color = ID_step1_color
            if args.opt.save_pose_prediction_maps:
                cv2.imwrite(save_path + '/STEP{}_ID_2.png'.format(step_num), ID_step2_color)


            # STEP3 : post-process ID-map with detection bbox
            parts_loc = args.parts_loc[step_num]
            cad_models = args.cad_models[step_num]
            ID_step3 = np.zeros(ID.shape).astype(np.int)
            for bbox, model in zip(parts_loc, cad_models):
                obj_id = args.cad_names.index(model) + 1
                # leave specific obj_id
                ID_id = (obj_id * (ID_step2 == obj_id)).astype(np.uint8)
                # handle (part5 or part6) region
                # part6 seg in part5 bbox is considered as part5, and vice versa
                # part5, part6 in part5 bbox && part6 bbox are labeled 5 + 6 = 11
                if obj_id in [5, 6]:
                    obj_id_similar = 5 if obj_id == 6 else 6
                    ID_id_similar = (obj_id * (ID_step2 == obj_id_similar)).astype(np.uint8)
                    ID_id += ID_id_similar
                # detection bbox mask
                x, y, w, h = bbox[0:4]
                x = int(x * _W / self.W)
                y = int(y * _H / self.H)
                w = int(w * _W / self.W)
                h = int(h * _H / self.H)
                mask_det_bbox = np.zeros_like(ID_id).astype(np.uint8)
                cv2.rectangle(mask_det_bbox, (x, y), (x+w, y+h), color=255, thickness=-1)
                # leave bbox region from specific ID map
                ID_id = cv2.bitwise_and(ID_id, ID_id, mask=mask_det_bbox)
                # add to template
                ID_step3 += ID_id
            ID_step3_color = np.zeros((_H, _W, 3)).astype(np.uint8)
            for h in range(_H):
                for w in range(_W):
                    ID_step3_color[h, w, :] = color(ID_step3[h, w])
            if args.opt.save_pose_prediction_maps:
                cv2.imwrite(save_path + '/STEP{}_ID_3.png'.format(step_num), ID_step3_color)            

            # PnP
            args.pose_return_dict[step_num] = []
            obj_ids = [args.cad_names.index(model) + 1 for model in cad_models]
            R_list = []
            parts_loc = args.parts_loc[step_num]
            for obj_id, bbox in zip(obj_ids, parts_loc):
                x, y, w, h = bbox[0:4]
                x = int(x * _W / self.W)
                y = int(y * _H / self.H)
                w = int(w * _W / self.W)
                h = int(h * _H / self.H)
                resized_bbox = [x, y, w, h]
                R, T = self.PnP(obj_id, resized_bbox, ID_step3, U, V, W, args.K)
                R_list.append(R)
                RT = np.concatenate((R, T), axis=1)
                args.pose_return_dict[step_num].append(RT)

            modifier_matrix = np.array([1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1]).reshape(3, 4)

            # Post Process 1 : Axis flip (step1)
            if len(set(obj_ids) - {2, 3, 7, 8}) == 0:
                for idx in range(len(args.pose_return_dict[step_num])):
                    z_direction = R_list[idx] @ np.array([0, 0, 1]).reshape(-1, 1)
                    z_direction = z_direction[1, 0]
                    if z_direction > 0: # z_axis heading down
                        args.pose_return_dict[step_num][idx] *= modifier_matrix

            # Post Process 1 : Axis flip (!step1)
            R_base = None # If part5 or part6 is detected, set their pose as base_pose
            if 5 in obj_ids:
                R_base = R_list[obj_ids.index(5)]
            if 6 in obj_ids:
                R_base = R_list[obj_ids.index(6)]
            if R_base is not None:
                def x_axis_angle_diff(R1, R2):
                    if np.array_equal(R1, R2):
                        return 0.0
                    else:
                        x1 = R1 @ np.array([1, 0, 0])
                        x2 = R2 @ np.array([1, 0, 0])
                        unit_vector_x1 = x1 / np.linalg.norm(x1)
                        unit_vector_x2 = x2 / np.linalg.norm(x2)
                        dot_product = np.dot(unit_vector_x1, unit_vector_x2)
                        angle_diff = np.rad2deg(np.arccos(dot_product))
                        return angle_diff
                def y_axis_angle_diff(R1, R2):
                    if np.array_equal(R1, R2):
                        return 0.0
                    else:
                        y1 = R1 @ np.array([0, 0, 1])
                        y2 = R2 @ np.array([0, 0, 1])
                        unit_vector_y1 = y1 / np.linalg.norm(y1)
                        unit_vector_y2 = y2 / np.linalg.norm(y2)
                        dot_product = np.dot(unit_vector_y1, unit_vector_y2)
                        angle_diff = np.rad2deg(np.arccos(dot_product))
                        return angle_diff
                def z_axis_angle_diff(R1, R2):
                    if np.array_equal(R1, R2):
                        return 0.0
                    else:
                        z1 = R1 @ np.array([0, 0, 1])
                        z2 = R2 @ np.array([0, 0, 1])
                        unit_vector_z1 = z1 / np.linalg.norm(z1)
                        unit_vector_z2 = z2 / np.linalg.norm(z2)
                        dot_product = np.dot(unit_vector_z1, unit_vector_z2)
                        angle_diff = np.rad2deg(np.arccos(dot_product))
                        return angle_diff
                x_axis_angle_diff_list = [x_axis_angle_diff(R_base, R) for R in R_list]
                y_axis_angle_diff_list = [y_axis_angle_diff(R_base, R) for R in R_list]
                z_axis_angle_diff_list = [z_axis_angle_diff(R_base, R) for R in R_list]
                if args.opt.print_pose_flip:
                    print('obj_ids :', obj_ids)
                    print('x_axis_angle_diff_list:', x_axis_angle_diff_list)
                    print('y_axis_angle_diff_list:', y_axis_angle_diff_list)
                    print('z_axis_angle_diff_list:', z_axis_angle_diff_list)

                def need_axis_flip(idx, obj_id, axis):
                    if obj_id in [5, 6]: # skip part5, part6
                        return 0
                    else:
                        if axis == 'x':
                            angle_diff_list = x_axis_angle_diff_list
                        elif axis == 'y':
                            angle_diff_list = y_axis_angle_diff_list
                        elif axis == 'z':
                            angle_diff_list = z_axis_angle_diff_list
                        angle_diff = angle_diff_list[idx]
                        if np.abs(angle_diff - 45) < 20 or np.abs(angle_diff - 135) < 20:
                            return 1
                        else:
                            return 0

                x_axis_need_flip = [need_axis_flip(idx, obj_id, 'x') for idx, obj_id in enumerate(obj_ids)]
                y_axis_need_flip = [need_axis_flip(idx, obj_id, 'y') for idx, obj_id in enumerate(obj_ids)]
                z_axis_need_flip = [need_axis_flip(idx, obj_id, 'z') for idx, obj_id in enumerate(obj_ids)]
                if args.opt.print_pose_flip:
                    print('x_axis_need_flip:', x_axis_need_flip)
                    print('y_axis_need_flip:', y_axis_need_flip)
                    print('z_axis_need_flip:', z_axis_need_flip)

                for idx in range(len(args.pose_return_dict[step_num])):
                    if x_axis_need_flip[idx] == 1 or y_axis_need_flip[idx] == 1 or z_axis_need_flip[idx] == 1:
                        args.pose_return_dict[step_num][idx] *= modifier_matrix
                        if args.opt.print_pose_flip:
                            print('flipped axis of part{}'.format(obj_ids[idx])) 
            
            if args.opt.use_part4_direction_estimation:
                # Post Process 2 : Flip Part4
                part_images = []
                part_images_torch = []
                H, W = args.steps[step_num].shape[:2]
                for i, crop_region in enumerate(args.parts_loc[step_num]):
                    x, y, w, h = crop_region[:4]
                    margin = 0
                    t = max(0, y - margin)
                    b = min(H, y + h + margin)
                    l = max(0, x - margin)
                    r = min(W, x + w + margin)
                    part_image = args.steps[step_num][t:b,l:r]
                    part_image = cv2.resize(part_image, (224, 224))
                    part_images.append(part_image)

                pred_directions = args.part4dir_model.test(part_images)
                cad_models = args.cad_models[step_num]
                obj_ids = [args.cad_names.index(model) + 1 for model in cad_models]
                directions = []
                for idx in range(len(args.pose_return_dict[step_num])):
                    RT = np.vstack((args.pose_return_dict[step_num][idx], np.array([0, 0, 0, 1])))
                    t = RT[:3, 3].reshape(3, 1)
                    z_h = np.array([0, 0, 1, 1]).reshape(4, 1)
                    z_im = RT.dot(z_h)
                    d = z_im[2] - t[2]
                    if d >= 0:
                        direction = 1
                    else:
                        direction = 0
                    directions.append(direction)

                if args.opt.print_pose_flip:
                    print('obj_ids:', obj_ids)
                    print('directions:', directions)
                    print('pred_directions:', pred_directions)

                for idx in range(len(args.pose_return_dict[step_num])):
                    obj_id = obj_ids[idx]
                    pred_direction = pred_directions[idx]
                    direction = directions[idx]
                    if obj_id == 4:
                        if pred_direction != direction:
                            R = args.pose_return_dict[step_num][idx][:3, :3]
                            modifier_matrix_1 = np.array([-1, 0, 0, 0, 1, 0, 0, 0, -1]).reshape(3, 3)
                            R = R @ modifier_matrix_1
                            args.pose_return_dict[step_num][idx][:3, :3] = R
                            if args.opt.print_pose_flip:
                                print('flipped part4 : pred direction {}, direction {}'.format(pred_direction, direction))

                args.part4dir_model.save_part4dir(step_num, directions, pred_directions, part_images)                       


    def PnP(self, obj_id, resized_bbox, ID, U, V, W, K):
        args = self.args
        # leave only region of object of interest
        mask = (255 * (ID == obj_id)).astype(np.uint8)
        x, y, w, h = resized_bbox
        mask_det_bbox = np.zeros_like(ID).astype(np.uint8)
        cv2.rectangle(mask_det_bbox, (x, y), (x+w, y+h), color=255, thickness=-1)
        mask = cv2.bitwise_and(mask, mask, mask=mask_det_bbox)
        # handle (part5 or part6) region
        if obj_id in [5, 6]:
            obj_id_sim = 5 + 6
            mask_sim = (255 * (ID == obj_id_sim)).astype(np.uint8)
            mask += mask_sim
        _H, _W = U.shape
        U = cv2.bitwise_and(U, U, mask=mask)
        V = cv2.bitwise_and(V, V, mask=mask)
        W = cv2.bitwise_and(W, W, mask=mask)
        U = U.flatten()
        V = V.flatten()
        W = W.flatten()
        correspondence_dict = self.correspondence_dicts[obj_id - 1]
        mapping_2d = []
        mapping_3d = []
        YX = []
        for y in range(_H):
            for x in range(_W):
                y_scaled = int(y * self.H / _H)
                x_scaled = int(x * self.W / _W)
                YX.append((x_scaled, y_scaled))
        for (u, v, w, yx) in zip(U, V, W, YX):
            if (u, v, w) in correspondence_dict and (u, v, w) != (0, 0, 0):
                y, x = yx
                mapping_2d.append([y, x])
                mapping_3d.append(correspondence_dict[(u, v, w)])

        mapping_2d = np.array(mapping_2d).astype(np.float32)
        mapping_3d = np.array(mapping_3d).astype(np.float32)
        if len(mapping_2d) >= 6 or len(mapping_3d) > 6:
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(mapping_3d, mapping_2d, K, distCoeffs=None,
                iterationsCount=300, reprojectionError=1.2, flags=1)
            ''' flags
            0 : iterative
            1 : epnp
            2 : p3p
            5 : ap3p
            '''
            rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
            return rot, tvecs
        else:
            return np.eye(3), np.array([0, 0, 25]).reshape(3, 1)


    def visualize(self, args, step_num):
        color = Color()
        if args.opt.save_pose_visualization:
            step_img = args.steps[step_num].copy()
            H, W, _ = step_img.shape
            cad_models = args.cad_models[step_num]
            obj_ids = [int(x.replace('part', '')) for x in cad_models]
            poses = args.pose_return_dict[step_num]
            K = args.K
            for obj_id, RT in zip(obj_ids, poses):
                obj_idx = obj_id - 1
                pts_color = color(obj_id)
                R = RT[:,:3]
                if np.all(R == np.eye(3)):
                    continue
                T = RT[:,3].reshape(3, 1)
                pts = self.point_clouds[obj_idx]

                # project points
                projected_2d_pts = self.project_pts(pts, K, R, T).astype(int)
                projected_2d_pts[:, 0] = np.clip(projected_2d_pts[:, 0], 0, W - 1)
                projected_2d_pts[:, 1] = np.clip(projected_2d_pts[:, 1], 0, H - 1)
                for pt in projected_2d_pts:
                    cv2.circle(step_img, (pt[0], pt[1]), 7, pts_color, -1)
                
                # draw axes
                arrow_length = 1
                axis_pts = np.array([[0 ,0, 0], [arrow_length, 0, 0], [0, arrow_length, 0], [0, 0, arrow_length]])
                projected_2d_pts = self.project_pts(axis_pts, K, R, T).astype(int)
                projected_2d_pts[:, 0] = np.clip(projected_2d_pts[:, 0], 0, W - 1)
                projected_2d_pts[:, 1] = np.clip(projected_2d_pts[:, 1], 0, H - 1)
                
                center = projected_2d_pts[0]
                xaxis = projected_2d_pts[1]
                yaxis = projected_2d_pts[2]
                zaxis = projected_2d_pts[3]
                cv2.arrowedLine(step_img, tuple(center), tuple(xaxis), (0, 0, 0), 12)
                cv2.arrowedLine(step_img, tuple(center), tuple(xaxis), (0, 0, 255), 8)
                cv2.arrowedLine(step_img, tuple(center), tuple(yaxis), (0, 0, 0), 12)
                cv2.arrowedLine(step_img, tuple(center), tuple(yaxis), (0, 255, 255), 8)
                cv2.arrowedLine(step_img, tuple(center), tuple(zaxis), (0, 0, 0), 12)
                cv2.arrowedLine(step_img, tuple(center), tuple(zaxis), (153, 0, 0), 8)
            cv2.imwrite(args.opt.initial_pose_estimation_visualization_path + '/STEP_{}.png'.format(step_num), step_img)        



    def project_pts(self, pts, K, R, t):
      """Projects 3D points.

      :param pts: nx3 ndarray with the 3D points.
      :param K: 3x3 ndarray with an intrinsic camera matrix.
      :param R: 3x3 ndarray with a rotation matrix.
      :param t: 3x1 ndarray with a translation vector.
      :return: nx2 ndarray with 2D image coordinates of the projections.
      """
      assert (pts.shape[1] == 3)
      P = K.dot(np.hstack((R, t)))
      pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
      pts_im = P.dot(pts_h.T)
      pts_im /= pts_im[2, :]
      return pts_im[:2, :].T


    def save_part_id_pose(self, args, step_num, matched_poses):
        if args.opt.save_part_id_pose:
            cad_models = args.cad_models[step_num]
            part_imgs_bboxed = args.parts_bboxed[step_num]
            for i, (matched_pose, cad_model, part_img_bboxed) in enumerate(zip(matched_poses, cad_models, part_imgs_bboxed)):
                plt.imsave(args.opt.part_id_pose_path + '/STEP{}_part{}_bbox'.format(step_num, i), part_img_bboxed)
                plt.clf()
                plt.figure(figsize=(4, 4))
                plt.axis('off')
                pose_img = args.VIEW_IMGS[args.cad_names.index(cad_model)][matched_pose]
                plt.imshow(pose_img)
                plt.title('pred cad : {}\npred pose : {}'.format(cad_model, matched_pose))
                if not os.path.exists(args.opt.part_id_pose_path):
                    os.makedirs(args.opt.part_id_pose_path)
                plt.savefig(args.opt.part_id_pose_path + '/STEP{}_part{}_pose'.format(step_num, i))
                plt.close()
    '''

    def save_part_id_pose(self, args, step_num, matched_poses):
        if args.opt.save_part_id_pose:
            cad_models = args.cad_models[step_num]
            part_imgs = args.parts[step_num]
            for i, (matched_pose, cad_model, part_img) in enumerate(zip(matched_poses, cad_models, part_imgs)):
                plt.clf()
                fig, ax = plt.subplots(1, 2, sharey=True)
                part_img = self.resize_and_pad(part_img)
                ax[0].imshow(part_img)
                ax[0].set_title('detection result')
                pose_img = args.VIEW_IMGS[args.cad_names.index(cad_model)][matched_pose]
                ax[1].imshow(pose_img)
                ax[1].set_title('pred cad : {}\npred pose : {}'.format(cad_model, matched_pose))
                if not os.path.exists(args.opt.part_id_pose_path):
                    os.makedirs(args.opt.part_id_pose_path)
                plt.savefig(args.opt.part_id_pose_path + '/STEP{}_part{}'.format(step_num, i))
                plt.close()

    
    def resize_and_pad(self, img, a=150):
        # find object region
        non_zero = np.nonzero(255 - img)
        y_min = np.min(non_zero[0])
        y_max = np.max(non_zero[0])
        x_min = np.min(non_zero[1])
        x_max = np.max(non_zero[1])
        img = img[y_min:y_max + 1, x_min:x_max + 1]
        # resize to 150, 150
        long_side = np.max(img.shape)
        ratio = a / long_side
        img = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio, interpolation = cv2.INTER_AREA)
        # pad to 224, 224
        pad_left = int(np.ceil((224 - img.shape[1]) / 2))
        pad_right = int(np.floor((224 - img.shape[1]) / 2))
        pad_top = int(np.ceil((224 - img.shape[0]) / 2))
        pad_bottom = int(np.floor((224 - img.shape[0]) / 2))
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, None, [255, 255, 255])
        return img
    '''
    
