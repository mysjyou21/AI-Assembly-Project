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
from ..utils.pose_gt import *

sys.path.append('./function/utilities')
from function.utilities.utils import *

class Color():
    def __init__(self):
        """color palette (BGR) """
        self.colors = {
            0 : (0, 0, 0),
            1 : (0, 0, 255),
            2 : (0, 165, 255),
            3 : (0, 255, 255),
            4 : (0, 128, 0),
            5 : (255, 0, 0),
            6 : (130, 0, 75),
            7 : (255, 0, 255),
            8 : (255, 255, 0),
        }
    def __call__(self, x):
        return self.colors[x]


class InitialPoseEstimation():
    def __init__(self, args):
        self.args = args.opt
        self.checkpoint_path = self.args.pose_model_path
        self.pose_function_path = './function/Pose'
        self.pose_data_path = self.pose_function_path + '/data'

        # load weights
        with args.graph_pose.as_default():
            sess = args.sess_pose
            self.model = CorrespondenceBlockModel(3, 9, 256) # in channels, id channels, uvw channels
            if args.mission1:
                print('POSE MODEL : Loading saved model from', self.checkpoint_path + '/correspondence_block_stefan.pt')
                checkpoint = torch.load(self.checkpoint_path + '/correspondence_block_stefan.pt', map_location='cuda:0')
            else:
                print('POSE MODEL : Loading saved model from', self.checkpoint_path + '/correspondence_block.pt')
                checkpoint = torch.load(self.checkpoint_path + '/correspondence_block.pt', map_location='cuda:0')
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
        refresh_folder(args.opt.initial_pose_estimation_visualization_separate_path)
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

            _H, _W = idmask_pred.shape
            ID_color = np.zeros((_H, _W, 3)).astype(np.uint8)
            for h in range(_H):
                for w in range(_W):
                    ID_color[h, w, :] = color(idmask_pred[h, w])

            ID = idmask_pred.astype(np.uint8)
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

            # post-process ID-map with detection classes (part2, part3, part7, part8)
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

            ID_color = np.zeros((_H, _W, 3)).astype(np.uint8)
            for h in range(_H):
                for w in range(_W):
                    ID_color[h, w, :] = color(ID[h, w])

            if args.opt.save_pose_prediction_maps:
                cv2.imwrite(save_path + '/STEP{}_ID_1.png'.format(step_num), ID_color)
            
            # post-process ID-map with detection bbox
            parts_loc = args.parts_loc[step_num]
            cad_models = args.cad_models[step_num]
            ID_new = np.zeros(ID.shape).astype(np.int)
            for bbox, model in zip(parts_loc, cad_models):
                obj_id = args.cad_names.index(model) + 1
                
                mask_id = (255 * (ID == obj_id)).astype(np.uint8)
                ID_id = (obj_id * (ID == obj_id)).astype(np.uint8)
                x, y, w, h = bbox[0:4]
                x = int(x * _W / self.W)
                y = int(y * _H / self.H)
                w = int(w * _W / self.W)
                h = int(h * _H / self.H)
                mask_det = np.zeros_like(mask_id).astype(np.uint8)
                cv2.rectangle(mask_det, (x, y), (x+w, y+h), color=255, thickness=-1)
                mask = cv2.bitwise_and(mask_id, mask_id, mask=mask_det)
                mask_inv = cv2.bitwise_not(mask)
                ID_new = cv2.bitwise_and(ID_new, ID_new, mask=mask_inv) # erase mask region
                ID_new += cv2.bitwise_and(ID_id, ID_id, mask=mask) # fill mask region
            ID_new_color = np.zeros((_H, _W, 3)).astype(np.uint8)
            for h in range(_H):
                for w in range(_W):
                    ID_new_color[h, w, :] = color(ID_new[h, w])
            if args.opt.save_pose_prediction_maps:
                cv2.imwrite(save_path + '/STEP{}_ID_2.png'.format(step_num), ID_new_color)

            # if no New part is found, infer from pose segmentation map
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
                if '104322' in grouped_serial_OCR:
                    # long screw
                    new_part_should_be_found_condition

            if new_part_should_be_found_condition:
                if not len(set(args.used_parts[step_num]) & set(args.unused_parts[step_num])): # no New part found

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
                    if max_pixels < 50:
                        # too small
                        max_pixels = 0
                        max_pixels_look_up_part_id = 0

                    # choose largest component of that part id
                    index = np.argmax(stats[max_pixels_look_up_part_id][:, -1])

                    # add to ID_new
                    ID_part_to_add = np.where(indexed_maps[max_pixels_look_up_part_id] == index, max_pixels_look_up_part_id, 0).astype(np.uint8)
                    ID_part_to_add_color = np.zeros((_H, _W, 3)).astype(np.uint8)
                    for h in range(_H):
                        for w in range(_W):
                            ID_part_to_add_color[h, w, :] = color(ID_part_to_add[h, w])
                    ID_new_modified = ID_new.astype(np.int) + ID_part_to_add.astype(np.int)
                    assert np.any(ID_new_modified <= 255), 'ID_new_modified pixels values exceed 255'
                    ID_new_modified = ID_new_modified.astype(np.uint8)
                    ID_new_modified_color = ID_new_color + ID_part_to_add_color
                    if args.opt.save_pose_prediction_maps:
                        cv2.imwrite(save_path + '/STEP{}_ID_3.png'.format(step_num), ID_new_modified_color)
                    print('Added "New" at step {} : part{} (part0 means none added)'.format(step_num, max_pixels_look_up_part_id)) #blue
                    # print('pixel counts : {}'.format(max_pixels)) #blue

                    # update self.used_parst, self.unused_parts, self.cad_models
                    if max_pixels_look_up_part_id != 0:
                        args.used_parts[step_num].append(max_pixels_look_up_part_id)
                        args.used_parts[step_num].sort()
                        args.unused_parts[step_num + 1].remove(max_pixels_look_up_part_id)
                        args.cad_models[step_num].append('part' + str(max_pixels_look_up_part_id))


            # apply to ID
            if max_pixels == 0:
                ID = ID_new
            else:
                ID = ID_new_modified

            # PnP
            args.pose_save_dict['000000'][str(step_num).zfill(6)] = []
            args.pose_return_dict[step_num] = []

            obj_ids = [args.cad_names.index(model) + 1 for model in cad_models]
            for obj_id in obj_ids:
                R, T = self.PnP(obj_id, ID, U, V, W, args.K)
                RT = np.concatenate((R, T), axis=1)
                args.pose_return_dict[step_num].append(RT)
                temp_dict = {}
                temp_dict["cam_R_m2c"] = R.flatten().tolist()
                temp_dict["cam_t_m2c"] = T.flatten().tolist()
                temp_dict["obj_id"] = obj_id
                args.pose_save_dict['000000'][str(step_num).zfill(6)].append(temp_dict)
            if not os.path.exists(os.path.dirname(args.opt.initial_pose_estimation_adr)):
                os.makedirs(os.path.dirname(args.opt.initial_pose_estimation_adr))
            with open(args.opt.initial_pose_estimation_adr, 'w') as f:
                json.dump(args.pose_save_dict, f, indent=2, sort_keys=True)


    def PnP(self, obj_id, ID, U, V, W, K):
        args = self.args
        # leave only region of object of interest
        mask = (255 * (ID == obj_id)).astype(np.uint8)
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
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(mapping_3d, mapping_2d, K, distCoeffs=None, iterationsCount=300, reprojectionError=1.2, flags=cv2.SOLVEPNP_EPNP)
            rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
            return rot, tvecs
        else:
            return np.eye(3), np.array([0, 0, 25]).reshape(3, 1)


    def visualize(self, args, step_num, save_sep=True):
        color = Color()
        save_sep = args.opt.save_pose_visualization_separate
        if args.opt.save_pose_visualization:
            step_img = args.steps[step_num].copy()
            if save_sep:
                step_imgs = []
                for i in range(len(args.cad_names)):
                    step_imgs.append(step_img.copy())
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
                if save_sep:
                    for pt in projected_2d_pts:
                        cv2.circle(step_imgs[obj_idx], (pt[0], pt[1]), 7, pts_color, -1)
                
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
                if save_sep:
                    cv2.arrowedLine(step_imgs[obj_idx], tuple(center), tuple(xaxis), (0, 0, 0), 12)
                    cv2.arrowedLine(step_imgs[obj_idx], tuple(center), tuple(xaxis), (0, 0, 255), 8)
                    cv2.arrowedLine(step_imgs[obj_idx], tuple(center), tuple(yaxis), (0, 0, 0), 12)
                    cv2.arrowedLine(step_imgs[obj_idx], tuple(center), tuple(yaxis), (0, 255, 255), 8)
                    cv2.arrowedLine(step_imgs[obj_idx], tuple(center), tuple(zaxis), (0, 0, 0), 12)
                    cv2.arrowedLine(step_imgs[obj_idx], tuple(center), tuple(zaxis), (153, 0, 0), 8)
            cv2.imwrite(args.opt.initial_pose_estimation_visualization_path + '/STEP_{}.png'.format(step_num), step_img)
            if save_sep:
                template = []
                for i in range(len(args.cad_names)):
                    step_img_sep = step_imgs[i]
                    template = np.concatenate((template, step_img_sep), axis=1) if len(template) else step_img_sep
                cv2.imwrite(args.opt.initial_pose_estimation_visualization_separate_path + '/STEP_{}.png'.format(step_num), template)
        



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
        cad_models = args.cad_models[step_num]
        part_imgs = args.parts_bboxed[step_num]
        for i, (matched_pose, cad_model, part_img) in enumerate(zip(matched_poses, cad_models, part_imgs)):
            plt.imsave(args.opt.part_id_pose_path + '/STEP{}_part{}_bbox'.format(step_num, i), part_img)
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
    
