import sys
sys.path.append('./function/frcnn')
import tensorflow as tf

import os
import cv2
import numpy as np
import pickle
from optparse import OptionParser
import time
import imageio

from keras import backend as K
from keras.layers import Input
from keras.models import Model

from keras_frcnn import config
from keras_frcnn import roi_helpers


class DetectionModel():

    def __init__(self, det_config_name, graph_detect, sess_detect):
        # Load detection model.
        self.det_config_name = det_config_name
        self.graph_detect = graph_detect
        self.sess_detect = sess_detect

        with open(self.det_config_name, 'rb') as f:
            self.C = pickle.load(f)

        if self.C.network == 'vgg':
            import keras_frcnn.vgg as nn
            num_features = 512
        elif self.C.network == 'resnet50':
            import keras_frcnn.vgg as nn
            num_features = 1024

        # test 시에는 모든 data augmentation들 중지
        self.C.use_horizontal_flips = False
        self.C.use_vertical_flips = False
        self.C.rot_90 = False
        self.C.use_scaling = False

        # class_mapping에 bg(background) 없는 경우 추가해줌
        if 'bg' not in self.C.class_mapping:
            self.C.class_mapping['bg'] = len(self.C.class_mapping)
        self.C.class_mapping = {v: k for k, v in self.C.class_mapping.items()}
        self.class_to_color = {self.C.class_mapping[v]: np.random.randint(0, 255, 3) for v in self.C.class_mapping}

        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)

        with self.graph_detect.as_default():
            with self.sess_detect.as_default():
                img_input = Input(shape=input_shape_img)
                roi_input = Input(shape=(self.C.num_rois, 4))
                feature_map_input = Input(input_shape_features)

                # define the base network
                shared_layers = nn.nn_base(img_input, trainable=True)

                # define the RPN, built on the base layers
                num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
                rpn_layers = nn.rpn(shared_layers, num_anchors)

                classifier = nn.classifier(feature_map_input, roi_input, self.C.num_rois, nb_classes=len(self.C.class_mapping), trainable=True)

                self.model_rpn = Model(img_input, rpn_layers)
                self.model_classifier_only = Model([feature_map_input, roi_input], classifier)
                self.model_classifier = Model([feature_map_input, roi_input], classifier)

                print('DETECTION MODEL : Loading weights from {}'.format(self.C.model_path))
                self.model_rpn.load_weights(self.C.model_path, by_name=True)
                self.model_classifier.load_weights(self.C.model_path, by_name=True)

    def test_for_components(self, img):

        bbox_threshold = 0.8

        X, ratio = self.format_img(img)
        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN

        with self.graph_detect.as_default():
            with self.sess_detect.as_default():
                [Y1, Y2, F] = self.model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, self.C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1, y1, x2, y2) to (x, y, w, h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // self.C.num_rois + 1):
            ROIs = np.expand_dims(R[self.C.num_rois * jk : self.C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                 break

            if jk == R.shape[0] // self.C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded
            with self.graph_detect.as_default():
                with self.sess_detect.as_default():
                    [P_cls, P_regr] = self.model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):
                cls_name = self.C.class_mapping[np.argmax(P_cls[0, ii, :])]
                ###########################
                # bbox_threshold = 0.94
                ###########################
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num : 4 * (cls_num + 1)]
                    tx /= self.C.classifier_regr_std[0]
                    ty /= self.C.classifier_regr_std[1]
                    tw /= self.C.classifier_regr_std[2]
                    th /= self.C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [self.C.rpn_stride * x, self.C.rpn_stride * y, self.C.rpn_stride * (x + w), self.C.rpn_stride * (y + h)]
                )
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        components_dict = {}

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.4)

            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]

                (real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)
                (real_x, real_y, real_w, real_h) = real_x1, real_y1, real_x2 - real_x1, real_y2 - real_y1

                if key not in components_dict:
                    components_dict[key] = []
                # if key not in confidences_dict:
                #     confidences_dict[key] = []

                components_dict[key].append([real_x, real_y, real_w, real_h, new_probs[jk]])
                # confidences_dict[key].append(new_probs[jk])

        for key in self.C.class_mapping.values():
            if key not in components_dict:
                components_dict[key] = []
            # if key not in confidences_dict:
            #     confidences_dict[key] = []

        return components_dict

    def test_for_parts(self, img, parts_threshold, step_num=None):

        bbox_threshold = parts_threshold

        X, ratio = self.format_img(img)
        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN

        with self.graph_detect.as_default():
            with self.sess_detect.as_default():
                [Y1, Y2, F] = self.model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, self.C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1, y1, x2, y2) to (x, y, w, h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        for key in self.C.class_mapping.values():
            if key not in bboxes:
                bboxes[key] = []
                probs[key] = []

        for jk in range(R.shape[0] // self.C.num_rois + 1):
            ROIs = np.expand_dims(R[self.C.num_rois * jk : self.C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                 break

            if jk == R.shape[0] // self.C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded
            with self.graph_detect.as_default():
                with self.sess_detect.as_default():
                    [P_cls, P_regr] = self.model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):
                cls_name = self.C.class_mapping[np.argmax(P_cls[0, ii, :])]
                ###########################
                # bbox_threshold = 0.94
                ###########################
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num : 4 * (cls_num + 1)]
                    tx /= self.C.classifier_regr_std[0]
                    ty /= self.C.classifier_regr_std[1]
                    tw /= self.C.classifier_regr_std[2]
                    th /= self.C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [self.C.rpn_stride * x, self.C.rpn_stride * y, self.C.rpn_stride * (x + w), self.C.rpn_stride * (y + h)]
                )
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        components_dict = {}

        for key in self.C.class_mapping.values():
            if key not in components_dict:
                components_dict[key] = []

        # part5, part6 을 제외한 부품들은 무조건 하나만 남김
        if step_num is not None:
            for key in ['1', '2', '3', '4', '7', '8']:
                if key not in bboxes:
                    continue
                bbox = np.array(bboxes[key])
                new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=-1)
                # step4: [[160, 448, 304, 576]]
                # step5: [[192, 464, 336, 592]]
                ##### Temp #######
#                if key == '4' and step_num == 4:
#                    new_boxes = np.asarray([[160, 448, 304, 576]])
#                    new_probs = np.asarray([0.999])
                if key == '4' and step_num == 4:
                    new_boxes = np.asarray([[192, 464, 336, 592]])
                    new_probs = np.asarray([0.999])
                ##################
                for jk in range(len(new_boxes)):
                    (x1, y1, x2, y2) = new_boxes[jk, :]

                    (real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)
                    (real_x, real_y, real_w, real_h) = real_x1, real_y1, real_x2 - real_x1, real_y2 - real_y1

                    components_dict[key].append([real_x, real_y, real_w, real_h, new_probs[jk]])
        else:
            for key in ['1', '2', '3', '4', '7', '8']:
                if key not in bboxes:
                    continue
                bbox = np.array(bboxes[key])
                new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0)
                if len(new_boxes) >= 2:
                    new_boxes = new_boxes[:2]
                    new_probs = new_probs[:2]

                for jk in range(len(new_boxes)):
                    (x1, y1, x2, y2) = new_boxes[jk, :]

                    (real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)
                    (real_x, real_y, real_w, real_h) = real_x1, real_y1, real_x2 - real_x1, real_y2 - real_y1

                    components_dict[key].append([real_x, real_y, real_w, real_h, new_probs[jk]])

        # part5, part6의 경우, 일단 하나로 합치고
        bbox_56 = []
        prob_56 = []
        for key in ['5', '6']:
            if key not in bboxes:
                continue
            bbox = bboxes[key]
            bbox_56 += bbox
            prob_56 += probs[key]

        # 그다음 overlap_thresh = 0으로 NMS 수행 (iou가 0, 즉 겹치지 않는 bbox들은 우선 살아남게)
        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(np.array(bbox_56), np.array(prob_56), overlap_thresh=0.4)
        if len(new_boxes) == 0:
            pass
        elif len(new_boxes) == 1:
            if new_boxes[0].tolist() in bboxes['5']:
                key = '5'
            else:   # i.e., new_boxes[0].tolist() in bboxes['6']
                key = '6'
            (x1, y1, x2, y2) = new_boxes[0, :]
            (real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)
            (real_x, real_y, real_w, real_h) = real_x1, real_y1, real_x2 - real_x1, real_y2 - real_y1
            components_dict[key].append([real_x, real_y, real_w, real_h, new_probs[0]])
        else:   # len(new_boxes) >= 2인 경우
            # confidence 기준으로 top 2 만 남김
            idxs = np.argsort(new_probs)[::-1]
            new_boxes = new_boxes[idxs][:2]
            new_probs = new_probs[idxs][:2]
            # keys: 남은 두 bbox 각각의 part id
            part_ids_temp = []
            part_ids_temp.append('5') if new_boxes[0].tolist() in bboxes['5'] else part_ids_temp.append('6')
            part_ids_temp.append('5') if new_boxes[1].tolist() in bboxes['5'] else part_ids_temp.append('6')
            # case1) part_ids 둘 다 part5 인 경우: confidence 더 낮은 건 part6으로.
            if part_ids_temp == ['5', '5']:
                part_ids = ['5', '6']
            # case2) part_ids 둘 다 part6 인 경우: confidence 더 낮은 건 part5로.
            elif part_ids_temp == ['6', '6']:
                part_ids = ['6', '5']
            # case3) part_ids가 ['5', '6'] 또는 ['6', '5']인 경우, 그대로 사용
            else:
                part_ids = part_ids_temp

            # sample2 step4에서 part5, part6 바뀐 경우 후처리
            if step_num == 4:
                if new_boxes[0, 1] < new_boxes[1, 1]:
                    part_ids = ['5', '6']
                else:
                    part_ids = ['6', '5']

            for jk, key in enumerate(part_ids):
                (x1, y1, x2, y2) = new_boxes[jk, :]

                (real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)
                (real_x, real_y, real_w, real_h) = real_x1, real_y1, real_x2 - real_x1, real_y2 - real_y1
                components_dict[key].append([real_x, real_y, real_w, real_h, new_probs[jk]])

        return components_dict

    def format_img_size(self, img):
        ## formats the image size based on config
        img_min_side = float(self.C.im_size)
        (height, width, _) = img.shape

        if width <= height:
            ratio = img_min_side / width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side / height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio

    def format_img_channels(self, img):
        ## formats the image channels based on config
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= self.C.img_channel_mean[0]
        img[:, :, 1] -= self.C.img_channel_mean[1]
        img[:, :, 2] -= self.C.img_channel_mean[2]
        img /= self.C.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def format_img(self, img):
        ## formats an image for model prediction based on config
        img, ratio = self.format_img_size(img)
        img = self.format_img_channels(img)
        return img, ratio

    def get_real_coordinates(self, ratio, x1, y1, x2, y2):
        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))
        return real_x1, real_y1, real_x2, real_y2


# os.chdir('..')
# img = cv2.imread('./input/stefan/cuts/stefan-03.png')
# config_filename1 = './model/detection/fine_tuned/12.pickle'
# config_filename2 = './model/detection/fine_tuned/3.pickle'
# model1 = DetectionModel(config_filename1)
# model2 = DetectionModel(config_filename2)
# components_dict1 = model1.test(img)
# components_dict2 = model2.test(img)
# print('')