import sys
import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from function.utils import print_time
from function.pose.pose_net import POSE_NET
from function.retrieval.codes.DCA import DCA
from function.detection.DetectionModel import DetectionModel
from function.detection.pascalvoc import pascalvoc
from function.test.pascalvoc import pascalvoc as pascalvoc_test
from function.test.pascalvoc_retrieval import pascalvoc as pascalvoc_retrieval
from function.pose.pose_utils import manage_duplicate_pose


class QuantReportPose():

    def __init__(self, args):
        self.args = args
        self.mode = args.mode

        # Load models
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        # 선지
        if args.mode in ['detection_unit_test', 'detection', 'retrieval', 'pose', 'test']:
            self.graph_detection = tf.Graph()
            with self.graph_detection.as_default():
                self.sess_detection = tf.Session(config=config)
                self.detection_model = DetectionModel(self.args, self.graph_detection, self.sess_detection)
        
        
        # 민우
        if args.mode in ['retrieval_unit_test', 'retrieval', 'pose', 'test']:
            self.graph_retrieval = tf.Graph()
            with self.graph_retrieval.as_default():
                self.sess_retrieval = tf.Session(config=config)
                self.retrieval_model = DCA(self) 

        # 이삭
        if args.mode in ['pose_unit_test', 'pose', 'test']:
            self.graph_pose = tf.Graph()
            with self.graph_pose.as_default():
                self.sess_pose = tf.Session(config=config)
                self.pose_model = POSE_NET(self)
        

        # Load data
        self.imgs, self.img_names, self.num_test_imgs, self.labels, self.cad_names = self.load_data()
        self.H, self.W, _ = self.imgs[0].shape

        # Results
        # WARINING : image index starts with 0
        # 선지
        self.detection_results = {} # (int) image index : [[x, y, w, h], ...]
        # 민우
        self.retrieval_results = {} # (int) image index : [cad_name, ...]
        # 이삭
        self.pose_results = {} # (int) image index : [pose_name, ...]


    def load_data(self):
        """load data from npy
        
        Returns:
            imgs : (uint8) [10, 1700, 1200, 3]
            num_test_imgs : (int) number of test images
            cad_names : (str) list of cad names

        """
        args = self.args

        # load image names
        files = sorted(glob.glob(args.image_path + '/*'))
        img_names = [os.path.basename(x).split('.')[0] for x in files]

        # load from npy
        with open(os.path.join(args.binary_path, 'test_data.npy'), 'rb') as f:
            imgs = np.load(f)
            print('loaded imgs : ', imgs.shape)
        num_test_imgs = len(imgs)

        # load labels
        labels = open(os.path.join(args.label_path, 'label.txt'), 'r').read().splitlines()

        # load cad names
        cad_adrs = sorted(glob.glob(args.cad_path + '/*'))
        cad_names = [os.path.splitext(os.path.basename(x))[0] for x in cad_adrs]

        return imgs, img_names, num_test_imgs, labels, cad_names


    def detection(self):
        """STEP1 : detection
        
        detect furniture parts in test images

        Update:
            self.detection_results = {}
                key : cad index
                value : [[x, y, w, h], ...]
        """
        # unit test
        if self.mode == 'detection_unit_test':
            detection_gt = self.ground_truth('detection')

        # do detection
        self.detection_results_withcls = {}

        for i in range(self.imgs.shape[0]):
            self.detection_results_withcls[i] = self.detection_model.test(self.imgs[i], i)
            temp = []
            for x in self.detection_results_withcls[i]['Mid']: temp.append(x)
            for x in self.detection_results_withcls[i]['New']: temp.append(x)
            temp = sorted(temp)
            self.detection_results[i] = temp
        if self.mode != 'test':
            pascalvoc(self.args).pascalvoc_calculate_iou()



    def retrieval(self): # 민우
        """STEP 2 : retrieval

        retrieve cad name from cropped images of detection results

        Update:
            self.retrieval_results = {}
                key : cad index
                value : [cad_name, ...]
        """
        # unit test
        if self.mode == 'retrieval_unit_test':
            self.detection_results = self.ground_truth('detection')
            retrieval_gt = self.ground_truth('retrieval')
            correct_num = 0
            for i in range(self.num_test_imgs):
                page_num = i
                whole_img = self.imgs[i]
                cropped_imgs = list()
                for x, y, w, h in self.detection_results[i]:
                    cropped_img = whole_img[y : y + h, x : x + w, :]
                    resized_img = self.retrieval_model.resize_and_pad(cropped_img)
                    cropped_img = np.expand_dims(resized_img, axis=0)
                    cropped_imgs = np.vstack((cropped_imgs, cropped_img)) if len(cropped_imgs) else cropped_img
                _, indices, correct_num_ = self.retrieval_model.test(self, page_num, cropped_imgs, retrieval_gt, self.cad_names)
                correct_num += correct_num_
                print("One Page Finished: {}".format(page_num))
            print('-------------------------')
            print('Accuracy : {}%'.format(correct_num / self.num_test_imgs * 10))
        else:
            retrieval_gt = []
            self.save_retrieval_gt()
            for i in range(self.num_test_imgs):
                page_num = i
                whole_img = self.imgs[i]
                cropped_imgs = list()
                for x, y, w, h in self.detection_results[i]:
                    cropped_img = whole_img[y : y + h, x : x + w, :]
                    resized_img = self.retrieval_model.resize_and_pad(cropped_img)
                    cropped_img = np.expand_dims(resized_img, axis=0)
                    cropped_img = np.expand_dims(resized_img, axis=0)
                    cropped_imgs = np.vstack((cropped_imgs, cropped_img)) if len(cropped_imgs) else cropped_img
                result_name, indices, _ = self.retrieval_model.test(self, page_num, cropped_imgs, retrieval_gt, self.cad_names)
                self.retrieval_results[i] = result_name
            self.save_retrieval_pred()
            _, __ = pascalvoc_retrieval(self.args).pascalvoc_calculate_iou()
            
    def pose(self): # 이삭
        """STEP 3 : pose
        
        get pose from detection, retrieval results

        Update:
            self.pose_results = {}
                key : cad index
                value : [pose, ...]
        """
        args = self.args

        # unit test
        if self.mode == 'pose_unit_test':
            self.detection_results = self.ground_truth('detection')
            self.retrieval_results = self.ground_truth('retrieval')
        self.pose_gt = self.ground_truth('pose')
        self.pose_results = self.pose_model.test()
        self.save_bbox()
        _, self.for_draw_bbox = pascalvoc_test(self.args).pascalvoc_calculate_iou()


    def output_visualization(self):
        """visualize test output """
        args = self.args
        for i in range(self.num_test_imgs):
            img = self.imgs[i].copy()
            img_name = self.img_names[i]
            detections = [x for x in self.for_draw_bbox if x[0] == img_name]
            for det in detections:
                class_name = det[1]
                cad_name, pose_idx = class_name.split('-')
                x, y, r, b = det[3]
                x, y, r, b = int(x), int(y), int(r), int(b)
                color = (0, 255, 0) if det[4] == 'TP' else (0, 0, 255)
                cv2.rectangle(img, (x, y), (r, b), color, 2)
                textLabel = 'class {}: {}'.format(cad_name, pose_idx)
                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (x, y - 0)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

            cv2.imwrite(os.path.join(args.output_path, img_name + '.png'), img)


    def save_test_data(self):
        """create and save test data in binary format
        
        Read test images and save them into binary format

        Saves:
            imgs : (uint8) [10, 1700, 1200, 3]
        
        """
        print('================')
        print(' SAVE TEST NPY')
        print('================')

        # read images
        img_adrs = sorted(glob.glob(self.args.image_path + '/*'))
        imgs = np.array([cv2.imread(img_adr) for img_adr in img_adrs])

        # save them as npy
        with open(os.path.join(self.args.binary_path, 'test_data.npy'), 'wb') as f:
            np.save(f, imgs)
            print('saved imgs : ', imgs.shape)

        # save ground truth labels for test map calculation
        self.detection_results = self.ground_truth('detection')
        self.retrieval_results = self.ground_truth('retrieval')
        self.pose_results = self.ground_truth('pose')
        self.new_mid_results = self.ground_truth('new_mid')

        # save ground_truth bbox answers
        self.save_detection_bbox_gt()
        self.save_bbox('gt')


    def ground_truth(self, type):
        detection_gt = {}
        retrieval_gt = {}
        pose_gt = {}
        new_mid_gt = {}
        
        NUM_CAD = 10

        index = 0
        for i in range(self.num_test_imgs):
            detection_gt[i] = []
            retrieval_gt[i] = []
            pose_gt[i] = []
            new_mid_gt[i] = []
            for j in range(NUM_CAD):
                _, x, y, w, h, new_mid, cad, pose = self.labels[index].split(',')
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                detection_gt[i].append([x, y, w, h])
                retrieval_gt[i].append(cad)
                pose_gt[i].append(int(pose))
                new_mid_gt[i].append(new_mid)
                index += 1

        if type == 'detection':
            return detection_gt
        if type == 'retrieval':
            return retrieval_gt
        if type == 'pose':
            return pose_gt
        if type == 'new_mid':
            return new_mid_gt


    def save_bbox(self, type='det'):
        """saves result dictionaries inorder to compute map
        Use this function after dictionaries are updated."""
        args = self.args
        if self.mode in ['pose', 'pose_unit_test', 'test']:
            save_path = args.pose_intermediate_results_path
        elif self.mode in ['test_data']:
            save_path = args.input_bbox_path

        for i in range(self.num_test_imgs):
            detection_results = self.detection_results[i]
            retrieval_results = self.retrieval_results[i]
            pose_results = self.pose_results[i]
            img_name = self.img_names[i]
            with open(os.path.join(save_path, img_name + '.txt'), 'w') as f:
                for det, ret, pose in zip(detection_results, retrieval_results, pose_results):
                    pose = manage_duplicate_pose(ret, pose)
                    class_name = ret + '-' + str(pose)
                    x, y, w, h = det
                    r, b = x + w, y + h
                    confidence = 1.0
                    if type == 'gt':
                        line = '%s %d %d %d %d\n' % (class_name, x, y, r, b)
                    else:
                        line = '%s %f %d %d %d %d\n' % (class_name, confidence, x, y, r, b)
                    f.write(line)


    def save_detection_bbox_gt(self):
        """saves detection bbox ground truth"""
        args = self.args
        save_path = args.detection_anns_label_path
        m = int(args.detection_bbox_margin)

        for i in range(self.num_test_imgs):
            img_name = self.img_names[i]
            new_mid_results = self.new_mid_results[i]
            detection_results = self.detection_results[i]
            with open(os.path.join(save_path, img_name + '.txt'), 'w') as f:
                for NM, det, in zip(new_mid_results, detection_results):
                    x, y, w, h = det
                    r, b = x + w, y + h
                    # bbox margin
                    x = max(0, x - m)
                    y = max(0, y - m)
                    r = min(self.W, r + m)
                    b = min(self.H, b + m)
                    line = '%s %d %d %d %d\n' % (NM, x, y, r, b)
                    f.write(line)

    def save_retrieval_gt(self):
        """saves detection bbox ground truth"""
        args = self.args
        save_path = args.retrieval_gt_path
        m = int(args.detection_bbox_margin)

        for i in range(self.num_test_imgs):
            img_name = self.img_names[i]
            retrieval_label = self.ground_truth('retrieval')[i]
            detection_label = self.ground_truth('detection')[i]
            with open(os.path.join(save_path, img_name + '.txt'), 'w') as f:
                for ret_label, det, in zip(retrieval_label, detection_label):
                    x, y, w, h = det
                    r, b = x + w, y + h
                    # bbox margin
                    x = max(0, x - m)
                    y = max(0, y - m)
                    r = min(self.W, r + m)
                    b = min(self.H, b + m)
                    line = '%s %d %d %d %d\n' % (ret_label, x, y, r, b)
                    f.write(line)

    def save_retrieval_pred(self):
        """saves result dictionaries inorder to compute map
        Use this function after dictionaries are updated."""
        args = self.args
        save_path = args.retrieval_intermediate_results_path

        for i in range(self.num_test_imgs):
            detection_results = self.detection_results[i]
            retrieval_results = self.retrieval_results[i]
            img_name = self.img_names[i]
            with open(os.path.join(save_path, img_name + '.txt'), 'w') as f:
                for det, ret in zip(detection_results, retrieval_results):
                    class_name = ret
                    x, y, w, h = det
                    r, b = x + w, y + h
                    confidence = 1.0
                    line = '%s %f %d %d %d %d\n' % (class_name, confidence, x, y, r, b)
                    f.write(line)



