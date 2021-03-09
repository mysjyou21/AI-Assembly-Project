""" cut: png image, step: a step corresponds to a step number, a cut can consist of multiple steps
Caution: Class 변수 추가할 때, __init__에 적어주기(모두가 어떤 variable이 있는지 쉽게 파악하게 하기위해) """
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import sys

sys.path.append('./function')
sys.path.append('./function/OCR')
sys.path.append('./function/Pose')
sys.path.append('./function/utilities')
sys.path.append('./function/Hole_detection')
from config import *
from function.utilities.utils import *  # set_connectors, set_steps_from_cut
from function.frcnn.DetectionModel import DetectionModel
from function.action import *
from function.bubbles import *
from function.numbers import *
from function.mission_output import *
from function.OCRs_new import *
from function.Pose.evaluation.initial_pose_estimation import InitialPoseEstimation
from function.Fastener.evaluation.fastener_detection import FastenerDetection
from function.Grouping_mid.hole_loader import base_loader, mid_loader
from function.Grouping_mid.grouping_RT import transform_hole, baseRT_to_midRT
from function.Hole_detection.MSN2_hole_detector import MSN2_hole_detector

import json
import shutil
import copy

from keras import backend as K

class Assembly():

    def __init__(self, opt):
        self.opt = opt
        self.cuts = []
        self.step_names = []
        self.steps = {}
        self.num_steps = 0


        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config = tf.ConfigProto(gpu_options = gpu_options)
        config.gpu_options.allow_growth = True
        self.graph_detect1 = tf.Graph()
        with self.graph_detect1.as_default():
            self.sess_detect1 = tf.Session(config=config)
            self.detect_model1 = DetectionModel(self.opt.det_config1_name, self.graph_detect1, self.sess_detect1)
        self.graph_detect2 = tf.Graph()
        with self.graph_detect2.as_default():
            self.sess_detect2 = tf.Session(config=config)
            self.detect_model2 = DetectionModel(self.opt.det_config2_name, self.graph_detect2, self.sess_detect2)
        self.graph_OCR = tf.Graph()
        with self.graph_OCR.as_default():
            self.sess_OCR = tf.Session(config=config)
            self.OCR_model = OCRModel(self)
        self.graph_pose = tf.Graph()
        with self.graph_pose.as_default():
            self.sess_pose = tf.Session(config=config)
            self.pose_model = InitialPoseEstimation(self)
        self.graph_fastener = tf.Graph()
        with self.graph_fastener.as_default():
            self.sess_fastener = tf.Session(config=config)
            self.fastener_model = FastenerDetection(self)
        self.hole_detector_init = MSN2_hole_detector(self.opt)

        # Detection variables
        # component detection results, (x, y, w, h)
        self.circles_loc = {}
        self.circles_separated_loc = {}
        self.rectangles_loc = {}
        self.connectors_serial_imgs = {}
        self.connectors_serial_loc = {}
        self.connectors_mult_imgs = {}
        self.connectors_mult_loc = {}
        self.connectors_loc = {}
        self.parts_loc = {} # x,y,w,h,c
        self.tools_loc = {}
        self.is_merged = {}
        self.is_tool = {}
        self.unused_parts = {1 : [1, 2, 3, 4, 5, 6, 7, 8]} # unused parts {step_num : list of part ids}
        self.used_parts = {} # used parts {step_num : list of part ids}
        self.parts = {}  # detected part images {step_num  : list of part images}
        self.parts_bboxed = {}  # detected part images shown as bbox on whole image {step_num  : list of part images}
        self.parts_info = {}  # {step_num: list of (part_id, part_pos, hole_info)}
        self.mid_base = {} # {step_num: [base parts which consist of mid part]}
        self.is_fail = False # possibility for hole matching failure

        # component recognition results, string
        self.connectors_serial_OCR = {}
        self.connectors_mult_OCR = {}

        # Retrieval variables
        self.candidate_classes = []  # candidate cad models for retrieval

        # Pose variables
        self.cad_models = {}  # names of cad models of retrieval results
        self.pose_return_dict = {} # RT of detected part images {step_num : list of RT (type : np.array, shape : [3, 4])}
        self.part_write = {} # closest gt RT of parts, along with mid-pose info. {step_num : { part# : ['closest RT index', 'mid_pose info']}}
        self.pose_indiv = {} # isaac : closest gt RT of parts {step_num : { part# : ['closest RT index', 'added in pose module : True == 1, False == 0', 'New part : True == 1, False == 0']}}
        self.cad_names = ['part1', 'part2', 'part3', 'part4', 'part5', 'part6', 'part7', 'part8']
        self.K = np.array([
            3444.4443359375,0.0,874.0,
            0.0,3444.4443359375,1240.0,
            0.0,0.0,1.0]).reshape(3,3) # camera intrinsic matrix
        self.RTs = np.load(self.opt.pose_data_path + '/RTs.npy') # 48 RTs to compare with pose_network prediction
        for RT in self.RTs:
            RT[:, 3] = 0
        self.VIEW_IMGS = np.load(self.opt.pose_data_path + '/view_imgs.npy') # VIEWS to display pose output

        # Hole variables
        self.fasteners_loc = {} # detected fastener(=small component) locations {step_num : fastener_dict}
        '''
        fastener_dict :
            key : 'bracket', 'long_screw', 'short_screw', 'wood_pin'
            value : list of indiv fastener loc (default = [])
                indiv fastener loc : [(x, y, w, h)->bbox, (x, y)->centroid position]
        '''
        self.hole_pairs = {}


        # Final output
        self.actions = {}  # dictionary!! # [part1_loc, part1_id, part1_pos, part2_loc, part2_id, part2_pos, connector1_serial_OCR, connector1_mult_OCR, connector2_serial_OCR, connector2_mult_OCR, action_label, is_part1_above_part2(0,1)]
        self.step_action = {}

        self.data_loader()

    def data_loader(self):  # 준형
        """
        Load whole cut(.png) images and extract steps from each cut,
        save cut images and step images in the self.cuts and self.steps respectively.
        self.cuts : assembly image list
        self.steps : assembly step dictionary.
            e.g, {1: step01_img, 2: step02_img, ..., 9: step09_img}
        self.connectors_cuts: connectors image list
        self.num_steps : number of step images, except the first page(our setting, the image in the material folder)
        :return
        """

        # Load all images in the 'self.opt.cut_path'
        filenames = glob.glob(os.path.join(self.opt.cut_path, "*"))
        if os.path.basename(filenames[0]).replace('.bmp','').replace('.png','').isdigit():
            cut_paths = sorted(filenames, key=lambda x:int(os.path.basename(x).replace('.bmp','').replace('.png',''))) #.png')))
        elif len(filenames)>0:
            cut_paths = sorted(filenames)
        else:
            print("No image in %s"%(self.opt.cut_path))
            cut_paths = filenames
        cuts = [np.asarray(Image.open(cut_paths[n]))[:, :, :3] for n in range(len(cut_paths))]
        # 준형: resize, cut만 읽어올 수 있게 추가(material 말고), preprocessing(양탄자 처리)
        for cut in cuts:
            self.cuts.append(cut)

        # if argument 'starting_cut' is -1(=default value), is_valid_cut function is used.
        idx = 1
        if self.opt.starting_cut == -1:
            for n, cut in enumerate(cuts):
                # resize
                cut_resized = resize_cut(cut)
                # preprocessing
                cut_resized = prep_carpet(cut_resized)
                if is_valid_cut(cut_resized):
                    self.step_names.append(cut_paths[n])
                    self.steps[idx] = cut_resized
                    self.num_steps += 1
                    idx += 1
        else:
            for n, cut in enumerate(cuts[self.opt.starting_cut-1:]):
                # resize
                cut_resized = resize_cut(cut)
                # preprocessing
                cut_resized = prep_carpet(cut_resized)
                self.step_names.append(cut_paths[self.opt.starting_cut + n - 1])
                self.steps[idx] = cut_resized
                self.num_steps += 1
                idx += 1

        print([os.path.basename(x) for x in self.step_names])

    def detect_step_component(self, step_num, print_result=True):  # 준형
        """
        Detect components(a grouped action information) in a step
        step_num: corresponding step number
        print_result: Print the grouped action information as a csv format
        """
        # Detect components  # 한 cut에 같은 부품 2번 등장하면 하나만
        self.step_connectors, self.step_tools, self.step_circles, self.step_rectangles, self.step_parts, self.step_part_info = self.component_detector(step_num)
        self.step_connectors_mult_imgs, self.step_connectors_mult_loc = self.mult_detector(step_num)
        self.step_connectors_serial_imgs, self.step_connectors_serial_loc = self.serial_detector(step_num)

        # Set components as the step's components
        self.connectors_serial_imgs[step_num] = self.step_connectors_serial_imgs
        self.connectors_serial_loc[step_num] = self.step_connectors_serial_loc
        self.connectors_mult_imgs[step_num] = self.step_connectors_mult_imgs
        self.connectors_mult_loc[step_num] = self.step_connectors_mult_loc
        self.circles_loc[step_num] = self.step_circles
        self.rectangles_loc[step_num] = self.step_rectangles
        self.connectors_loc[step_num] = self.step_connectors
        self.tools_loc[step_num] = self.step_tools
        self.parts_loc[step_num] = self.step_parts
        self.parts_info[step_num] = self.step_part_info
        # self.base_info = ['part1', 'part2', ..., 'part6, 'part7', 'part8'']
        self.connector_serial_OCR, self.connector_mult_OCR = self.OCR(step_num)
        self.connectors_serial_OCR[step_num] = self.connector_serial_OCR
        self.connectors_mult_OCR[step_num] = self.connector_mult_OCR
        self.group_components(step_num)

        # visualization(준형) - image/bounding boxes (parts 제외, OCR결과도 그림에) self.opt.group_image_path -> 수정(한 cut 이미지에), flag self.opt.detection_path에 추가로
        # num of colors in palette: 7
        palette = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 128, 0), (255, 0, 128), (255, 0, 255)]

        num_groups = len(self.circles_loc[step_num])
        step_group_img = np.copy(self.steps[step_num])

        for n in range(num_groups):
            color = palette[n]
            # circles
            circle_loc = self.circles_loc[step_num][n]
            x, y, w, h, conf = circle_loc
            cv.rectangle(step_group_img, (x, y), (x + w, y + h), color=color, thickness=2)
            cv2.putText(step_group_img, 'circle: %.2f' % conf, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5,
                        color=color, thickness=3)
            # connectors
            connectors_loc = self.connectors_loc[step_num][n]
            for loc in connectors_loc:
                x, y, w, h, conf = loc
                cv.rectangle(step_group_img, (x, y), (x + w, y + h), color=color, thickness=2)
                cv2.putText(step_group_img, 'connector: %.2f' % conf, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5,
                            color=color, thickness=3)
            # serials
            serials_loc = self.connectors_serial_loc[step_num][n]
            for i, loc in enumerate(serials_loc):
                x, y, w, h, angle = loc
                theta = angle * np.pi / 180
                w, h = w / 2, h / 2
                pt1 = [int(x - w * np.cos(theta) - h * np.sin(theta)), int(y + w * np.sin(theta) - h * np.cos(theta))]
                pt2 = [int(x - w * np.cos(theta) + h * np.sin(theta)), int(y + w * np.sin(theta) + h * np.cos(theta))]
                pt3 = [int(x + w * np.cos(theta) + h * np.sin(theta)), int(y - w * np.sin(theta) + h * np.cos(theta))]
                pt4 = [int(x + w * np.cos(theta) - h * np.sin(theta)), int(y - w * np.sin(theta) - h * np.cos(theta))]
                pts = np.array([pt1, pt2, pt3, pt4], dtype=int)
                pts = np.reshape(pts, (-1, 1, 2))
                cv.polylines(step_group_img, [pts], True, color, 2)
                # serial_OCRs
                serial_OCR = self.connectors_serial_OCR[step_num][n][i]
                margin = 20
                px = int(np.min([pt[0] for pt in [pt1, pt2, pt3, pt4]]) - margin)
                py = int(np.max([pt[1] for pt in [pt1, pt2, pt3, pt4]]) + margin)
                cv.putText(step_group_img, serial_OCR, (px, py), cv.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=color,
                           thickness=3)
            # mults
            mults_loc = self.connectors_mult_loc[step_num][n]
            for i, loc in enumerate(mults_loc):
                x, y, w, h = loc
                cv.rectangle(step_group_img, (x, y), (x + w, y + h), color, 2)
                mult_OCR = self.connectors_mult_OCR[step_num][n][i]
                margin = 5
                px = int(x + w / 2)
                py = int(y - margin)
                cv.putText(step_group_img, mult_OCR, (px, py), cv.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=color,
                           thickness=3)
        if self.opt.save_group_image:
            if step_num == 1:
                if os.path.exists(self.opt.group_image_path):
                    shutil.rmtree(self.opt.group_image_path)
            if not os.path.exists(self.opt.group_image_path):
                os.makedirs(self.opt.group_image_path)

            img_name = os.path.join(self.opt.group_image_path, '%02d.png' % step_num)
            cv.imwrite(img_name, step_group_img)

        # parts
        step_part_img = np.copy(self.steps[step_num])
        parts_color = (255, 0, 0)
        for n in range(len(self.parts_loc[step_num])):
            x, y, w, h, conf = self.parts_loc[step_num][n]
            cls = self.parts_info[step_num][n]
            cv2.rectangle(step_part_img, (x, y), (x + w, y + h), color=parts_color, thickness=2)
            cv2.putText(step_part_img, '%s: %.2f' % (cls, conf), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5,
                        color=parts_color, thickness=3)

        if self.opt.save_part_image:
            if step_num == 1:
                if os.path.exists(self.opt.part_image_path):
                    shutil.rmtree(self.opt.part_image_path)
            if not os.path.exists(self.opt.part_image_path):
                os.makedirs(self.opt.part_image_path)
            img_name = os.path.join(self.opt.part_image_path, '%02d.png' % step_num)
            cv.imwrite(img_name, step_part_img)

        # crop part images from step images with detection results
        step_part_images = []
        step_part_images_bboxed = []
        for i, crop_region in enumerate(self.parts_loc[step_num]):
            x, y, w, h = crop_region[:4]
            step_part_image = self.steps[step_num][y:y + h, x:x + w]
            step_part_images.append(step_part_image)
            step_part_image_bboxed = self.steps[step_num].copy()
            step_part_image_bboxed = cv2.rectangle(step_part_image_bboxed, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
            step_part_images_bboxed.append(step_part_image_bboxed)
        self.parts[step_num] = step_part_images
        self.parts_bboxed[step_num] = step_part_images_bboxed

        # save detection result images
        if self.opt.save_detection:
            if step_num == 1:
                if os.path.exists(self.opt.detection_path):
                    shutil.rmtree(self.opt.detection_path)
            if not os.path.exists(self.opt.detection_path):
                os.makedirs(self.opt.detection_path)

            if not os.path.exists(self.opt.detection_path):
                os.makedirs(self.opt.detection_path)
            for i in range(len(self.parts[step_num])):
                cv2.imwrite(self.opt.detection_path + '/STEP{}_part{}.png'.format(step_num, i),
                            self.parts[step_num][i])

        # update self.used_parts[step_num] / self.unused_parts[step_num + 1]
        self.used_parts[step_num] = sorted(list(set([int(part_id.replace('part', '')) for part_id in self.parts_info[step_num]])))
        self.unused_parts[step_num + 1] = sorted(list(set(self.unused_parts[step_num]) - set(self.used_parts[step_num])))

    def component_detector(self, step_num):  # 준형
        """ Detect the components in the step image, return the detected components' locations (x, y, w, h, group_index)
            component list: (using Faster R-CNN) connector(image), tool(image), circle, rectangle, (다른게 있으면 추가로!) """
        step_img = self.steps[step_num]

        components_dict1 = self.detect_model1.test_for_components(step_img)
        if self.opt.temp:
            components_dict2 = self.detect_model2.test_for_parts(step_img, parts_threshold=self.opt.parts_threshold, step_num=step_num)  # temp
        else:
            components_dict2 = self.detect_model2.test_for_parts(step_img, parts_threshold=self.opt.parts_threshold)    # temp

        # print(components_dict1)
        # print(components_dict2)
        circles = components_dict1['Guidance_Circle']
        if 'Guidance_Sqaure' in components_dict1.keys():
            rectangles = components_dict1['Guidance_Sqaure']
        else:
            rectangles = components_dict1['Guidance_Square']
        connectors = components_dict1['Elements']
        tools = components_dict1['Tool']
        part_ids = sorted(list(components_dict2.keys()))
        part_ids.remove('bg')

        # 첫 번째 step 제외한 모든 step 에 대해,
        # part7(또는 part8)이 검출됐는데 첫 번째 step 에 part2(또는 part3) 이 없으면 오검출로 간주
        if step_num != 1:
            if 2 not in self.used_parts[1] and '7' in part_ids:
                part_ids.remove('7')
            if 3 not in self.used_parts[1] and '8' in part_ids:
                part_ids.remove('8')

        parts = []
        for key in part_ids:
            parts += components_dict2[key]
        parts_info = []
        for part_id in part_ids:
            parts_info += ['part%s' % part_id] * len(components_dict2[part_id])

        return connectors, tools, circles, rectangles, parts, parts_info

    def hole_detector(self, step_num, step_parts_id, step_parts_pose):
        step_img = (self.steps[step_num]).copy()
        step_parts_loc = self.parts_loc[step_num]
        hole_info = []
        for loc in step_parts_loc:
            hole_info += [[]]
        connectivity = ''
        step_parts_loc_origin = step_parts_loc.copy()
        y1, y2, x1, x2 = step_img.shape[0], 0, step_img.shape[1], 0
        for x, y, w, h, _ in step_parts_loc:
            if x < x1:
                x1 = x
            if y < y1:
                y1 = y
            if x + w > x2:
                x2 = x + w
            if y + h > y2:
                y2 = y + h
        step_roi = step_img[y1 - 20:y2 + 30, x1 - 50:x2 + 20]

        step_connectors_OCR = self.connectors_serial_OCR[step_num]

        if step_num == 1:
            if os.path.exists(self.opt.part_hole_path):
                shutil.rmtree(self.opt.part_hole_path)
        if not os.path.exists(self.opt.part_hole_path):
            os.makedirs(self.opt.part_hole_path)

        # visualization
        for i in range(len(step_parts_loc_origin)):
            x, y, w, h, _ = step_parts_loc_origin[i]
            step_img = cv2.rectangle(step_img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
            step_img = cv2.putText(step_img, step_parts_id[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                   color=(255, 0, 0), thickness=2)
            holes = hole_info[i]
            if holes == []:
                continue
            for hole in holes:
                hole_type, p = hole
                px1, py1, px2, py2 = int(p[1]) + (x1 - 50) - 5, int(p[2]) + (y1 - 20) - 5, int(p[3]) + (
                        x1 - 50) + 5, int(p[4]) + (y1 - 20) + 5
                if step_num == 3:
                    py1 -= 90
                    py2 -= 90
                if step_num == 6:
                    px1 -= 50
                    px2 -= 50
                    py1 -= 180
                    py2 -= 180
                step_img = cv2.rectangle(step_img, (px1, py1), (px2, py2), color=(0, 255, 0), thickness=2)
                step_img = cv2.putText(step_img, hole_type[0:5], (px1, py1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                                       color=(0, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(self.opt.part_hole_path, '%.2d.png' % step_num), step_img)

        return hole_info, connectivity

    def serial_detector(self, step_num):  # 이삭
        """
        check serial_numbers in a step-image
        return: serial_loc = [midpoint(x), midpoint(y), w, h, angle_degrees]
        """
        serial_loc = serial_detect(self, step_num)
        return serial_loc

    def mult_detector(self, step_num):  # 이삭
        """
        check multiple_numbers in a step-image
        return: mult_loc = [x, y, w, h]
        """
        mult_loc = mult_detect(self, step_num)
        return mult_loc

    def group_components(self, step_num):  # 준형 -상하관계도 고려!!-변수 추가부탁함..!
        """
        OCR, 상하관계 아직 구현 X
        Group components(circle, rectangle, connector_serial, connector_mult, connector_img, tool)
        Update each variables.
        For example, self.serial_numbers[step_num]=[SERIAL1, SERIAL2, SERIAL3](existing) => self.serial_numbers[step_num]=[[SERIAL1, SERIAL2], [SERIAL3], []](updated) """
        step_circles = self.circles_loc[step_num]
        step_rectangles = self.rectangles_loc[step_num]
        step_connectors = self.connectors_loc[step_num]
        step_connectors_serial = self.connectors_serial_loc[step_num]
        step_connectors_mult = self.connectors_mult_loc[step_num]
        step_tools = self.tools_loc[step_num]
        step_connectors_serial_OCR = self.connectors_serial_OCR[step_num]
        step_connectors_mult_OCR = self.connectors_mult_OCR[step_num]
        step_parts = self.parts_loc[step_num]
        parts_info = self.parts_info[step_num]

        # DIVIDE step-components into several groups, regarding connector_mult-connector_serial pairs!
        step_circles_new, step_connectors_new, step_connectors_serial_new, step_connectors_mult_new, \
        step_tools_new, step_connectors_serial_OCR_new, step_connectors_mult_OCR_new, \
        step_parts_new, parts_info_new, is_merged = \
            grouping(step_circles, step_rectangles, step_connectors, step_connectors_serial,
                     step_connectors_mult, step_tools, step_connectors_serial_OCR,
                     step_connectors_mult_OCR, step_parts, parts_info)

        # update variables
        self.circles_loc[step_num] = step_circles_new
        self.circles_separated_loc[step_num] = step_circles
        self.connectors_loc[step_num] = step_connectors_new
        self.connectors_serial_loc[step_num] = step_connectors_serial_new
        self.connectors_mult_loc[step_num] = step_connectors_mult_new
        self.tools_loc[step_num] = step_tools_new
        self.connectors_serial_OCR[step_num] = step_connectors_serial_OCR_new
        self.connectors_mult_OCR[step_num] = step_connectors_mult_OCR_new
        self.parts_loc[step_num] = step_parts_new
        self.parts_info[step_num] = parts_info_new
        self.is_merged[step_num] = is_merged

    def OCR(self, step_num):  # 준형
        """ OCR this step's serial numbers and multiple numbers
        return: connector_serial_OCR, connector_mult_OCR
        """

        # serials_list: 예를 들어 현재 step에 부품 번호가 2개 있으면
        # [[img1, ..., img6], [img1, ..., img6]] 와 같은 형식. imgN은 N번째 자리에 해당하는 숫자의 이미지(array).
        # mults_list도 마찬가지
        serials_list = self.connectors_serial_imgs[step_num]
        mults_list = self.connectors_mult_imgs[step_num]
        for i in range(len(serials_list) - 1, -1, -1):
            if serials_list[i] == []:
                serials_list.remove(serials_list[i])

        connector_serial_OCR = []
        connector_mult_OCR = []
        true_serials = [['101350'], ['122620'], ['104322'], ['122925']]

        for serials in serials_list:
            serial_OCR = self.OCR_model.run_OCR_serial(args=self, imgs=serials)

            if (serial_OCR not in true_serials) and (serial_OCR != '') and (serial_OCR != '100001'):
                count = [0, 0, 0, 0]
                for idx, key in enumerate(true_serials):
                    for i in range(6):
                        if serial_OCR[i] == key[0][i]: count[idx] = count[idx] + 1
                max_idx = np.argmax(np.asarray(count))
                serial_OCR = true_serials[max_idx][0]

            connector_serial_OCR.append(serial_OCR)

        for mults in mults_list:
            mult_OCR = self.OCR_model.run_OCR_mult(args=self, imgs=mults)
            connector_mult_OCR.append(mult_OCR)

        return connector_serial_OCR, connector_mult_OCR

    def predict_pose(self, step_num):
        """ Pose prediction
        Pose prediction of detected regions of image.
        Args:
            step_num: step number
        Uses:
            self.steps : (dict) {step_num : step image }
                step image : (array) (uint8)
                        step images

            self.parts_loc : (dict) {step_num : (list) bbox }
                bbox : (list) (int) x, y, w, h
                        detection bbox results

        Updates:
            self.parts : (dict) {step_num : (list) part images }
                part images : (array) (uint8)

                        part images cropped from step images

            self.cad_models : (dict) {step_num : (list) (str) classified cad names }

                        detection class results + "New" from pose segmentation map
            self.pose_return_dict : (dict) {step_num : (list) (array) [3, 4] RT of pose prediction }

            matched_poses : (list) (int) pose prediction as number 0~47
        Returns:
            None

        """
        # ensure directories
        # :resume from this step functionality

        # detection classification results
        self.cad_models[step_num] = self.parts_info[step_num].copy()
        assert len(self.parts[step_num]) == len(self.cad_models[step_num]), "length of retrieval input/output doesn't match"
        print('')
        print('classified classes : ', self.cad_models[step_num]) #blue
        parts = self.cad_models[step_num].copy()

        # pose prediction
        # update self.pose_return_dict, self.cad_models
        self.pose_model.test(self, step_num)
        # update self.parts_info
        self.parts_info[step_num] = self.cad_models[step_num]

        # pose visualization
        self.pose_model.visualize(self, step_num)

        # classifiy to nearest GT pose
        def R_distance(RT1, RT2):
            R1 = RT1[:, :3]
            R2 = RT2[:, :3]
            R = R1.T @ R2
            theta = np.rad2deg(np.arccos((np.trace(R) - 1)/2))
            return theta
        def closest_gt_RT_index(RT_pred):
            return np.argmin([R_distance(RT, RT_pred) for RT in self.RTs])
        def closest_gt_RT_index_list(RT_pred):
            return [R_distance(RT, RT_pred) for RT in self.RTs]
        matched_poses = [closest_gt_RT_index(RT_pred) for RT_pred in self.pose_return_dict[step_num]]
        matched_poses_list = [closest_gt_RT_index_list(RT_pred) for RT_pred in self.pose_return_dict[step_num]]
        R_list = [RT_pred[:, :3] for RT_pred in self.pose_return_dict[step_num]]

        # individual parts pose visualization
        self.pose_model.save_part_id_pose(self, step_num, matched_poses)

        parts_modified = self.cad_models[step_num].copy()
        print('modified classified classes : ', parts_modified) #blue
        print('matched_poses', matched_poses) #blue

        # update self.part_write
        step_part_write_dict = {}
        for part_name, matched_pose in zip(parts_modified, matched_poses):
            matched_pose_list = []
            matched_pose_list.append(int(matched_pose))
            step_part_write_dict[part_name] = matched_pose_list
        self.part_write[step_num] = step_part_write_dict

        # update self.pose_indiv
        step_part_write_dict = {}
        for part_name, matched_pose in zip(parts_modified, matched_poses):
            matched_pose_list = []
            matched_pose_list.append(int(matched_pose))
            added_in_pose_module = 1 if part_name not in parts else 0
            matched_pose_list.append(int(added_in_pose_module))
            step_part_write_dict[part_name] = matched_pose_list
        self.pose_indiv[step_num] = step_part_write_dict

        # save pose indiv
        # json file for inidividual pose comparison with gt
        if self.opt.save_pose_indiv:
            try:
                with open('./isaac/check_pose_indiv/predict/' + self.opt.assembly_name + '.json', 'w') as f:
                    json.dump(self.pose_indiv, f, indent=2, sort_keys=True)
            except:
                pass

    def fastener_detector(self, step_num):
        """ Fastener detection

        '122620' = 'bracket',
        '104322' = 'long_screw',
        '122925' = 'short_screw',
        '101350' = 'wood_pin'

        Args:
            step_num: step number
        Uses:
            self.steps
            self.circles_loc
            self.rectangles_loc
            self.connectors_serial_OCR

        Updates:
            self.fasteners_loc :
                detected fastener(=small component) locations {step_num : fastener_dict}
                    fastener_dict :
                        key : 'bracket', 'long_screw', 'short_screw', 'wood_pin'
                        value : list of indiv fastener loc (default = [])
                            indiv fastener loc : [(x, y, w, h)->bbox, (x, y)->centroid position]
        Returns:
            None
        """
        # run
        self.fastener_model.test(self, step_num)
        # visualize
        self.fastener_model.visualize(self, step_num)


    def group_as_action(self, step_num):  # action 관련 정보들을 묶어서 self.actions에 넣고 ./output에 json write
        """ Group components in action-unit
        [part1_loc, part1_id, part1_pos, part2_loc, part2_id, part2_pos, connector1_serial_OCR, connector1_mult_OCR, connector2_serial_OCR, connector2_mult_OCR, action_label, is_part1_above_part2(0,1)]
        Update self.actions """

        material = self.connectors_serial_OCR[step_num]
        if (len(material) != 0) and ('100001' in material[0]):
            key = material[0].index('100001')
            material.pop(key)
        circle_mult = self.connectors_mult_OCR[step_num]

        f = open(os.path.join('function', 'utilities', 'action_label.csv'), 'r')  # , encoding='utf-8')
        csv_reader = csv.reader(f)
        act_dic = {}  # key: '100001' value: ['A001']
        next(csv_reader)  # ignore first line
        for line in csv_reader:
            part_lab = line[0]
            act_dic[part_lab] = line[1:]
        f.close()

        ########## 0. prepare for write mission / change pose_id ##########

        with open('./function/utilities/label_to_pose.json', 'r') as f:
            pose_dic = json.load(f)

        if step_num == 1:
            if os.path.exists(self.opt.output_dir):
                shutil.rmtree(self.opt.output_dir)

        filename = self.opt.cut_path.split('/')[-2]
        if not os.path.exists(self.opt.output_dir):
            os.makedirs(self.opt.output_dir)

        ######## 1. mapping action first #######
        if self.is_merged[step_num] == True:  # (원 두개 붙어서 나오는 경우  ex - stefan step9) 따로 해결
            # serial이 한개면 step9와 같은 상황이라고 판정.
            # 그외 케이스 일 경우 action을 뱉지 X
            connectors, tools, circles, rectangles, parts, _ = self.component_detector(step_num)
            self.step_action['number_of_circle'] = len(circles)
            self.step_action['serials'] = self.connectors_serial_OCR[step_num]
            self.step_action['mult'] = self.connectors_mult_OCR[step_num]
            self.step_action['action'] = ['']
            if len(material) == 1 and ['122925'] in material: self.step_action['action'] = ['A003', 'A001']

        else:
            if len(material) == 1 and material != [[]]:
                serials, circle_action, circle_num = map_action(self, material[0], circle_mult[0], act_dic, step_num)
                self.step_action['serials'] = serials
                self.step_action['mult'] = circle_num
                self.step_action['action'] = circle_action
            elif (material == []) and (len(self.parts_info[step_num]) - 1) == 1:  # 원이 안나오고 part 1개면 -> A005
                self.step_action['serials'] = ['']
                self.step_action['mult'] = ['1']
                self.step_action['action'] = ['A006']
            elif material == [[], []]:  # 원이 두갠데 serial이 하나도 안나오는 경우
                self.step_action['serials'] = ['']
                self.step_action['mult'] = ['1']
                self.step_action['action'] = ['A005']
            else:  # error : 한 원에 material 2개
                print('Exception case in action mapping')
                self.step_action['serials'] = ['']
                self.step_action['mult'] = ['1']
                self.step_action['action'] = ['']

        self.step_action['parts#'] = len(self.parts_info[step_num]) - 1
        connectivity = self.parts_info[step_num][-1]

        ########### 2. write mission json ########
        f = open(os.path.join(self.opt.output_dir, 'mission_%s.json' % str(step_num)), 'w')

        step_dic = OrderedDict()
        step_dic['File_name'] = filename
        step_dic['Label_step_number'] = str(step_num)

        # import ipdb; ipdb.set_trace()

        ######## 1. mapping action ##############
        if self.is_merged[step_num] == True:  ## exception_case
            for act_i in range(len(self.step_action['action'])):
                mults = []  ##exception
                action_dic = OrderedDict()
                for i in range(0, 4):
                    part_dic = OrderedDict()
                    if i < self.step_action['parts#']:
                        part = copy.deepcopy(self.parts_info[step_num][i])
                        mults.append([str(len(part[2]))])
                    else:
                        part = ['', '', '']
                    if part[0] != '':
                        part_id = part[0]  # 'string'
                        if part_id == 'part7': part_id = 'step1_b'
                        if part_id == 'part8': part_id = 'step1_a'
                        part_pose_ind = part[1]
                        part_pose_lab = pose_dic[str(part_pose_ind)]
                        part[1] = part_pose_lab.split('_')
                        part_pose = part[1]  # [theta,phi,alpha,additional]
                        part_holes = part[2]
                        part_dic['label'] = part_id
                        part_dic['theta'] = part_pose[0]
                        part_dic['phi'] = part_pose[1]
                        part_dic['alpha'] = part_pose[2]
                        part_dic['additional'] = part_pose[3]
                        part_dic['#'] = '1'  # default 1

                        part_holes = part[-1]
                        part_dic['hole'] = part[-1]
                    action_dic['Part%d' % i] = part_dic
                action_lab_dic = OrderedDict()
                action_lab_dic['label'] = self.step_action['action'][act_i]

                if action_lab_dic['label']=='A003':
                    action_lab_dic['#'] = mults[0][0]
                    action_dic['Action'] = action_lab_dic

                    connector_dic = OrderedDict()
                    connector_dic['label'] = ''
                    connector_dic['#'] = '1'
                    action_dic['Connector'] = connector_dic
                    action_dic['HolePair'] = self.hole_pairs[step_num]
                else:
                    serials = self.step_action['serials']
                    action_lab_dic['#'] = mults[0][0]
                    action_dic['Action'] = action_lab_dic

                    connector_dic = OrderedDict()
                    connector_dic['label'] = 'C' + serials[0][0]
                    connector_dic['#'] = mults[0][0]
                    action_dic['Connector'] = connector_dic
                    action_dic['HolePair'] = self.hole_pairs[step_num]
                    action_dic['Fail'] = self.is_fail


                step_dic['Action%d' % act_i] = action_dic
            if len(self.step_action['action']) == 0: step_dic['Action0'] = return_empty_dict

            pass
        ###### 여기 위까진 완성 !! #########
        ######### 이제 이 밑에 일반적인 case들 시작 #######

        elif self.step_action['serials'] != [''] and len(self.step_action['serials']) == 1:
            # serial이 허수는 아니며 정상적으로 들어옴 ! ( general case )
            if connectivity[0] == '':  # serial은 1개 인데 connectivity가 X -> step1 같은 상황
                step_part = copy.deepcopy(self.parts_info[step_num])
                if self.step_action['parts#']==2:
                    part_id = []
                    for idx in range(0,2):
                        part_id.append(step_part[idx][0])
                    if part_id == ['part2','part3']:
                        temp = copy.deepcopy(step_part)
                        step_part[0] = temp[1]
                        step_part[1] = temp[0]

                for act_i in range(len(step_part) - 1):
                    action_dic = OrderedDict()
                    for i in range(0, 4):
                        part_dic = OrderedDict()
                        if i == 0:
                            part = copy.deepcopy(step_part[act_i])
                            if self.step_action['parts#']!=1: mults = [str(len(part[2]))]
                            else: mults = self.step_action['mult']
                        else:
                            part = ['', '', '']
                        if part[0] != '':
                            part_id = part[0]  # 'string'
                            part_pose_ind = part[1]
                            part_pose_lab = pose_dic[str(part_pose_ind)]
                            part[1] = part_pose_lab.split('_')
                            part_pose = part[1]  # [theta,phi,alpha,additional]
                            part_holes = part[2]
                            part_dic['label'] = part_id
                            part_dic['theta'] = part_pose[0]
                            part_dic['phi'] = part_pose[1]
                            part_dic['alpha'] = part_pose[2]
                            part_dic['additional'] = part_pose[3]
                            part_dic['#'] = '1'  # default 1

                            part_dic['hole'] = part[-1]
                        action_dic['Part%d' % i] = part_dic
                    action_lab_dic = OrderedDict()
                    action_lab_dic['label'] = circle_action[0]
                    if len(mults) == 0:
                        mults = ['1']
                    action_lab_dic['#'] = mults[0]
                    action_dic['Action'] = action_lab_dic

                    connector_dic = OrderedDict()
                    if len(serials) == 0:
                        serials = ['']
                    elif len(serials[0]) == 0:
                        mults[0] = '1'
                    connector_dic['label'] = 'C' + serials[0] if len(serials[0]) > 0 else serials[0]
                    connector_dic['#'] = mults[0]
                    action_dic['Connector'] = connector_dic
                    action_dic['HolePair'] = self.hole_pairs[step_num]
                    action_dic['Fail'] = self.is_fail

                    step_dic['Action%d' % act_i] = action_dic

            else:  # serial은 1개 인데 connectivity != '' -> 무언가로 연결해야함 (1:1 or 1:many 연결)
                connectivity = connectivity[0].split('#')
                if len(connectivity) == 2:  # 1:1 connecivity
                    action_dic = OrderedDict()

                    for i in range(0, 4):
                        part_dic = OrderedDict()
                        if i < self.step_action['parts#']:
                            part = copy.deepcopy(self.parts_info[step_num][i])
                        else:
                            part = ['', '', '']
                        if part[0] != '':
                            part_id = part[0]  # 'string'
                            if part_id == 'part7':
                                if step_num==2 and self.used_parts[1]==[2] : part_id = 'step1'
                                else: part_id = 'step1_b'
                            if part_id == 'part8':
                                if step_num == 2 and self.used_parts[1] == [3]:
                                    part_id = 'step1'
                                else:
                                    part_id = 'step1_a'
                            part_pose_ind = part[1]
                            part_pose_lab = pose_dic[str(part_pose_ind)]
                            part[1] = part_pose_lab.split('_')
                            part_pose = part[1]  # [theta,phi,alpha,additional]
                            part_holes = part[2]
                            part_dic['label'] = part_id
                            part_dic['theta'] = part_pose[0]
                            part_dic['phi'] = part_pose[1]
                            part_dic['alpha'] = part_pose[2]
                            part_dic['additional'] = part_pose[3]
                            part_dic['#'] = '1'  # default 1

                            part_dic['hole'] = part[-1]
                        action_dic['Part%d' % i] = part_dic
                    action_lab_dic = OrderedDict()
                    action_lab_dic['label'] = self.step_action['action'][0]

                    mults = copy.deepcopy(self.step_action['mult'])
                    if len(mults) == 0:
                        mults = ['1']
                    action_lab_dic['#'] = mults[0]
                    action_dic['Action'] = action_lab_dic

                    connector_dic = OrderedDict()
                    serials = copy.deepcopy(self.step_action['serials'])
                    if len(serials) == 0:
                        serials = ['']
                    elif len(serials[0]) == 0:
                        mults[0] = '1'
                    connector_dic['label'] = 'C' + serials[0] if len(serials[0]) > 0 else serials[0]
                    connector_dic['#'] = mults[0]
                    action_dic['Connector'] = connector_dic
                    action_dic['HolePair'] = self.hole_pairs[step_num]
                    action_dic['Fail'] = self.is_fail

                    step_dic['Action%d' % 0] = action_dic
                elif len(connectivity) == 4:  # temp 1:many -> to be added
                    action_dic = OrderedDict()
                    for i in range(0, 4):
                        part_dic = {}
                        action_dic['Part%d' % i] = part_dic
                    action_lab_dic = OrderedDict()
                    action_lab_dic['label'] = ''
                    action_lab_dic['#'] = ''
                    action_dic['Action'] = action_lab_dic

                    connector_dic = OrderedDict()
                    connector_dic['label'] = ''
                    connector_dic['#'] = ''
                    action_dic['Connector'] = connector_dic
                    action_dic['HolePair'] = []
                    action_dic['Fail'] = self.is_fail

                    step_dic['Action%d' % 0] = action_dic


        elif self.step_action['action'] in [['A005'], ['A006']]:
            action_dic = OrderedDict()
            for i in range(0, 4):
                part_dic = OrderedDict()
                if i < self.step_action['parts#']:
                    part = copy.deepcopy(self.parts_info[step_num][i])
                else:
                    part = ['', '', '']
                if part[0] != '':
                    part_id = part[0]  # 'string'
                    if part_id == 'part7': part_id = 'step1_b'
                    if part_id == 'part8': part_id = 'step1_a'
                    part_pose_ind = part[1]
                    part_pose_lab = pose_dic[str(part_pose_ind)]
                    part[1] = part_pose_lab.split('_')
                    part_pose = part[1]  # [theta,phi,alpha,additional]
                    part_holes = part[2]
                    part_dic['label'] = part_id
                    part_dic['theta'] = part_pose[0]
                    part_dic['phi'] = part_pose[1]
                    part_dic['alpha'] = part_pose[2]
                    part_dic['additional'] = part_pose[3]
                    part_dic['#'] = '1'  # default 1

                    part_dic['hole'] = part[-1]
                action_dic['Part%d' % i] = part_dic
            action_lab_dic = OrderedDict()
            action_lab_dic['label'] = self.step_action['action'][0]

            mults = copy.deepcopy(self.step_action['mult'])
            if len(mults) == 0:
                mults = ['1']
            action_lab_dic['#'] = mults[0]
            action_dic['Action'] = action_lab_dic

            connector_dic = OrderedDict()
            serials = copy.deepcopy(self.step_action['serials'])
            connector_dic['label'] = ''
            connector_dic['#'] = '1'
            action_dic['Connector'] = connector_dic
            action_dic['HolePair'] = self.hole_pairs[step_num]
            action_dic['Fail'] = self.is_fail

            step_dic['Action%d' % 0] = action_dic
        else:
            action_dic = OrderedDict()
            for i in range(0, 4):
                part_dic = {}
                action_dic['Part%d' % i] = part_dic
            action_lab_dic = OrderedDict()
            action_lab_dic['label'] = ''
            action_lab_dic['#'] = ''
            action_dic['Action'] = action_lab_dic

            connector_dic = OrderedDict()
            connector_dic['label'] = ''
            connector_dic['#'] = ''
            action_dic['Connector'] = connector_dic
            action_dic['HolePair'] = []
            action_dic['Fail'] = self.is_fail

            step_dic['Action%d' % 0] = action_dic

        json.dump(step_dic, f, indent=2)

        f.close()
        self.actions[step_num] = self.step_action

        ########## print_pred ########
        if self.opt.print_pred:
            if self.opt.pred_dir is None: self.opt.pred_dir = os.path.join(self.opt.output_dir, 'pred')
            if step_num == 1:
                if os.path.exists(self.opt.pred_dir):
                    shutil.rmtree(self.opt.pred_dir)
            if not os.path.exists(self.opt.pred_dir):
                os.makedirs(self.opt.pred_dir)

            pred_dic = copy.deepcopy(step_dic)
            mid_name = 'step%s' % str(step_num-1)

            for i in range(len(pred_dic)-2):
                Action_dic = copy.deepcopy(pred_dic['Action%s' % str(i)])
                for a in range(0,4):
                    if Action_dic['Part%s' % str(a)]=={}: pass
                    else:
                        part_original = copy.deepcopy(Action_dic['Part%s' % str(a)])
                        part_changed = OrderedDict()
                        part_changed['label'] = part_original['label']

                        if part_original['label'] == mid_name: ## 중간산출물인 경우 (2개인 경우는 고려 X)
                            sub_part_dic = OrderedDict()
                            for part_name in self.mid_base[step_num]:
                                if part_name in self.part_write[step_num].keys():
                                    sub_part_dic[part_name] = [str(pose) for pose in self.part_write[step_num][part_name]]

                        else: ## 기본부품일 경우
                            sub_part_dic = OrderedDict()
                            part_name = part_original['label']
                            sub_part_dic[part_name] = [str(pose) for pose in self.part_write[step_num][part_name]]

                        part_changed['sub_part'] = sub_part_dic
                        part_changed['#'] = part_original["#"]
                        part_changed['hole'] = part_original['hole']

                        Action_dic['Part%s' % str(a)] = part_changed


                pred_dic['Action%s' % str(i)] = Action_dic

            f = open(os.path.join(self.opt.pred_dir, 'mission_%s.json' % str(step_num)), 'w')
            json.dump(pred_dic, f, indent=2)
            f.close()

    def group_RT_mid(self, step_num):
        ######### detection에서 검출된 기본부품의 pose 반영 hole 위치 #######
        base_RT_list = self.pose_return_dict[step_num]
        # base_id_list = [base_part[0] for base_part in self.parts_info[step_num][:-1]]
        base_id_list = self.parts_info[step_num]

        if (len(base_id_list) < 2):
            pass  ######## 설명서내에 검출된 기본 part가 2개 미만인 경우 #########

        base_hole_dict = {}
        for i, id in enumerate(base_id_list):
            base_hole_XYZ, _ = base_loader(id, self.opt.hole_path, self.opt.cad_path)
            base_RT = base_RT_list[i]
            base_hole_dict[id] = transform_hole(base_RT, base_hole_XYZ)


        ######## 중간산출물 hole 위치 loading  ##########
        mid_hole_dict, _ = mid_loader('step%i'%(step_num-1), self.opt.hole_path, self.opt.cad_path)

        mid_RT, mid_id_list, find_mid, mid_checked = baseRT_to_midRT(base_hole_dict, mid_hole_dict)

        self.find_mid = find_mid
        self.mid_id_list = mid_id_list
        self.mid_RT = mid_RT

        if find_mid:
            for key in self.part_write[step_num]:
                if key in mid_checked:
                    self.part_write[step_num][key] += [1]
                else:
                    self.part_write[step_num][key] += [0]

            for key in mid_id_list:
                if key in base_id_list:
                    idx = base_id_list.index(key)
                    base_RT_list[idx] = mid_RT
                else:
                    base_id_list.append(key)
                    base_RT_list.append(mid_RT)

            self.pose_return_dict[step_num] = base_RT_list
            self.parts_info[step_num] = list(zip(base_id_list, base_RT_list))
        else:
            self.parts_info[step_num] = list(zip(base_id_list, base_RT_list))


    def msn2_hole_detector(self, step_num):
        ##### General Variables #####
        step_images = self.steps.copy()
        connectors = self.connectors_serial_OCR.copy()
        mults = self.connectors_mult_OCR.copy()

        if len(connectors[step_num]) != 0:
            if len(connectors[step_num][0]) != 0:
                pass
            else:
                connectors[step_num] = [[]]
        else:
            connectors[step_num] = [[]]

        if len(mults[step_num]) != 0:
            if len(mults[step_num][0]) != 0:
                pass
            else:
                mults[step_num] = [[]]
        else:
            mults[step_num] = [[]]


        ##### Eliminate from fastener candidate #####
        component_list = list()
        circles_loc = self.circles_loc[step_num].copy()
        rectangles_loc = self.rectangles_loc[step_num].copy()
        connectors_loc = self.connectors_loc[step_num].copy()
        tools_loc = self.tools_loc[step_num].copy()
        if len(circles_loc)!=0:
            for circle in circles_loc:
                component_list.append(circle)

        if len(rectangles_loc)!=0:
            for rectangle in rectangles_loc:
                component_list.append(rectangle)

        if len(connectors_loc)!=0: ##### Need to make consistent format #####
            if len(connectors_loc[0])!=0:
                for connector in connectors_loc[0]:
                    component_list.append(connector)

        if len(tools_loc)!=0:
            if len(tools_loc[0])!=0:
                for tool in tools_loc[0]:
                    component_list.append(tool)

        ##### Main Code #####
        self.is_fail = False
        if step_num > 2:
            find_mid = self.find_mid
            K = self.K
            mid_RT = self.mid_RT
            mid_id_list = self.mid_id_list
            RTs_dict = self.RTs

            parts_info = self.parts_info
            hole_pairs = self.hole_pairs
            mid_base = self.mid_base

            parts_info, hole_pairs, mid_base, self.is_fail = self.hole_detector_init.main_hole_detector(step_num, step_images, parts_info, connectors, mults, \
            mid_id_list, K, mid_RT, RTs_dict, hole_pairs, component_list, mid_base, find_mid, used_parts=self.used_parts[step_num-1], fasteners_loc=self.fasteners_loc)

            self.parts_info = parts_info
            self.hole_pairs = hole_pairs
            self.mid_base = mid_base
        else:
            find_mid = False
            K = self.K
            mid_id_list = list()
            mid_RT = np.zeros([3,4])
            RTs_dict = self.RTs

            new_id_list = self.parts_info[step_num]
            new_pose_list = self.pose_return_dict[step_num]
            assert len(new_id_list) == len(new_pose_list)

            parts_info_list = list()
            for i in range(len(new_id_list)):
                part_info = (new_id_list[i], new_pose_list[i])
                parts_info_list.append(part_info)
            self.parts_info[step_num] = parts_info_list.copy()
            parts_info = self.parts_info
            hole_pairs = self.hole_pairs
            mid_base = self.mid_base

            parts_info, hole_pairs, mid_base, self.is_fail = self.hole_detector_init.main_hole_detector(step_num, step_images, parts_info, connectors, mults, \
            mid_id_list, K, mid_RT, RTs_dict, hole_pairs, component_list, mid_base, find_mid, fasteners_loc=self.fasteners_loc)

            self.parts_info = parts_info
            self.hole_pairs = hole_pairs
            self.mid_base = mid_base

