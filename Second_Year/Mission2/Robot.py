""" cut: png image, step: a step corresponds to a step number, a cut can consist of multiple steps
Caution: Class 변수 추가할 때, __init__에 적어주기(모두가 어떤 variable이 있는지 쉽게 파악하게 하기위해) """

import sys

sys.path.append('./function')
sys.path.append('./function/OCR')
sys.path.append('./function/Pose')
sys.path.append('./function/utilities')
from config import *
from function.utilities.utils import *  # set_connectors, set_steps_from_cut
from function.frcnn.DetectionModel import DetectionModel
from function.action import *
from function.bubbles import *
from function.numbers import *
from function.mission_output import *
from function.OCRs_new import *
from function.Pose.evaluation.initial_pose_estimation_2 import InitialPoseEstimation
from function.hole import *
from function.Grouping_mid.hole_loader import base_loader, mid_loader
from function.Grouping_mid.grouping_RT import transform_hole, baseRT_to_midRT
from function.Hole_detection.MSN2_hole_detector import MSN2_hole_detector

from pathlib import Path
import json
import shutil
import platform

from keras import backend as K

class Assembly():

    def __init__(self, opt):
        self.opt = opt
        self.cuts = []
        self.cut_names = []
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
        # 이삭 to 민우 : 이거 두 개도 중간 step부터 시작할때 loading 되게 해야 함
        self.unused_parts = {1 : [1, 2, 3, 4, 5, 6, 7, 8]} # unused parts {step_num : list of part ids}
        self.used_parts = {} # used parts {step_num : list of part ids}

        # component recognition results, string
        self.connectors_serial_OCR = {}
        self.connectors_mult_OCR = {}

        # Retrieval variables
        self.part_H = self.part_W = 224
        self.parts = {}  # detected part images {step_num  : list of part images}
        self.parts_bboxed = {}  # detected part images shown as bbox on whole image {step_num  : list of part images}
        self.parts_info = {}  # {step_num: list of (part_id, part_pos, hole_info)}
        self.cad_models = {}  # names of cad models of retrieval results
        self.candidate_classes = []  # candidate cad models for retrieval (not used)
        self.hole_pairs = {}

        # Pose variables
        self.pose_return_dict = {} # RT of detected part images {step_num : list of RT (type : np.array, shape : [3, 4])}
        self.pose_save_dict = {} # RT of detected part images (save version) {000000 : step_num(6 digit string) : list of obj_dicts} ("cam_R_m2c", "cam_t_m2c", "obj_id" in obj_dict)
        self.pose_save_dict['000000'] = {}
        self.cad_names = [os.path.basename(x).split('.')[0] for x in sorted(glob.glob(self.opt.cad_path + '/*.obj'))] # cad names in cad_path
        self.K = np.array([
            3444.4443359375,0.0,874.0,
            0.0,3444.4443359375,1240.0,
            0.0,0.0,1.0]).reshape(3,3) # camera intrinsic matrix
        self.RTs = np.load(self.opt.pose_data_path + '/RTs.npy') # 48 RTs to compare with pose_network prediction
        self.VIEW_IMGS = np.load(self.opt.pose_data_path + '/view_imgs.npy') # VIEWS to display pose output

        # Final output
        self.actions = {}  # dictionary!! # [part1_loc, part1_id, part1_pos, part2_loc, part2_id, part2_pos, connector1_serial_OCR, connector1_mult_OCR, connector2_serial_OCR, connector2_mult_OCR, action_label, is_part1_above_part2(0,1)]
        self.step_action = []

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
        self.cut_names = cut_paths
        print([os.path.basename(x) for x in cut_paths])
        cuts = [np.asarray(Image.open(cut_paths[n]))[:, :, :3] for n in range(len(cut_paths))]
        # 준형: resize, cut만 읽어올 수 있게 추가(material 말고), preprocessing(양탄자 처리)
        for cut in cuts:
            self.cuts.append(cut)

        idx = 1
        for _, cut in enumerate(cuts):
            # resize
            cut_resized = resize_cut(cut)
            # preprocessing
            cut_resized = prep_carpet(cut_resized)
            # material 소개 cut 제외
            if is_valid_cut(cut_resized):
                self.steps[idx] = cut_resized
                self.num_steps += 1
                idx += 1


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

        components_dict1 = self.detect_model1.test(step_img)
        components_dict2 = self.detect_model2.test(step_img)
        # print(components_dict1)
        # print(components_dict2)
        circles = components_dict1['Guidance_Circle']
        if 'Guidance_Sqaure' in components_dict1.keys():
            rectangles = components_dict1['Guidance_Sqaure']
        else:
            rectangles = components_dict1['Guidance_Square']
        connectors = components_dict1['Elements']
        tools = components_dict1['Tool']
        # parts = components_dict2['Mid'] + components_dict2['New']
        part_ids = sorted(list(components_dict2.keys()))
        part_ids.remove('bg')
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

        # DIVIDE step-components into several groups, regarding connector_mult-connector_serial pairs!
        step_circles_new, step_connectors_new, step_connectors_serial_new, step_connectors_mult_new, step_tools_new, \
        step_connectors_serial_OCR_new, step_connectors_mult_OCR_new, is_merged = grouping(step_circles,
                                                                                           step_rectangles,
                                                                                           step_connectors,
                                                                                           step_connectors_serial,
                                                                                           step_connectors_mult,
                                                                                           step_tools,
                                                                                           step_connectors_serial_OCR,
                                                                                           step_connectors_mult_OCR)

        # update variables
        self.circles_loc[step_num] = step_circles_new
        self.circles_separated_loc[step_num] = step_circles
        self.connectors_loc[step_num] = step_connectors_new
        self.connectors_serial_loc[step_num] = step_connectors_serial_new
        self.connectors_mult_loc[step_num] = step_connectors_mult_new
        self.tools_loc[step_num] = step_tools_new
        self.connectors_serial_OCR[step_num] = step_connectors_serial_OCR_new
        self.connectors_mult_OCR[step_num] = step_connectors_mult_OCR_new
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
        true_serials = [['100001'], ['101350'], ['122620'], ['104322'], ['122925']]

        for serials in serials_list:
            serial_OCR = self.OCR_model.run_OCR_serial(args=self, imgs=serials)

            if (serial_OCR not in true_serials) and (serial_OCR != ''):
                count = [0, 0, 0, 0, 0]
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
        self.part_dir = self.opt.cad_path + '/' + str(step_num)
        if not os.path.exists(self.part_dir):
            os.mkdir(self.part_dir)

        # detection classification results
        # retrieved_classes = self.parts_info[step_num]
        self.cad_models[step_num] = self.parts_info[step_num].copy()
        assert len(self.parts[step_num]) == len(self.cad_models[step_num]), 'length of retrieval input/output don\'t match'
        print('')
        print('classified classes : ', self.cad_models[step_num]) #blue

        # pose prediction
        # update self.pose_return_dict, self.cad_models
        self.pose_model.test(self, step_num)


        ########################################################################################
        # 이삭 to 민우 : 이거 옮겨 간다고 하지 않았나...
        def closest_gt_RT_index(RT_pred):
            return np.argmin([np.linalg.norm(RT - RT_pred) for RT in self.RTs])

        matched_poses = [closest_gt_RT_index(RT_pred) for RT_pred in self.pose_return_dict[step_num]]
        if self.opt.save_part_id_pose:
            self.pose_model.save_part_id_pose(self, step_num, matched_poses)

        print('modified classified classes : ', self.cad_models[step_num]) #blue
        classified_classes = self.cad_models[step_num]
        # parts_info update
        print('matched_poses', matched_poses) #blue
        # update self.parts_info[step_num]
        holes = [[] for x in range(len(matched_poses))]
        connectivity = ''
        self.parts_info[step_num] = list(zip(classified_classes, matched_poses, holes))  #-> 앞으로
        self.parts_info[step_num] = [list(x) for x in self.parts_info[step_num]]
        self.parts_info[step_num].append(connectivity)
        print(self.parts_info[step_num])
        ##########################################################################################

    def group_as_action(self, step_num):   # action 관련 정보들을 묶어서 self.actions에 넣고 ./output에 json write
        """ Group components in action-unit
        [part1_loc, part1_id, part1_pos, part2_loc, part2_id, part2_pos, connector1_serial_OCR, connector1_mult_OCR, connector2_serial_OCR, connector2_mult_OCR, action_label, is_part1_above_part2(0,1)]
        Update self.actions """

        material = self.connectors_serial_OCR[step_num]
        circle_mult = self.connectors_mult_OCR[step_num]

        f = open(os.path.join('function', 'utilities', 'action_label.csv'), 'r')  # , encoding='utf-8')
        csv_reader = csv.reader(f)
        act_dic = {}  # key: '100001' value: ['A001']
        next(csv_reader)  # ignore first line
        for line in csv_reader:
            part_lab = line[0]
            act_dic[part_lab] = line[1:]
        f.close()
        connectivity = self.parts_info[step_num][-1]

        if len(material) == 1 and material != [[]]:
            serials, circle_action, circle_num = map_action(self, material[0], circle_mult[0], act_dic, step_num)
            if connectivity == '': # material은 1개 인데 connectivity가 X -> step1 같은 상황
                action_group_step = []
                for p_ind in range(max(0, len(self.parts_info[step_num])-1)):
                    parts_info = self.parts_info[step_num][p_ind]
#                    parts_info = [self.parts_loc[step_num][p_ind]] + [x for x in self.parts_info[step_num][p_ind]]
                    temp_action_group_step = [parts_info, [''], [''], [''], serials, circle_num, circle_action]
                    action_group_step += [temp_action_group_step]
                self.actions[step_num] = action_group_step

            else: # material은 1개 인데 part가 여러개인 상황 -> 즉 모든 부품을 하나의 action으로 연결.
                action_group_step = []
                temp_action_group_step = [[''], [''], [''], [''], serials, circle_num, circle_action]
                for p_ind in range(max(0, len(self.parts_info[step_num])-1)):
                    parts_info = self.parts_info[step_num][p_ind]
#                    parts_info = [self.parts_loc[step_num][p_ind]] + [x for x in self.parts_info[step_num][p_ind]]
                    temp_action_group_step[p_ind] = parts_info
                action_group_step += [temp_action_group_step]
                self.actions[step_num] = action_group_step


        else:  # 1. error 거나(material이 여러개) 2. circle이 두개 라서 len==2인데 action은 1개 !
            action_group_step = []
            for p_ind in range(max(0, len(self.parts_info[step_num])-1)):
                parts_info = self.parts_info[step_num][p_ind]
#                parts_info = [self.parts_loc[step_num][p_ind]] + [x for x in self.parts_info[step_num][p_ind]]
                temp_action_group_step = [parts_info, [''], [''], [''], material, circle_mult, ['']]
                action_group_step += [temp_action_group_step]
            self.actions[step_num] = action_group_step

    def group_RT_mid(self, step_num):
        ######### detection에서 검출된 기본부품의 pose 반영 hole 위치 #######
        base_RT_list = self.pose_return_dict[step_num]
        base_id_list = [base_part[0] for base_part in self.parts_info[step_num][:-1]]

        if (len(base_id_list) < 2):
            pass  ######## 설명서내에 검출된 기본 part가 2개 미만인 경우 #########

        base_hole_dict = {}
        for i, id in enumerate(base_id_list):
            base_hole_XYZ = base_loader(id, self.opt.hole_path)
            base_RT = base_RT_list[i]
            base_hole_dict[id] = transform_hole(base_RT, base_hole_XYZ)


        ######## 중간산출물 hole 위치 loading  ##########
        mid_hole_dict = mid_loader('step%i'%(step_num-1), self.opt.hole_path)

        mid_RT, mid_id_list, find_mid = baseRT_to_midRT(base_hole_dict, mid_hole_dict)

        self.find_mid = find_mid
        self.mid_id_list = mid_id_list
        self.mid_RT = mid_RT

        if find_mid:
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
            pass


    def write_csv_mission(self, step_num, option=0):  # 은지(전체파트), 선지(write_csv_mission2부분 action labeling관련 다듬기)
        """ Write the results in csv file, option=1, 2, 3 is 1-year challenge's output """
        if option == 1:
            write_csv_mission1_1st_year(self.connector_serial_OCR_index, self.connector_mult_OCR_each_group,
                                        self.opt.cut_path, self.opt.csv_dir)
        elif option == 2:
            write_csv_mission2_1st_year(self.connector_serial_OCR_index, self.connector_mult_OCR_each_group,
                                        self.opt.cut_path, self.opt.csv_dir)
        elif option == 0:
            with open('./function/utilities/label_to_pose.json', 'r') as f:
                pose_dic = json.load(f)
            step_actions = self.actions[step_num]
            step_hole_pair = self.hole_pairs.get(step_num, [])
#            print('hole %d'%(step_num), step_hole_pair)
            for action in step_actions:
                for i in range(0, 4):
                    part = list(action[i])
                    if part[0] != '':
                        part_pose_ind = part[1]
                        part_pose_lab = pose_dic[str(part_pose_ind)]
                        action[i][1] = part_pose_lab.split('_')
            if step_hole_pair != []:
                step_actions[0].append(step_hole_pair)
            if step_num == 1:
                if os.path.exists(self.opt.csv_dir):
                    shutil.rmtree(self.opt.csv_dir)

            write_json_mission(step_actions, self.opt.cut_path, str(step_num), self.opt.csv_dir)

    def msn2_hole_detector(self, step_num):

        if self.find_mid:
            step_images = self.steps
            parts_info = self.parts_info
            connectors = self.connectors_serial_OCR
            mults = self.connectors_mult_OCR
            K = self.K
            mid_RT = self.mid_RT
            mid_id_list = self.mid_id_list
            RTs_dict = self.RTs
            hole_pairs = self.hole_pairs
            component_list = []

            parts_info, hole_pairs = self.hole_detector_init.main_hole_detector(step_num, step_images, parts_info, connectors, mults, \
            mid_id_list, K, mid_RT, RTs_dict, hole_pairs, component_list)

            self.parts_info = parts_info
            self.hole_pairs = hole_pairs
        else:
            pass
