""" cut: png image, step: a step corresponds to a step number, a cut can consist of multiple steps
Caution: Class 변수 추가할 때, __init__에 적어주기(모두가 어떤 variable이 있는지 쉽게 파악하게 하기위해) """

import sys
sys.path.append('./function')
sys.path.append('./function/OCR')
sys.path.append('./function/Pose')
sys.path.append('./function/utilities')
sys.path.append('./function/retrieval/codes')
sys.path.append('./function/retrieval/codes/miscc')
sys.path.append('./function/retrieval/render')
from config import *
from function.utilities.utils import *  # set_connectors, set_steps_from_cut
from function.frcnn.DetectionModel import DetectionModel
from function.action import *
from function.bubbles import *
from function.numbers import *
from function.mission_output import *
from function.OCRs_new import *
from function.retrieval.codes.DCA import DCA
from function.retrieval.render.render_run import *
from function.Pose.pose_net import POSE_NET
from function.hole import *
from pathlib import Path
import json
import shutil
import platform


class Assembly():

    def __init__(self, opt):
        self.opt = opt
        self.cuts = []
        self.steps = {}
        self.num_steps = 0

        config = tf.ConfigProto()
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
        self.graph_retrieval = tf.Graph()
        with self.graph_retrieval.as_default():
            self.sess_retrieval = tf.Session(config=config)
            self.retrieval_model = DCA(self)
        self.graph_pose = tf.Graph()
        with self.graph_pose.as_default():
            self.sess_pose = tf.Session(config=config)
            self.pose_model = POSE_NET(self)

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
        self.parts_loc = {}
        self.tools_loc = {}
        self.is_merged = {}
        self.is_tool = {}

        # component recognition results, string
        self.connectors_serial_OCR = {}
        self.connectors_mult_OCR = {}

        # Retrieval variables
        self.part_H = self.part_W = 224
        self.parts = {}  # detected part images
        self.parts_info = {}  # {step_num: list of (part_id, part_pos, hole_info)}
        self.cad_models = {}  # names of cad models of retrieval results
        self.candidate_classes = []  # candidate cad models for retrieval
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
        cut_paths = sorted(glob.glob(os.path.join(self.opt.cut_path, '*.png')))
        cuts = [np.asarray(Image.open(cut_paths[n]))[:, :, :3] for n in range(len(cut_paths))]
        for cut in cuts:
            self.cuts.append(cut)

        # for each img (in right order)
        # Extract steps from a cut
        idx = 1
        self.connectors_cuts = []
        for _, cut in enumerate(cuts):
            step_nums, is_steps, step_imgs = set_steps_from_cut(cut)

            for is_step, step_img in zip(is_steps, step_imgs):
                if is_step:
                    self.steps[idx] = step_img
                    self.num_steps += 1
                    idx += 1
                else:
                    self.connectors_cuts.append(step_img)

    def detect_step_component(self, step_num, print_result=True):  # 준형
        """
        Detect components(a grouped action information) in a step
        step_num: corresponding step number
        print_result: Print the grouped action information as a csv format
        """
        # Detect components
        self.step_connectors, self.step_tools, self.step_circles, self.step_rectangles, self.step_parts = self.component_detector(step_num)
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
        self.connector_serial_OCR, self.connector_mult_OCR = self.OCR(step_num)
        self.connectors_serial_OCR[step_num] = self.connector_serial_OCR
        self.connectors_mult_OCR[step_num] = self.connector_mult_OCR
        self.group_components(step_num)

        # visualization(준형) - image/bounding boxes (parts 제외, OCR결과도 그림에) self.opt.group_image_path
        # num of colors in palette: 7
        palette = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 128, 0), (255, 0, 128), (255, 0, 255)]
        num_groups = len(self.circles_loc[step_num])
        step_img = np.copy(self.steps[step_num])

        for n in range(num_groups):
            color = palette[n]
            # circles
            circle_loc = self.circles_loc[step_num][n]
            x, y, w, h = circle_loc
            cv.rectangle(step_img, (x, y), (x + w, y + h), color, 2)
            # connectors
            connectors_loc = self.connectors_loc[step_num][n]
            for loc in connectors_loc:
                x, y, w, h = loc
                cv.rectangle(step_img, (x, y), (x + w, y + h), color, 2)
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
                cv.polylines(step_img, [pts], True, color, 2)
                # serial_OCRs
                serial_OCR = self.connectors_serial_OCR[step_num][n][i]
                margin = 20
                px = int(np.min([pt[0] for pt in [pt1, pt2, pt3, pt4]]) - margin)
                py = int(np.max([pt[1] for pt in [pt1, pt2, pt3, pt4]]) + margin)
                cv.putText(step_img, serial_OCR, (px, py), cv.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=color, thickness=2)
            # mults
            mults_loc = self.connectors_mult_loc[step_num][n]
            for i, loc in enumerate(mults_loc):
                x, y, w, h = loc
                cv.rectangle(step_img, (x, y), (x + w, y + h), color, 2)
                mult_OCR = self.connectors_mult_OCR[step_num][n][i]
                margin = 5
                px = int(x + w / 2)
                py = int(y - margin)
                cv.putText(step_img, mult_OCR, (px, py), cv.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=color, thickness=2)

        if not os.path.exists(self.opt.group_image_path):
            os.makedirs(self.opt.group_image_path)

        img_name = os.path.join(self.opt.group_image_path, '%02d.png' % step_num)
        cv.imwrite(img_name, step_img)

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
        parts = components_dict2['Mid'] + components_dict2['New']
        return connectors, tools, circles, rectangles, parts

    def hole_detector(self, step_num, step_parts_id, step_parts_pose):
        step_img = (self.steps[step_num]).copy()
        step_parts_loc = self.parts_loc[step_num]
        step_parts_loc_origin = step_parts_loc.copy()
        y1, y2, x1, x2 = step_img.shape[0], 0, step_img.shape[1], 0
        for x, y, w, h in step_parts_loc:
            if x < x1:
                x1 = x
            if y < y1:
                y1 = y
            if x + w > x2:
                x2 = x + w
            if y + h > y2:
                y2 = y + h
        step_roi = step_img[y1 - 20:y2 + 30, x1 - 50:x2 + 20]

        ### roi modified for step_5 ###
        if step_num == 3:
            step_roi = step_img[y1 - 100:y2 + 30, x1 - 50:x2 + 20]

        elif step_num == 5:
            step_roi = step_img[y1 - 50:y2 + 30, x1 - 50:x2 + 20]

        elif step_num == 6:
            step_roi = step_img[y1 - 200:y2 + 200, x1 - 100:x2 + 100]

        for i in range(len(step_parts_loc)):
            x, y, w, h = step_parts_loc[i]
            step_parts_loc[i] = [x - (x1 - 50), y - (y1 - 20), w, h]

        step_connectors_OCR = self.connectors_serial_OCR[step_num]

        ud_check = False if (step_num == 5 or step_num == 9) else True
        in_check = False if step_num == 9 else True
        h_obj_th = 50 if (step_num == 2 or step_num == 3) else 100  # small when detection result is not correct..
        if len(step_connectors_OCR) > 0:
            if len(step_connectors_OCR[0]) > 0:
                step_connector_OCR = step_connectors_OCR[0][0]  # assumption: only one OCR result in one step
            else:
                step_connector_OCR = None
        else:
            step_connector_OCR = None
        if step_connector_OCR == '501350':  # temporary
            step_connector_OCR = '101350'
        connect_step_num_list = [1, 2, 3, 4, 5, 6, 9]
        step_connector = step_connector_OCR if step_num in connect_step_num_list else None

        if step_num == 1:
            if os.path.exists(self.opt.part_hole_path):
                shutil.rmtree(self.opt.part_hole_path)
        if not os.path.exists(self.opt.part_hole_path):
            os.makedirs(self.opt.part_hole_path)

        ############################## Hyperparameters modified for each step ########################################
        rate_max_th = None
        if step_num == 1:
            h_min_th = 50
            h_max_th = 150
            rate_min_th = 4

        elif step_num == 2:
            h_min_th = 22
            h_max_th = 100
            rate_min_th = 4

        elif step_num == 3:
            h_min_th = 1
            h_max_th = 150
            rate_min_th = 1

        elif step_num == 4:
            h_min_th = 30
            h_max_th = 150
            rate_min_th = 4

        elif step_num == 5:
            h_min_th = 55
            h_max_th = 120
            rate_min_th = 4

        elif step_num == 6:
            h_min_th = 90
            h_max_th = 150
        #     h_min_th = 10
        #     h_max_th = 500
            rate_min_th = 4

        elif step_num == 7 or step_num == 8:
            h_min_th = 15
            h_max_th = 16
            rate_min_th = 4

        elif step_num == 9:
            h_min_th = 65
            h_max_th = 150
            rate_min_th = 15
            rate_max_th = 19

        else:
            h_min_th = 15
            h_max_th = 150
            rate_min_th = 4
        hole_info = detect_fasteners(step_roi, os.path.join(self.opt.part_hole_path, '%.2d.png' % step_num),
                                        step_parts_loc, step_parts_id, ud_check, in_check, h_obj_th, h_min_th, h_max_th, rate_min_th, rate_max_th)
        ###########################################################################################
        hole_info, connectivity = convert_view_assembly_to_CAD(hole_info, step_parts_id, step_parts_pose, self.parts_loc[step_num], step_connector)

        # visualization
        for i in range(len(step_parts_loc_origin)):
            x, y, w, h = step_parts_loc_origin[i]
            step_img = cv2.rectangle(step_img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
            step_img = cv2.putText(step_img, step_parts_id[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
            holes = hole_info[i]
            if holes == []:
                continue
            for hole in holes:
                hole_type, p = hole
                px1, py1, px2, py2 = int(p[1]) + (x1 - 50) - 5, int(p[2]) + (y1 - 20) - 5, int(p[3]) + (x1 - 50) + 5, int(p[4]) + (y1 - 20) + 5
                if step_num == 3:
                    py1 -= 90
                    py2 -= 90  
                if step_num == 6:
                    px1 -= 50
                    px2 -= 50
                    py1 -= 180
                    py2 -= 180
                step_img = cv2.rectangle(step_img, (px1, py1), (px2, py2), color=(0, 255, 0), thickness=2)
                step_img = cv2.putText(step_img, hole_type[0:5], (px1, py1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0), thickness=2)
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
        step_parts = self.parts_loc[step_num]
        step_connectors_serial_OCR = self.connectors_serial_OCR[step_num]
        step_connectors_mult_OCR = self.connectors_mult_OCR[step_num]

        # DIVIDE step-components into several groups, regarding connector_mult-connector_serial pairs!
        step_circles_new, step_connectors_new, step_connectors_serial_new, step_connectors_mult_new, step_tools_new,\
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

        for serials in serials_list:
            serial_OCR = self.OCR_model.run_OCR_serial(args=self, imgs=serials)
            connector_serial_OCR.append(serial_OCR)
        for mults in mults_list:
            mult_OCR = self.OCR_model.run_OCR_mult(args=self, imgs=mults)
            connector_mult_OCR.append(mult_OCR)

        return connector_serial_OCR, connector_mult_OCR

    def retrieve_part(self, step_num, list_added_obj, list_added_stl):  # 민우
        """ Retrieve part identity from CAD models, part images as queries
        return : update self.parts_info = {step_num : (part_id(str), part_pose(index)))
            """

        if len(list_added_obj) != 0 or len(list_added_stl) != 0 or step_num == 1:

            ### Prior work for rendering(copy Cad file to new_cad directory in cad directory) ###
            cad_path = self.opt.cad_path
            list_converted_obj = []
            if len(list_added_stl) != 0:
                stl_to_obj(self)
                list_converted_obj = [os.path.splitext(s)[0] + '.obj' for s in list_added_stl]

            list_added_obj = sorted(list(set(list_added_obj) | set(list_converted_obj)))

            ### Point Cloud ###
            center_model(self)
            create_pointcloud(self, list_added_obj)

            ### Rendering ###
            # views_gray_black (for retrieval)
            create_rendering(self, list_added_obj, 'views_gray_black')
            # views (for pose)
            create_rendering(self, list_added_obj, 'views')

            ### Posterior work after rendering###
            files = sorted(glob.glob(os.path.join(self.opt.cad_path, '*.STL')))
            for file in files:
                filename = os.path.splitext(file)[0]
                if platform.system() == 'Windows':
                    os.remove(str(Path(filename + '.obj')))
                else:
                    os.system('rm ' + filename + '.obj')

        cad_list = sorted(glob.glob(self.opt.cad_path + '/*'))
        cad_list = [os.path.splitext(os.path.basename(v))[0] for v in cad_list if os.path.splitext(v)[-1] != '']
        prev_retrieval_classes = []
        if step_num > 1:
            prev_retrieval_classes = []
            for i in range(1, step_num):
                for model in self.cad_models[i]:
                    prev_retrieval_classes.append(model)
        self.candidate_classes = sorted(list(set(cad_list) - set(prev_retrieval_classes)))
        if step_num == 1:
            self.candidate_classes = [v for v in self.candidate_classes if 'part' in v]

        # crop part images from step images with detection results
        step_part_images = []
        for i, crop_region in enumerate(self.parts_loc[step_num]):
            x, y, w, h = crop_region[:4]
            step_part_image = self.steps[step_num][y:y + h, x:x + w]
            step_part_images.append(step_part_image)
        self.parts[step_num] = step_part_images

        # save detection result images
        if self.opt.save_detection:
            if not os.path.exists(self.opt.detection_path):
                os.makedirs(self.opt.detection_path)
            for i in range(len(self.parts[step_num])):
                cv2.imwrite(self.opt.detection_path + '/STEP{}_part{}.png'.format(step_num, i),
                            self.parts[step_num][i])

        # retrieval # 민우 : 1. retrieval 할때 self.steps랑 detection 결과로 image crop해서 사용 --> 이삭 해결
        #                    2.input_images를 images directory 대신 crop 된 image 그 자체의 list로 변경
        retrieved_classes = self.retrieval_model.test(self, step_num, self.candidate_classes)
        self.cad_models[step_num] = retrieved_classes
        print('\nretrieved classes : ', retrieved_classes)
        assert len(self.parts[step_num]) == len(self.cad_models[step_num]), 'length of retrieval input/output don\'t match'

        # pose
        matched_poses = self.pose_model.test(self, step_num)

        # hole
        holes, connectivity = self.hole_detector(step_num, retrieved_classes, matched_poses)
        self.parts_info[step_num] = list(zip(retrieved_classes, matched_poses, holes))
        self.parts_info[step_num].append(connectivity)
#        print('%d parts, info: ' % (len(self.parts_info[step_num]) - 1), self.parts_info[step_num])
        # part retrieval,pose 결과 : 이삭(query image | retrieved model image) self.opt.part_id_pose_path
        # part hole 결과: 은지(전체 이미지에서 bb, hole 위치, label) self.opt.part_hole_path

    def group_as_action(self, step_num):
        """ Group components in action-unit
        [part1_loc, part1_id, part1_pos, part2_loc, part2_id, part2_pos, connector1_serial_OCR, connector1_mult_OCR, connector2_serial_OCR, connector2_mult_OCR, action_label, is_part1_above_part2(0,1)]
        Update self.actions
        """
        # making action dictionary
        f = open(os.path.join('function', 'utilities', 'action_label.csv'), 'r')  # , encoding='utf-8')
        csv_reader = csv.reader(f)
        act_dic = {}  # key: '100001' value: ['A001']
        next(csv_reader)  # ignore first line
        for line in csv_reader:
            part_lab = line[0]
            act_dic[part_lab] = line[1:]
        f.close()

        cut_material = self.connectors_serial_OCR[step_num]
        cut_mult = self.connectors_mult_OCR[step_num]

        if self.tools_loc[step_num] == ([] or [[]]):
            self.is_tool[step_num] = [0]
        else:
            self.is_tool[step_num] = self.tools_loc[step_num]
            for idx, tool in enumerate(self.is_tool[step_num]):
                self.is_tool[step_num][idx] = 1 if tool != [] else 0

        if self.is_merged[step_num] == True:
            connectors, tools, circles, rectangles, parts = self.component_detector(step_num)
            circle_num = len(circles)
            material_temp = self.connectors_serial_OCR[step_num]
            mult_temp = [str(len(self.parts_info[step_num][0][2]))]#self.connectors_mult_OCR[step_num] only 9 step
            cut_material = []
            cut_mult = []

            self.is_tool[step_num] = []

            for idx, circle_loc in enumerate(circles):
                x, y, h, w = circle_loc
                circle_info = []
                for i in range(len(connectors)):
                    if (x < connectors[i][0] and connectors[i][0] < x + h):
                        circle_info += material_temp[i]
                    else:
                        circle_info += []
                for j in range(len(tools)):
                    if (x < tools[j][0] and tools[j][0] < x + h):
                        self.is_tool[step_num] += [1]
                    else:
                        self.is_tool[step_num] += [0]

                cut_material += [circle_info]
                cut_mult += [mult_temp]

            if step_num == 9: #### temp
                self.is_tool[step_num] = [1,1] ####

        ############## mapping action #############
        step_action = []
        if (cut_material == []) and len(self.parts[step_num]) == 1:
            circle_action = ['A006']
            circle_num = ['1']
            self.step_action += [[[circle_action, circle_num]]]

        for idx, material in enumerate(cut_material):
            if (cut_material == [[], []]) and len(self.parts[step_num]) == 2:
                circle_action = ['A005']
                circle_num = ['1']
                self.step_action += [[[circle_action, circle_num]]]
                self.connectors_mult_OCR[step_num] = circle_num
                self.connectors_serial_OCR[step_num] = ['']
                break

            elif material == []:
                if self.is_tool[step_num][idx] == 1:
                    circle_action = ['A003']
                    circle_num = [str(len(self.parts_info[step_num][0][2]))]  # maybe revised ?
                    step_action += [[circle_action, circle_num]]
                    self.connectors_mult_OCR[step_num] = circle_num
                    self.connectors_serial_OCR[step_num] += ['']
            else:
                circle_mult = cut_mult[idx]
                serials, circle_action, circle_num = map_action(self, material, circle_mult, act_dic, step_num)
                step_action += [[circle_action, circle_num]]  # self.step_action has every step's action as element.
                self.connectors_serial_OCR[step_num] = serials

        if step_action != []:
            self.step_action += [step_action]
        ################################################

        ################### connectivity ###############################
        connectivity = self.parts_info[step_num][-1]
        part_num = len(self.parts_info[step_num]) - 1
        connector_num = len(self.connectors_serial_OCR[step_num])

        if connectivity == '':
            if part_num > 1 and self.step_action[step_num - 1][0][0] != ['A005']:
                action = self.step_action[step_num - 1][0][0][0]
                step_action = []
                action_group_step = []
                for i in range(part_num):
                    hole_num = len(self.parts_info[step_num][i][2])
                    step_action += [[action, [str(hole_num)]]]

                    action_group = [[""], [""], [""], [""]]
                    temp = [self.parts_loc[step_num][i]]
                    temp += self.parts_info[step_num][i]
                    action_group[0] = temp  # [[parts_loc[step_num], parts_info[step_num][0], ]]  # part1 loc, id, pos, hole
                    action_group += [self.connectors_serial_OCR[step_num], [str(hole_num)], action]  # add action mult
                    action_group_step += [action_group]
                self.actions[step_num] = action_group_step

                self.step_action[step_num - 1] = step_action
                return

            else:
                pass
        else:
            pass

        #####################################################################
        if step_num != 9:
            self.connectors_serial_OCR[step_num] = [self.connectors_serial_OCR[step_num] for x in range(len(self.step_action[step_num-1]))]
        else:
            self.connectors_serial_OCR[step_num] = [[x] for x in self.connectors_serial_OCR[step_num]]
        # group every action parameters - per action
        action_group_step = []
        for idx, action in enumerate(self.step_action[step_num - 1]):
            action_group = [[""], [""], [""], [""]]
            for i in range(len(self.parts_loc[step_num])):
                temp = [self.parts_loc[step_num][i]]
                temp += self.parts_info[step_num][i]
                action_group[i] = temp  # [[parts_loc[step_num], parts_info[step_num][0], ]]  # part1 loc, id, pos, hole

            action_group += [self.connectors_serial_OCR[step_num][idx],
                             self.step_action[step_num - 1][idx][1]]  # add action mult

            action_group += self.step_action[step_num - 1][idx][0]  # action_label
            action_group_step += [action_group]

        if self.is_merged[step_num] is True:
            circle_xs = [x[0] for x in circles]
            temp_action_group_step = sorted(zip(circle_xs, action_group_step), key=lambda t: t[0])
            action_group_step = [x[1] for x in temp_action_group_step]
        self.actions[step_num] = action_group_step

    def write_csv_mission(self, step_num, option=0):  # 은지(전체파트), 선지(write_csv_mission2부분 action labeling관련 다듬기)
        """ Write the results in csv file, option=1, 2, 3 is 1-year challenge's output """
        if option == 1:
            write_csv_mission1_1st_year(self.connector_serial_OCR_index, self.connector_mult_OCR_each_group, self.opt.cut_path, self.opt.csv_dir)
        elif option == 2:
            write_csv_mission2_1st_year(self.connector_serial_OCR_index, self.connector_mult_OCR_each_group, self.opt.cut_path, self.opt.csv_dir)
        elif option == 0:
            with open('./function/utilities/label_to_pose.json', 'r') as f:
                pose_dic = json.load(f)
            step_actions = self.actions[step_num]
            for action in step_actions:
                for i in range(0, 4):
                    part = action[i]
                    if part[0] != '':
                        part_pose_ind = part[2]
                        part_pose_lab = pose_dic[str(part_pose_ind)]
                        action[i][2] = part_pose_lab.split('_')
            if step_num == 1:
                if os.path.exists(self.opt.csv_dir):
                    shutil.rmtree(self.opt.csv_dir)

            write_csv_mission(step_actions, self.opt.cut_path, str(step_num), self.opt.csv_dir)
