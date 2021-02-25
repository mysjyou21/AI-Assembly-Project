""" cut: png image, step: a step corresponds to a step number, a cut can consist of multiple steps
Caution: Class 변수 추가할 때, __init__에 적어주기(모두가 어떤 variable이 있는지 쉽게 파악하게 하기위해) """
import copy
import sys
sys.path.append('./function')
sys.path.append('./function/OCR_new')
sys.path.append('./function/utilities')

from config import *
from function.utilities.utils import *  # set_connectors, set_steps_from_cut
from function.frcnn.DetectionModel import DetectionModel
from function.numbers import *
from function.OCRs_new import *
from pascalvoc import pascalvoc_calculate_iou
from pathlib import Path
import editdistance as ed
from collections import OrderedDict
import json
import shutil
import platform

# sys.path.append('./function/Pose')
# sys.path.append('./function/retrieval/codes')
# sys.path.append('./function/retrieval/codes/miscc')
# sys.path.append('./function/retrieval/render')
# from function.action import *
# from function.bubbles import *
# from function.mission_output import *
# from function.retrieval.codes.DCA import DCA
# from function.retrieval.render.render_run import *
# from function.Pose.pose_net import POSE_NET
# from function.hole import *


class Assembly():

    def __init__(self, opt):
        self.opt = opt
        self.cuts = []
        self.cut_names = []

        # Detection variables
        # component detection results, (x, y, w, h)
        self.circles_loc = []
        self.rectangles_loc = []
        self.connectors_serial_imgs = []
        self.connectors_serial_loc = []
        self.connectors_mult_imgs = []
        self.connectors_mult_loc = []
        self.connectors_loc = []
        self.parts_loc = []
        self.tools_loc = []
        self.fp_triggers_loc = []

        # component recognition results, string
        self.connectors_serial_OCR = []
        self.connectors_mult_OCR = []

        # detection & OCR results
        self.editdistance = None
        self.AP = []
        self.mAP = None

        self.data_loader()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph_detect1 = tf.Graph()
        with self.graph_detect1.as_default():
            self.sess_detect1 = tf.Session(config=config)
            self.detect_model1 = DetectionModel(self.cut_names, self.opt.det_config1_name, self.graph_detect1,
                                                self.sess_detect1)
        self.graph_detect2 = tf.Graph()
        with self.graph_detect2.as_default():
            self.sess_detect2 = tf.Session(config=config)
            self.detect_model2 = DetectionModel(self.cut_names, self.opt.det_config2_name, self.graph_detect2,
                                                self.sess_detect2)
        self.graph_OCR = tf.Graph()
        with self.graph_OCR.as_default():
            self.sess_OCR = tf.Session(config=config)
            self.OCR_model = OCRModel(self)

    def data_loader(self):
        """
        Load whole cut(.png) images, and save cut images in the self.cuts.
        self.cuts : assembly image list
        """
        # Load all images in the 'self.opt.cut_path'
        cut_paths = sorted(glob.glob(os.path.join(self.opt.cut_path, '*.png')))
        self.cut_names = [os.path.basename(cutpath) for cutpath in cut_paths]
        cuts = [np.asarray(Image.open(cut_paths[n]))[:, :, :3] for n in range(len(cut_paths))]
        for cut in cuts:
            self.cuts.append(cut)


    def detect_cut_component(self, cut, cut_number):
        """
        Detect components in a whole cut.
        cut: whole cut image
        """
        # Detect components
        self.cut_connectors, self.cut_tools, self.cut_circles, self.cut_rectangles, self.cut_parts, self.cut_fp_triggers = self.component_detector(cut, cut_number)
        self.cut_connectors_mult_imgs, self.cut_connectors_mult_loc = self.mult_detector(cut, cut_number)
        self.cut_connectors_serial_imgs, self.cut_connectors_serial_loc = self.serial_detector(cut, cut_number)

        # Merge close rectangles
        self.merge_rectangles()

        # Remove objects inside rectangles
        self.object_remover()

        # Postprocess bboxes (optional)
        if self.opt.postprocess_bboxes:
            self.postprocess_bboxes(cut_number)

        # # Set components as the cut's components
        # self.connectors_serial_imgs.append(self.cut_connectors_serial_imgs)
        # self.connectors_serial_loc.append(self.cut_connectors_serial_loc)
        # self.connectors_mult_imgs.append(self.cut_connectors_mult_imgs)
        # self.connectors_mult_loc.append(self.cut_connectors_mult_loc)
        # self.circles_loc.append(self.cut_circles)
        # self.rectangles_loc.append(self.cut_rectangles)
        # self.connectors_loc.append(self.cut_connectors)
        # self.tools_loc.append(self.cut_tools)
        # self.parts_loc.append(self.cut_parts)
        # self.fp_triggers_loc.append(self.cut_fp_triggers)

        # Set components as the cut's components
        self.connectors_serial_imgs.append(self.cut_connectors_serial_imgs)
        self.connectors_serial_loc.append(self.cut_connectors_serial_loc)
        self.connectors_mult_imgs.append(self.cut_connectors_mult_imgs)
        self.connectors_mult_loc.append(self.cut_connectors_mult_loc)
        self.circles_loc.append(self.cut_circles[:-1])
        self.rectangles_loc.append(self.cut_rectangles[:-1])
        self.connectors_loc.append(self.cut_connectors[:-1])
        self.tools_loc.append(self.cut_tools[:-1])
        self.parts_loc.append(self.cut_parts[:-1])
        self.fp_triggers_loc.append(self.cut_fp_triggers[:-1])

        # OCR
        self.connector_serial_OCR, self.connector_mult_OCR = self.OCR(cut, cut_number)
        self.connectors_serial_OCR.append(self.connector_serial_OCR)
        self.connectors_mult_OCR.append(self.connector_mult_OCR)

        # visualization - circles, rectangles, connectors, tools, serials, mults, parts, fp triggers
        colors = {'1': (62, 42, 255), '2': (56, 98, 41), '3': (11, 246, 101), '4': (180, 230, 140), '5': (20, 183, 173), '6': (173, 75, 91), '7': (177, 56, 204)}
        cut_img = np.copy(cut)

        with open(os.path.join(self.opt.detection_bbox_path, self.cut_names[cut_number][:-4]+'.txt'), 'w') as f:
            # circles
            for circle_loc in self.cut_circles:
                x, y, w, h, prob = circle_loc
                draw_bbox(cut_img, (x, y, w, h), prob, 'Guidance_Circle', colors['1'])
                f.write('%s %f %d %d %d %d\n' % ('Guidance_Circle', prob, x, y, x+w, y+h))

            # rectangles
            for rectangle_loc in self.cut_rectangles:
                x, y, w, h, prob = rectangle_loc
                draw_bbox(cut_img, (x, y, w, h), prob, 'Guidance_Square', colors['2'])
                f.write('%s %f %d %d %d %d\n' % ('Guidance_Square', prob, x, y, x + w, y + h))

            # connectors
            for connector_loc in self.cut_connectors:
                x, y, w, h, prob = connector_loc
                draw_bbox(cut_img, (x, y, w, h), prob, 'Elements', colors['3'])
                f.write('%s %f %d %d %d %d\n' % ('Elements', prob, x, y, x + w, y + h))

            # tools
            for tool_loc in self.cut_tools:
                x, y, w, h, prob = tool_loc
                draw_bbox(cut_img, (x, y, w, h), prob, 'Tool', colors['4'])
                f.write('%s %f %d %d %d %d\n' % ('Tool', prob, x, y, x + w, y + h))

            # parts
            for part_loc in self.cut_parts:
                x, y, w, h, prob = part_loc
                draw_bbox(cut_img, (x, y, w, h), prob, 'Parts', colors['5'])
                f.write('%s %f %d %d %d %d\n' % ('Parts', prob, x, y, x + w, y + h))

            # fp triggers
            for fp_trigger_loc in self.cut_fp_triggers:
                x, y, w, h, prob = fp_trigger_loc
                draw_bbox(cut_img, (x, y, w, h), prob, 'FP Trigger', colors['6'])
                f.write('%s %f %d %d %d %d\n' % ('FP Trigger', prob, x, y, x + w, y + h))

        # serials & serials OCR
        for n, serial_loc in enumerate(self.cut_connectors_serial_loc):
            x, y, w, h, angle = serial_loc
            theta = angle * np.pi / 180
            w, h = w / 2, h / 2
            pt1 = [int(x - w * np.cos(theta) - h * np.sin(theta)), int(y + w * np.sin(theta) - h * np.cos(theta))]
            pt2 = [int(x - w * np.cos(theta) + h * np.sin(theta)), int(y + w * np.sin(theta) + h * np.cos(theta))]
            pt3 = [int(x + w * np.cos(theta) + h * np.sin(theta)), int(y - w * np.sin(theta) + h * np.cos(theta))]
            pt4 = [int(x + w * np.cos(theta) - h * np.sin(theta)), int(y - w * np.sin(theta) - h * np.cos(theta))]
            pts = np.array([pt1, pt2, pt3, pt4], dtype=int)
            pts = np.reshape(pts, (-1, 1, 2))
            cv.polylines(cut_img, [pts], True, colors['7'], 2)
            # temp!!###########
            # f.write('{},{},{},{},{},{},{},{},{}\n'.format(cut_number, pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1], pt4[0], pt4[1]))
            ###################
            serial_OCR = self.connector_serial_OCR[n]
            margin = 20
            px = int(np.min([pt[0] for pt in [pt1, pt2, pt3, pt4]]) - margin)
            py = int(np.max([pt[1] for pt in [pt1, pt2, pt3, pt4]]) + margin)
            cv.putText(cut_img, serial_OCR, (px, py), cv.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=colors['7'],
                       thickness=2)

        # mults & mults OCR
        for n, mult_loc in enumerate(self.cut_connectors_mult_loc):
            x, y, w, h = mult_loc
            x1, y1, x2, y2 = x, y, x+w, y+h
            cv.rectangle(cut_img, (x1, y1), (x2, y2), colors['7'], 2)
            # temp!!################
            # f.write(os.path.join('./input/test/cuts', self.cut_names[cut_number]) + ',{},{},{},{}'.format(x1,y1,x2,y2))
            ########################
            mult_OCR = self.connector_mult_OCR[n]
            margin = 5
            px = int(x + w / 2)
            py = int(y - margin)
            cv.putText(cut_img, mult_OCR, (px, py), cv.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=colors['7'],
                       thickness=2)

        if not os.path.exists(self.opt.group_image_path):
            os.makedirs(self.opt.group_image_path)

        img_name = os.path.join(self.opt.group_image_path, self.cut_names[cut_number])
        cv.imwrite(img_name, cut_img)


    def component_detector(self, cut, cut_number):
        """
        Detect the components in the cut image, return the detected components' locations (x, y, w, h).
        component list: (using Faster R-CNN) connector(image), tool(image), circle, rectangle
        """
        cut_img = np.copy(cut)

        components_dict1 = self.detect_model1.test(cut_img, self.opt, cut_number)
        components_dict2 = self.detect_model2.test(cut_img, self.opt, cut_number)

        circles = components_dict1['Guidance_Circle']
        if 'Guidance_Sqaure' in components_dict1.keys():
            rectangles = components_dict1['Guidance_Sqaure']
        else:
            rectangles = components_dict1['Guidance_Square']
        connectors = components_dict1['Elements']
        tools = components_dict1['Tool']
        fptriggers = components_dict1['FP Trigger']
        parts = components_dict2['Mid'] + components_dict2['New']

        return connectors, tools, circles, rectangles, parts, fptriggers


    def serial_detector(self, cut, cut_number):
        serial_loc = serial_detect(self, cut, cut_number)
        return serial_loc


    def mult_detector(self, cut, cut_number):
        """
        check multiple_numbers in a step-image
        return: mult_loc = [x, y, w, h]
        """
        mult_loc = mult_detect(self, cut, cut_number)
        return mult_loc


    def OCR(self, cut, cut_number):  # 준형
        """ OCR this step's serial numbers and multiple numbers
        return: connector_serial_OCR, connector_mult_OCR
        """

        # serials_list: 예를 들어 현재 cut에 부품 번호가 2개 있으면
        # [[img1, ..., img6], [img1, ..., img6]] 와 같은 형식. imgN은 N번째 자리에 해당하는 숫자의 이미지(array).
        # mults_list도 마찬가지

        # ylist = [loc[1] for loc in self.connectors_serial_loc[cut_number]]
        # index_array = np.argsort(ylist)
        # serials_list_temp = []
        # for n in range(len(index_array)):
        #     serials_list_temp.append(self.connectors_serial_imgs[cut_number][index_array[n]])
        # self.connectors_serial_imgs[cut_number] = serials_list_temp

        serials_list = self.connectors_serial_imgs[cut_number]
        mults_list = self.connectors_mult_imgs[cut_number]
        for i in range(len(serials_list)-1, -1, -1):
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


    def merge_rectangles(self):
        if len(self.cut_rectangles) != 2:
            pass
        else:
            rec1, rec2 = self.cut_rectangles[0], self.cut_rectangles[1]
            if np.abs(rec1[1] + rec1[3] - rec2[1]) < 20 or np.abs(rec1[1] - rec2[1] - rec2[3]) < 20:
                x = np.min((rec1[0], rec2[0]))
                y = np.min((rec1[1], rec2[1]))
                w = np.max((rec1[2], rec2[2]))
                h = np.max((rec2[1]+rec2[3]-rec1[1], rec1[1]+rec1[3]-rec2[1]))
                prob = np.max((rec1[4], rec2[4]))
                self.cut_rectangles = [[x, y, w, h, prob]]
            else:
                pass


    def object_remover(self):
        # remove objects inside rectangles.
        for rectangle in self.cut_rectangles:
            x_rec, y_rec, w_rec, h_rec, _ = rectangle
            for obj_loc_list in [self.cut_circles, self.cut_tools, self.cut_connectors, self.cut_parts,
                                 self.cut_connectors_mult_loc, self.cut_connectors_serial_loc]:
                remove_index_list = []
                for n, obj_loc in enumerate(obj_loc_list):
                    x, y, w, h = obj_loc[:4]
                    if (x >= x_rec-20) and (x + w < x_rec + w_rec + 20) and (y >= y_rec-20) and (y + h <= y_rec + h_rec + 20):
                        obj_loc_list.remove(obj_loc)
                        remove_index_list.append(n)

                if obj_loc_list in [self.cut_connectors_mult_loc, self.cut_connectors_serial_loc]:
                    for index in remove_index_list[::-1]:
                        self.connectors_serial_imgs.remove(self.connectors_serial_imgs[index])


    def postprocess_bboxes(self, cut_number):

        loc_lists = [self.cut_connectors, self.cut_tools, self.cut_circles, self.cut_rectangles, self.cut_parts, self.cut_fp_triggers]

        for loc_list in loc_lists:
            for n, loc in enumerate(loc_list):
                x, y, w, h, prob = loc
                x1, y1, x2, y2 = x, y, x+w, y+h
                img_cropped = np.copy(self.cuts[cut_number][y:y+h, x:x+w, :])
                img_cropped_inv = 255 - cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
                H, W = img_cropped_inv.shape

                # case 1: sum(top_row) == 0
                if np.sum(img_cropped_inv[0, :]) == 0:
                    vertical_sum = np.sum(img_cropped_inv, axis=1)
                    not_zero_index = np.squeeze(np.argwhere(vertical_sum != 0))[0]
                    y1 += not_zero_index

                # case 2: sum(bottom_row) == 0
                if np.sum(img_cropped_inv[-1, :]) == 0:
                    vertical_sum = np.sum(img_cropped_inv, axis=1)
                    vertical_sum = np.flip(vertical_sum, axis=0)
                    not_zero_index = np.squeeze(np.argwhere(vertical_sum != 0))[0]
                    y2 -= not_zero_index

                # case 3: sum(left_column) == 0
                if np.sum(img_cropped_inv[:, 0]) == 0:
                    horizontal_sum = np.sum(img_cropped_inv, axis=0)
                    not_zero_index = np.squeeze(np.argwhere(horizontal_sum != 0))[0]
                    x1 += not_zero_index

                # case 4: sum(right_column) == 0
                if np.sum(img_cropped_inv[:, -1]) == 0:
                    horizontal_sum = np.sum(img_cropped_inv, axis=0)
                    horizontal_sum = np.flip(horizontal_sum, axis=0)
                    not_zero_index = np.squeeze(np.argwhere(horizontal_sum != 0))[0]
                    x2 -= not_zero_index

                new_loc = [x1, y1, x2-x1, y2-y1, prob]
                loc_list[n] = new_loc


    def evaluate(self):
        OCR_recog_rate = self.calculate_OCR_recognition_rate()
        mAP = self.calculate_mAP()
        recall = self.calculate_recall()
        mean_score = (OCR_recog_rate + 6*mAP + recall) / 8
        print('---------------')
        print('Mean score: %.2f%%' % mean_score)

    def calculate_OCR_recognition_rate(self):
        """
        Calculate OCR recognition rate for serial & mult numbers.
        :return: OCR recognition rate(= 1-mean edit distance) as percentage
        """
        OCR_label = os.path.join(self.opt.OCR_labelpath, 'OCR_labels.txt')
        with open(OCR_label, 'r') as f:
            lines = f.readlines()

        sum_distance = 0.
        total_numbers = 0
        for n, line in enumerate(lines):
            # OCR labels: label_serials, label_mults
            imagename = os.path.basename(line.split(' ')[0])
            label_serials = line.split(' ')[1]
            label_serials = eval(label_serials)
            label_mults = line.split(' ')[2]
            if '\n' in label_mults:
                label_mults = label_mults[:-1]
            label_mults = eval(label_mults)

            # OCR results: cut_serials, cut_mults
            cut_serials = (self.connectors_serial_OCR[n]).copy()
            cut_mults = (self.connectors_mult_OCR[n]).copy()

            # calculate edit distance for serials
            if len(label_serials) > len(cut_serials):
                diff = len(label_serials) - len(cut_serials)
                for _ in range(diff):
                    cut_serials.append('')
            elif len(label_serials) < len(cut_serials):
                diff = len(cut_serials) - len(label_serials)
                for _ in range(diff):
                    label_serials.append('')
            else: # len(label_serials) == len(cut_serials)
                pass

            total_numbers += len(label_serials)

            for label_serial in label_serials:
                distance = min([get_editdistance(label_serial, cut_serial) for cut_serial in cut_serials])
                sum_distance += distance

            # calculate edit distance for mults
            if len(label_mults) > len(cut_mults):
                diff = len(label_mults) - len(cut_mults)
                for _ in range(diff):
                    cut_mults.append('')
            elif len(label_mults) < len(cut_mults):
                diff = len(cut_mults) - len(label_mults)
                for _ in range(diff):
                    label_mults.append('')
            else: # len(label_mults) == len(cut_mults)
                pass

            total_numbers += len(label_mults)

            for label_mult in label_mults:
                distance = min([get_editdistance(label_mult, cut_mult) for cut_mult in cut_mults])
                sum_distance += distance

        mean_distance = sum_distance / total_numbers

        # print('mean edit distance: %.3f' % mean_distance)
        recog_rate = 100 * (1 - mean_distance)
        print('OCR recognition rate: %.2f%%' % recog_rate)
        return recog_rate


    def calculate_mAP(self):
        """
        Calculate mAP(mean Average Precision) of
        [elements, parts, tools, circle guidances, rectangle guidances, FP triggers].
        :return: mAP as percentage.
        """
        # parse original GT annotations image by image
        detection_label_path = self.opt.det_annpath
        detection_label_indiv_path = self.opt.det_ann_indivpath
        detection_bbox_path = self.opt.detection_bbox_path
        refresh_folder(detection_label_indiv_path)

        with open(os.path.join(detection_label_path, 'detection_label.txt'), 'r') as fr:
            lines = fr.readlines()

        imagenames = []
        for line in lines:
            imagepath = line.split(',')[0]
            imagename = os.path.basename(imagepath)
            imagenames.append(imagename)

        for n, imagename in enumerate(imagenames):
            line = lines[n]
            x1, y1, x2, y2 = line.split(',')[1:5]
            objtype = line.split(',')[-1]
            if '\n' in objtype:
                objtype = objtype[:-1]
            filename_to_be_saved = os.path.join(detection_label_indiv_path, imagename[:-4] + '.txt')

            with open(filename_to_be_saved, 'a') as fa:
                fa.write('%s %s %s %s %s\n' % (objtype, x1, y1, x2, y2))

        # mAP = os.system('python pascalvoc.py -gt %s -det %s -sp %s -np' %
        #                 (detection_label_indiv_path, detection_bbox_path, self.opt.output_path))
        mAP = pascalvoc_calculate_iou(detection_label_indiv_path, detection_bbox_path)
        return mAP


    def calculate_recall(self, iou_threshold=0.5):
        """
        Calcutate recall(=TP/(TP+FN)) for serial & mult bboxes.
        """
        # load GT anns from txt file
        detection_label_path = self.opt.det_annpath
        with open(os.path.join(detection_label_path, 'detection_label_cc.txt'), 'r') as f:
            lines = f.readlines()

        # GT ann dicts for serial, mult
        GT_serial_dict = OrderedDict()
        GT_mult_dict = OrderedDict()
        for line in lines:
            cutname = os.path.basename(line.split(',')[0])
            points = line.split(',')[1:-1]
            points = [int(point) for point in points]
            type = line.split(',')[-1]
            if '\n' in type:
                type = type[:-1]

            if type == 'Serial':
                if cutname not in GT_serial_dict:
                    GT_serial_dict[cutname] = []
                GT_serial_dict[cutname].append(points)
            elif type == 'Mult':
                if cutname not in GT_mult_dict:
                    GT_mult_dict[cutname] = []
                GT_mult_dict[cutname].append(points)

        # Form dicts from detected serial & mult locations
        # for serials: [x, y, w, h, angle] -> [x1,y1,x2,y2,x3,y3,x4,y4]
        # for mults: [x, y, w, h] -> [x1,y1,x2,y2]
        det_serial_dict = OrderedDict()
        det_mult_dict = OrderedDict()

        cut_names = self.cut_names
        for n, cutname in enumerate(cut_names):
            # serials
            if cutname not in det_serial_dict and len(self.connectors_serial_loc[n]) != 0:
                det_serial_dict[cutname] = []
            serial_locs = self.connectors_serial_loc[n]
            for serial_loc in serial_locs:
                x, y, w, h, angle = serial_loc
                theta = angle * np.pi / 180
                w, h = w / 2, h / 2
                x1, y1 = int(x - w * np.cos(theta) - h * np.sin(theta)), int(y + w * np.sin(theta) - h * np.cos(theta))
                x2, y2 = int(x - w * np.cos(theta) + h * np.sin(theta)), int(y + w * np.sin(theta) + h * np.cos(theta))
                x3, y3 = int(x + w * np.cos(theta) + h * np.sin(theta)), int(y - w * np.sin(theta) + h * np.cos(theta))
                x4, y4 = int(x + w * np.cos(theta) - h * np.sin(theta)), int(y - w * np.sin(theta) - h * np.cos(theta))
                serial_loc_new = [x1, y1, x2, y2, x3, y3, x4, y4]
                det_serial_dict[cutname].append(serial_loc_new)
            # mults
            if cutname not in det_mult_dict and len(self.connectors_mult_loc[n]) != 0:
                det_mult_dict[cutname] = []
            mult_locs = self.connectors_mult_loc[n]
            for mult_loc in mult_locs:
                x, y, w, h = mult_loc
                x1, y1, x2, y2 = x, y, x+w, y+h
                mult_loc_new = [x1, y1, x2, y2]
                det_mult_dict[cutname].append(mult_loc_new)

        # Calculate recall
        num_T = 0
        num_TP = 0
        for cutname in GT_serial_dict:
            GT_serial_locs = GT_serial_dict[cutname]
            num_T += len(GT_serial_locs)
            if cutname not in det_serial_dict:
                continue
            else:
                det_serial_locs = det_serial_dict[cutname]
                for GT_serial_loc in GT_serial_locs:
                    iou = np.max([iou_serial(GT_serial_loc, det_serial_loc) for det_serial_loc in det_serial_locs])
                    if iou >= iou_threshold:
                        num_TP += 1
        for cutname in GT_mult_dict:
            GT_mult_locs = GT_mult_dict[cutname]
            num_T += len(GT_mult_locs)
            if cutname not in det_mult_dict:
                continue
            else:
                det_mult_locs = det_mult_dict[cutname]
                for GT_mult_loc in GT_mult_locs:
                    iou = np.max([iou_mult(GT_mult_loc, det_mult_loc) for det_mult_loc in det_mult_locs])
                    if iou >= iou_threshold:
                        num_TP += 1

        recall = num_TP / num_T
        print('---------------')
        print('recall for serial & mult bboxes: %.2f%%' % (100 * recall))
        return 100*recall
