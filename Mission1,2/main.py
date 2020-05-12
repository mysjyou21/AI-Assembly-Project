import sys
sys.path.append('./function')
sys.path.append('./function/OCR')
sys.path.append('./function/utilities')
from config import *
from function.utilities.utils import *
from function.action import *
from function.bubbles import *
from function.numbers import *
from function.mission_output import *
from function.OCRs import *
from function.OCR.model import digit_recognizer

class Stefan():

    def __init__(self, opt):
        self.opt = opt
        self.data_loader()
        self.sess = tf.Session()

    def data_loader(self):
        """
        load images
        self.cut : assembly image list
        :return
        """
        cut_list = sorted(glob.glob(self.opt.cutpath + '/*'))
        self.cut = []
        for i, name in enumerate(cut_list):
            try:
                I = np.asarray(Image.open(name))[:, :, :3]  # [0, 255], uint8
                if i == 0:
                    I = div_cut(I)
            except:
                I = plt.imread(name)[:, :, :3]
                I = np.around(I * 255).astype(np.uint8)  # [0, 255], uint8
                if i == 0:
                    I = div_cut(I)
            self.cut.append(I)
        self.N_I = len(self.cut)

    def rectangle_detector(self):
        """
        check whether speech rectangles are in images
        :return: check : check(i) shows number of existence of rectangle in i-th image
                 loc : loc(i) shows (x, y, w, h) of rectangle location in i-th image (if not exist, [])
        """
        print('\nrectangle')
        refresh_folder(opt.rectanglepath)
        self.rectangle_check = np.zeros(shape=[self.N_I])
        self.rectangle_loc = []
        for i in range(self.N_I):
            self.rectangle_loc.append(rectangle_detect(self, self.cut[i], i))
            if len(self.rectangle_loc[i]):
                self.rectangle_check[i] = int(len(self.rectangle_loc[i]))
        if int(self.opt.eval_print):
            rect_output = self.rectangle_check - return_label_matrix_rect(self.opt.assembly_name)
            print(rect_output)
        return self.rectangle_check, self.rectangle_loc

    def circle_detector(self):
        """
        check whether speech circles are in images
        :return: check : check(i) shows number of existance of circle in i-th image
                 loc : loc(i) shows (x, y, w, h, group_index) of circle location in i-th image (if not exist, [])
                 group_loc : group_loc(i) shows (x, y, w, h, group_index) of grouped circles location in i-th image (if not exist [])
        """
        print('\ncircle')
        refresh_folder(opt.circlepath)
        # make circle_check, circle_loc
        self.circle_check = np.zeros(shape=[self.N_I])
        self.circle_loc = []
        self.circle_group_loc = []
        for i in range(self.N_I):
            self.circle_loc.append(circle_detect(self, self.cut[i], i))
            if len(self.circle_loc[i]):
                self.circle_check[i] = int(len(self.circle_loc[i]))
        # make circle_group_loc
        for i in range(self.N_I):
            circle_group_loc_temp = []
            if self.circle_check[i] == 0:
                circle_group_loc_temp = self.circle_loc[i]
            else:
                circle_loc = np.array(self.circle_loc[i])
                circle_loc[:, 2] = circle_loc[:, 0] + circle_loc[:, 2]
                circle_loc[:, 3] = circle_loc[:, 1] + circle_loc[:, 3]
                group_indices = np.unique(circle_loc[:, 4])
                for group_index in group_indices:
                    x_min, x_max, y_min, y_max = 0, 0, 0, 0
                    for n in range(circle_loc.shape[0]):
                        if circle_loc[n, 4] == group_index:
                            if x_min == 0:
                                x_min, y_min, x_max, y_max = circle_loc[n, :4]
                            else:
                                x_min = min([x_min, circle_loc[n, 0]])
                                y_min = min([y_min, circle_loc[n, 1]])
                                x_max = max([x_max, circle_loc[n, 2]])
                                y_max = max([y_max, circle_loc[n, 3]])
                    circle_group_loc_temp.append([x_min, y_min, x_max - x_min, y_max - y_min, group_index])
            self.circle_group_loc.append(circle_group_loc_temp)
        if int(self.opt.eval_print):
            circle_output = self.circle_check - return_label_matrix_circle(self.opt.assembly_name)
            print(circle_output)
        return self.circle_check, self.circle_loc, self.circle_group_loc

    def serial_detector(self):
        """
        check number of serial_numbers in image
        : return: check: check(i) shows number of existance of serial_number in i-th image
        """
        print('\nserial')
        refresh_folder(self.opt.serialpath)
        self.serial_check = np.zeros(shape=[self.N_I])
        self.serial_loc = []
        self.serial_loc, _ = serial_detect(self)
        for i in range(self.N_I):
            if len(self.serial_loc[i]):
                self.serial_check[i] = int(len(self.serial_loc[i]))
        if self.opt.eval_print:
            serial_output = self.serial_check - return_label_matrix_serial(self.opt.assembly_name)
            print(serial_output)
        return self.serial_check, self.serial_loc

    def action_detector(self):
        """
        check whether actions are in images
        : return: check: check(i) shows existance of action in i - th image, numpy array
                 loc: loc(i) shows(c, x, y, w, h) of action location in i - th image(if not exist, []), list
        """
        print('\naction')
        refresh_folder(self.opt.actionpath)
        self.action_check = np.zeros(shape=[self.N_I])
        self.action_loc = []
        for i, cut in enumerate(self.cut):
            temp_action_loc = action_checker(cut, self.opt.actionpath, image_order=i)
            self.action_loc.append(temp_action_loc)
            if sum(self.action_loc[i][0]) > 0:
                self.action_check[i] = 1
        return self.action_check, self.action_loc

    def mult_detector(self):
        """
        check number of serial_numbers in image
        : return: check: check(i) shows number of existance of serial_number in i-th image
        """
        print('\nmult')
        refresh_folder(self.opt.multpath)
        self.mult_check = np.zeros(shape=[self.N_I])
        self.mult_loc = []
        self.mult_loc = mult_detect(self)
        for i in range(self.N_I):
            if len(self.mult_loc[i]):
                self.mult_check[i] = int(len(self.mult_loc[i]))

        if int(self.opt.eval_print):
            mult_output = self.mult_check - return_label_matrix_mult(self.opt.assembly_name)
            print(mult_output)
        return self.mult_check, self.mult_loc

    def OCR(self):
        assembly_name = self.opt.assembly_name
        serialpath = self.opt.serialpath
        multpath = self.opt.multpath
        ocr_modelpath = self.opt.ocr_modelpath
        N_I = self.N_I
        serial_loc = self.serial_loc
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu

        img_pl = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
        logit_tf = digit_recognizer(img_pl, is_train=False, reuse=tf.AUTO_REUSE)

        self.OCR_serial_result, mult_intermidiate_result = OCR_serial(self.sess, img_pl, logit_tf, assembly_name, serialpath, ocr_modelpath, N_I,
                                                                      serial_loc)
        self.OCR_mult_result = OCR_mult(self.sess, img_pl, logit_tf, assembly_name, multpath, ocr_modelpath, self.OCR_serial_result,
                                        mult_intermidiate_result)
        return self.OCR_serial_result, self.OCR_mult_result

    def write_csv_mission(self, option=0):
        self.OCR_serial_index, self.OCR_mult_result_mod = serial_number_to_index('./function/utilities/material_label.csv', self.OCR_serial_result, self.OCR_mult_result)
        if option == 0:
            write_csv_mission1(self.OCR_serial_index, self.OCR_mult_result_mod, self.opt.cutpath, self.opt.csv_dir)
        else:
            write_csv_mission2(self.OCR_serial_index, self.OCR_mult_result_mod, self.opt.cutpath, self.opt.csv_dir)

if __name__ == '__main__':
    tic_main = time.time()

    # define arguments
    opt = init_args()
    s = Stefan(opt)

    s.circle_detector()
    '''
    [circle]
    :return: circle check, loc [x, y, w, h, group_index], group loc [x, y, w, h, group_index]
    '''

    s.mult_detector()
    '''
    [multiply count]
    return mult loc [x, y, w, h, group_index]
    ex) gamleby_00_02_00_3.png
    -> assembly name : gamleby, cut: 00, mult_num: 02, digit: 00, group_index: 3 (if no group, 0)
    '''

    s.rectangle_detector()
    '''
    [rectangle]
    return rectangle check, loc
    '''

    s.serial_detector()
    '''
    save cut digit images and numpy
    ex) gamleby_{cut}_{serial num}_{digit}.png
    :return: serial loc [x(mid point), y(mid point), group_index, angle, width, height] (if no group, 0)
    :return: is_pair_matrices ( serial i, serial j are right next to each other if is_pair_matrices[#cut][i , j] = 1)
    '''

#    s.action_detector()
    '''
    [action]
    return action check, loc
    '''

    save_group_intermediate(s)
    '''
    [save group intermediate results]
    save colored group images of bubble, serial_num, mult_num
    '''

    # [ OCR & combine ]
    s.OCR()

    # mission csv output
    s.write_csv_mission(0)
    s.write_csv_mission(1)

    toc_main = time.time()
    minute, second = divmod(toc_main - tic_main, 60)
    print(f'main.py total : {minute} minutes {second} seconds')
