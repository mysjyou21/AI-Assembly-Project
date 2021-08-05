import sys
sys.path.append('./function/utilities')
from function.utilities.utils import *
import cv2 as cv
import cv2

def serial_detect(self, step_num):
    """
    Leaves only serial numbers in images. Uses connected components.
    Numbers in rectangle bubble region, material list region (of usuallly first step) are ignored
    Numbers are gouped to nearest circle
    : return: 
            self.serial_loc: serial_loc(i) is appended list of[x(mid_point), y(mid_point), group_index, angle, width, height]
                             of serial_numbers(if not exist, [])
            self.is_pair_matrices: is_pair_matrices(i) show pair relation between n number components with nxn
                                   matrix. If is_pair_matrices[i][j, k] == 1, jth number is right above kth number
    """
    """
    SAVE_SERIAL_BLACK : Whether to save intermediate results with black background
    SAVE_SERIAL_WHITE : Whether to save intermediate results with white background
    SAVE_SERIAL_WHOLE : Whether to save whole 6 digit number
    SAVE_SERIAL_INDIV_NUMPY : Whether to save NUMPY file of individual number
    PRINT_PROGRESS : Whether to print progress of this function
    REMOVE_RECTANGLE : Whether to erase rectangle bubble region
    REMOVE_MATERIAL_LIST : Whether to erase rectangle bubble region
    REMOVE_BOTTOM: Whether to erase bottom region (some manuals have codes similar to serial numbers at bottom)
    RESOLUTION_DEPENDENT : Whether function is resolution dependent (effects cc algorithm)
    """
    SAVE_SERIAL = self.opt.save_serial
    SAVE_SERIAL_WHOLE = self.opt.save_serial_whole
    SAVE_SERIAL_BLACK = self.opt.save_serial_black
    SAVE_SERIAL_WHITE = self.opt.save_serial_white
    PRINT_PROGRESS = self.opt.print_serial_progress
    SAVE_SERIAL_INDIV_NUMPY = self.opt.save_serial_npy
    REMOVE_RECTANGLE = True
    REMOVE_NOT_NEAR_CIRCLE = True
    REMOVE_MATERIAL_LIST = True
    REMOVE_BOTTOM = False
    RESOLUTION_DEPENDENT = False

    refresh_folder(self.opt.serial_whole_path)
    refresh_folder(self.opt.serial_path)
    refresh_folder(self.opt.serial_black_path)
    refresh_folder(self.opt.serial_white_path)

    self.circles_loc[step_num] = self.step_circles
    self.rectangles_loc[step_num] = self.step_rectangles

    image_list = [] # legacy
    step_img = self.steps[step_num]
    img = np.copy(step_img)
    if PRINT_PROGRESS:
        print('step num %d : serial detect' % step_num)

    # remove rectangle bubble area (fill white)
    if REMOVE_RECTANGLE:
        if len(self.rectangles_loc[step_num]) > 0:
            for rect_loc in self.rectangles_loc[step_num]:
                x, y, w, h = rect_loc[:4]
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)


    # only look around circle bubble region
    if REMOVE_NOT_NEAR_CIRCLE:
        if len(self.rectangles_loc[step_num]) > 0:
            for rect_loc in self.rectangles_loc[step_num]:
                x, y, w, h = rect_loc[:4]
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    
    # convert image to inversed binary image
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.threshold(img, 140, 255, cv.THRESH_BINARY_INV)[1]

    # remove material list area
    if REMOVE_MATERIAL_LIST:
        horizontal_line = np.ones([1, img.shape[1]], dtype=np.float32) / img.shape[1]
        result = np.matmul(img, horizontal_line.T)
        maximum_avg_intensity = result.max()
        if maximum_avg_intensity > 200: # resolution dependent 한 것 같은데... # yellow
            line_idx = int(np.argmax(result, axis=0))
            img[:line_idx, :] = 0

    


    # find connected components
    ret, labels, stats, c = cv.connectedComponentsWithStats(img, connectivity=8)

    """
    ret : number of components
    labels : number pixel values of each component
    stats : left(x), top(y), w, h, area
    c : center positions of each component
    """
    remain_index = []
    h_, w_ = img.shape[:2]
    CC_NUM_THRESHOLD = 300
    ret_list = np.flip(np.argsort(stats[:,4]), 0)
    if ret > CC_NUM_THRESHOLD:
        ret_list = ret_list[:CC_NUM_THRESHOLD]
    for r in ret_list:
        remain_condition = True # condition whether to keep the component
        cc_x, cc_y, cc_w, cc_h, cc_area = stats[r]

        if RESOLUTION_DEPENDENT:
            if cc_y > 0.94 * h_:
                # at bottom
                if REMOVE_BOTTOM:
                    remain_condition = False
            if cc_area > 6.36945e-5 * h_ * w_:
                # too big
                remain_condition = False
            if cc_area < 4.88326e-6 * h_ * w_:
                # too small
                remain_condition = False
            if cc_w > 2.2695035e-2 * w_ or cc_h > 2.2695035e-2 * w_:
                # too long
                remain_condition = False
            if cc_w < 5.673759e-3 * w_ and cc_h < 5.673759e-3 * w_:
                # too short
                remain_condition = False
        else:
            if cc_y > 0.94 * h_:
                # at bottom
                if REMOVE_BOTTOM:
                    remain_condition = False
            if cc_area > 500:
                # too big
                remain_condition = False
            if cc_area < 23:
                # too small
                remain_condition = False
            if cc_w > 40 or cc_h > 40:
                # too long
                remain_condition = False
            if cc_w < 10 and cc_h < 10:
                # too short
                remain_condition = False

        if remain_condition:
            remain_index.append(r)

    if len(remain_index) > 500:
        # if too much components are detected, abort (might be instruction cut)
        remain_index = []        

    # visualize remaining components
    if SAVE_SERIAL_BLACK:
        img = (np.isin(labels, remain_index) * 255).astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for l in remain_index:
            img = cv.circle(img, (int(c[l, 0]), int(c[l, 1])), 5, (0, 0, 255), -1)  # red # circle component centroids

    # find end points of serial numbers
    end_points = []

    for n, v in enumerate(remain_index):
        for w in remain_index[n + 1:]:
            # vector v - w
            vector_base = c[w] - c[v]
            mag_base, angle_base = cv.cartToPolar(vector_base[0], vector_base[1], angleInDegrees=1)
            mag_base, angle_base = mag_base[0], angle_base[0]
            if mag_base > img.shape[0] / 4:
                # if too long, continue
                continue
            # vectors v - all
            vectors = c[remain_index] - c[v]
            mags, angles = cv.cartToPolar(vectors[:, 0], vectors[:, 1], angleInDegrees=1)
            angles_abs = np.absolute(angles - angle_base)
            angles_abs = np.absolute((angles_abs + 180) % 360 - 180)
            component_index = []
            for z, mag in enumerate(mags):
                # distance check
                if mag > 0 and mag < mag_base:
                    # angle check
                    if angles_abs[z] < 15:
                        component_index.append(z)
            if len(component_index) == 4:
                component_mags = mags[component_index]
                a = np.array(0.0)
                a = np.reshape(a, (-1, 1))
                component_mags = np.concatenate((component_mags, a))
                b = np.array(mag_base)
                b = np.reshape(b, (-1, 1))
                component_mags = np.concatenate((component_mags, b))
                component_angles = angles[component_index]
                component_angles = (component_angles + 180) % 360 - 180
                angle_base = (angle_base + 180) % 360 - 180
                angle_avg = (angle_base + np.sum(component_angles)) / 5
                component_mags = np.sort(np.squeeze(component_mags))
                component_mags_diff = np.diff(component_mags)
                diff_max = np.max(component_mags_diff)
                diff_min = np.min(component_mags_diff)
                # even spaced check
                if diff_max / diff_min < 1.5:
                    mag_angle_info = np.zeros((2))
                    mag_angle_info[0] = mag_base
                    mag_angle_info[1] = -angle_base  # since opencv y-axis head down, invert it
                    overlap_flag = np.zeros((2))
                    components = [remain_index[x] for x in component_index]
                    components.append(v)
                    components.append(w)
                    x = stats[components, 0]
                    y = stats[components, 1]
                    w_ = stats[components, 2]
                    h = stats[components, 3]
                    w_min = np.min(w_)
                    h_min = np.min(h)
                    im_sum = 0
                    im_err_sum = 0
                    for iter in range(6):
                        im_cur = img[y[iter]:y[iter] + h_min, x[iter]:x[iter] + w_min]
                        im_sum += np.sum(im_cur >= 0)
                    for iter in range(5):
                        im_cur = img[y[iter]:y[iter] + h_min, x[iter]:x[iter] + w_min]
                        im_next = img[y[iter + 1]:y[iter + 1] + h_min, x[iter + 1]:x[iter + 1] + w_min]
                        im_err = np.sum(np.absolute(im_next - im_cur))
                        im_err_sum += im_err
                    # print(im_err_sum / im_sum)
                    # show(img[y[iter]:y[iter] + h_min, x[iter]:x[iter] + w_min])
                    # if im_err_sum / im_sum > 10: # yellow 이거 썼을때 별로였나? ...
                    end_points.append([c[v], c[w], mag_angle_info, overlap_flag])

    # choose (start, end) of each endpoint pair
    # x[0] becomes start point
    for x in end_points:
        swap = False
        if x[0][0] < x[1][0]:
            x_start, x_end = x[0], x[1]
        else:
            x_start, x_end = x[1], x[0]
            swap = True
        vector_x = x[0] - x[1]
        if vector_x[1] < 0:
            vector_x = -1 * vector_x
        mag_x, angle_x = cv.cartToPolar(vector_x[0], vector_x[1], angleInDegrees=1)
        mag_x, angle_x = mag_x[0], angle_x[0]
        if angle_x > 80 and angle_x < 100:
            if x[0][1] < x[1][1]:
                x_start, x_end = x[1], x[0]
                swap = True
            else:
                x_start, x_end = x[0], x[1]
        x[0], x[1] = x_start, x_end
        if swap == True:
            x[2][1] = x[2][1] + 180


    # manage serial numbers that are longer than 6 digits (concatenate overlapping serial number regions)
    end_points = np.array(end_points)
    serial_images_dict = {}
    for j, x in enumerate(end_points):
        serial_images_dict[j] = []
        mid_point, angle_deg, width, height = get_serial_detect_values(x)
        ul, ur, lr, ll = get_serial_detect_corners(mid_point, angle_deg, width, height)
        corner0 = np.reshape(np.array((ul[0], ul[1])).astype(np.int32), (1, -1))
        corner1 = np.reshape(np.array((ur[0], ur[1])).astype(np.int32), (1, -1))
        corner2 = np.reshape(np.array((lr[0], lr[1])).astype(np.int32), (1, -1))
        corner3 = np.reshape(np.array((ll[0], ll[1])).astype(np.int32), (1, -1))
        contour = np.reshape(np.concatenate((corner0, corner1, corner2, corner3), axis=0), (4, 1, 2))
        for k, w in enumerate(end_points):
            if j == k:
                continue
            start_inside = cv.pointPolygonTest(contour, tuple(w[0].astype(np.int32)), False)
            end_inside = cv.pointPolygonTest(contour, tuple(w[1].astype(np.int32)), False)
            if (start_inside == 1.0) and (end_inside == 1.0):
                w[0] = x[0]
                w[1] = x[1]
                x[3][0] += 0.5
                w[3][0] += 0.5
            if (start_inside == 1.0) and (end_inside != 1.0):
                w[0] = x[0]
                x[1] = w[1]
                x[3][0] += 0.5
                w[3][0] += 0.5
            if (start_inside != 1.0) and (end_inside == 1.0):
                w[1] = x[1]
                x[0] = w[0]
                x[3][0] += 0.5
                w[3][0] += 0.5
    if end_points.shape[0] != 0:
        if np.sum(end_points[:, 3, 0]) != 0:
            end_points = end_points[np.unique(end_points[:, 0, 1], return_index=True)[1]]


    # crop serial number regions from image
    im = Image.fromarray(step_img)
    im_white = np.copy(im)
    delete_end_points_index = []
    print_flags = False # for debugging purpose
    if print_flags:
        print('step', step_num)
    for j, x in enumerate(end_points):
        mid_point, angle_deg, width, height = get_serial_detect_values(x)
        im_cropped = crop(im, mid_point, angle_deg, width, height)
        im_cropped = np.asarray(im_cropped)
        ratio = 32 / im_cropped.shape[0]
        im_cropped = cv.resize(im_cropped, None, fx=ratio, fy=ratio, interpolation=cv.INTER_CUBIC)
        # cropped region too long
        if im_cropped.shape[1] > 160:
            delete_end_points_index.append(j)
            if print_flags:
                print('f1')
            continue
        zero_padding = np.full((32, 160 - im_cropped.shape[1], 3), 255.0)
        im_cropped = np.concatenate((im_cropped, zero_padding), axis=1).astype(np.uint8)

        # split to individual digits
        is_screw = False
        im_cropped_gray = cv.cvtColor(im_cropped, cv.COLOR_BGR2GRAY)
        vertical_sum = np.sum(im_cropped_gray, axis=0)
        vert_max = vertical_sum.max()
        vert_min = vertical_sum.min()
        cut_index = []
        for idx, vert in enumerate(vertical_sum):
            if vert > 0.88 * vert_max:
                cut_index.append(idx)
        cut_index_temp = []
        for idx in range(len(cut_index)):
            try:
                condition1 = bool(cut_index[idx] != cut_index[idx - 1] + 1)
                condition2 = bool(cut_index[idx] != cut_index[idx + 1] - 1)
                if condition1 or condition2:
                    cut_index_temp.append(cut_index[idx])
                    if condition1 and condition2:
                        cut_index_temp.append(cut_index[idx])
            except:
                pass
        cut_index = cut_index_temp[2:]
        cut_index_temp = [0]
        for idx in range(len(cut_index) // 2):
            mid_value = int(round((cut_index[2 * idx] + cut_index[2 * idx + 1]) / 2) + 1)
            cut_index_temp.append(mid_value)
        try:
            cut_index_temp.append(cut_index[-1])
        except:
            # error : screw FP
            is_screw = True  
            delete_end_points_index.append(j)
            if print_flags:
                print('f2')
            continue
        cut_index = cut_index_temp

        if False: # for debugging
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            x_axis = list(range(len(vertical_sum)))
            ax.bar(x_axis, vertical_sum)
            plt.show()

        im_cropped_binary = cv.threshold(im_cropped_gray, 140, 255, cv.THRESH_BINARY_INV)[1]
        ret, _, _, c = cv.connectedComponentsWithStats(im_cropped_binary, connectivity=8)
        delete_index = []
        for r in range(ret):
            if c[r, 1] < 8 or c[r, 1] > 24:
                delete_index.append(r)
        c = np.delete(c, delete_index, 0)
        ret = c.shape[0]

        # 6 digits but len(cut_index) > 7 case handling
        if len(cut_index) > 7:
            if cut_index[-1] < 135:
                if print_flags:
                    print('cut_index cropped')
                delete_index = np.argmin(np.diff(np.array(cut_index)))
                del cut_index[delete_index]

        # error : more than 10 digits
        if ret >= 10:
            is_screw = True
            delete_end_points_index.append(j)
            if print_flags:
                print('f3')
                show(im_cropped)
            continue
        
        # error : less than 6 digits
        if len(cut_index) < 7:
            if ret != 7:
                if print_flags:
                    print('f4')
                is_screw = True
                delete_end_points_index.append(j)
                continue
            else:
                # handle '74'
                diff = np.diff(np.array(cut_index))
                diff_index = np.argmax(diff)
                add_value = diff[diff_index] // 2
                cut_index.insert(diff_index + 1, cut_index[diff_index] + add_value)

        # error : longer than 32 bits
        diff = np.diff(np.array(cut_index))
        if np.sum(diff > 32) >= 1:
            if print_flags:
                print('f5')
            is_screw = True
            delete_end_points_index.append(j)
            continue

        # error : vertical sum of pixels minimum value too high
        if vert_min > 5000:
            if print_flags:
                print('f6')
            is_screw = True
            delete_end_points_index.append(j)
            continue

        # save individual digits
        for k in range(len(cut_index) - 1):
            im_serial_indiv = im_cropped[:, cut_index[k]:cut_index[k + 1], :]
            indiv = cv.cvtColor(im_serial_indiv, cv.COLOR_RGB2GRAY)
            indiv = 255 - indiv
            x_start = (32 - indiv.shape[1]) // 2
            zero_padding = np.zeros((32, 32)).astype(np.uint8)
            try:
                zero_padding[:, x_start:x_start + indiv.shape[1]] = indiv
                img_indiv = zero_padding
                image_list.append(img_indiv) #black background
                serial_images_dict[j].append(img_indiv)
                if SAVE_SERIAL:
                    cv.imwrite(os.path.join(self.opt.serial_path, self.opt.assembly_name + '_' + str(step_num).zfill(2) + '_' + str(j).zfill(2) + '_' + str(k).zfill(2) + '.png'), img_indiv)
            except:
                pass

        if SAVE_SERIAL_WHITE and is_screw == False:
            im_white = drawBoundingBox(im_white, mid_point, angle_deg, width, height)

        # save whole 6 digits
        if SAVE_SERIAL_WHOLE and is_screw == False:
            cv.imwrite(os.path.join(self.opt.serial_whole_path, self.opt.assembly_name + '_' + str(step_num).zfill(2) + '_' + str(j).zfill(2) + '.png'), im_cropped)

    # delete False Positives
    delete_end_points_index = np.unique(np.array(delete_end_points_index))
    end_points = np.delete(end_points, delete_end_points_index, 0)

    # save serial black
    if SAVE_SERIAL_BLACK:
        for x in end_points:
            img1 = cv.circle(img, tuple(x[0].astype(np.int)), 10, (0, 255, 0), -1)  # green # start point
            img1 = cv.circle(img, tuple(x[1].astype(np.int)), 10, (255, 0, 0), -1)  # blue # end point
            pass
        cv.imwrite(os.path.join(self.opt.serial_black_path, self.opt.assembly_name + '_' + str(step_num).zfill(2) + '.png'), img)

    # SAVE serial white
    if SAVE_SERIAL_WHITE:
        cv.imwrite(os.path.join(self.opt.serial_white_path, self.opt.assembly_name + '_' + str(step_num).zfill(2) + '.png'), im_white)  # red # serial region


    # save individal serial digits as numpy
    if SAVE_SERIAL_INDIV_NUMPY:
        image_npy = np.array(image_list)
        image_npy = image_npy.reshape(-1, 32, 32, 1)
        print('\nnpy shape : {}'.format(image_npy.shape))
        np.save(os.path.join(self.opt.serial_path, self.opt.assembly_name + '_serial.npy'), image_npy)

    # return location
    step_serial_loc = []
    N = len(end_points)
    # end_points : each element(4x2) : [[start(x), start(y)], [end(x), end(y)], [mag_base, angle_base], [overlap_flag, Nothing(0)]]
    if N != 0:
        for x in end_points:
            mid_point, angle_deg, width, height = get_serial_detect_values(x)
            mid_point = mid_point.astype(np.int)
            mid_point = np.append(mid_point, [width, height, angle_deg]).tolist()
            step_serial_loc.append(mid_point)
    else:
        step_serial_loc = []

    # return images
    step_serial_images = []
    for serial_idx in list(serial_images_dict):
        step_serial_images.append(serial_images_dict[serial_idx]) # serial_idx is not digit_idx

    """
    step_serial_images: [[img_serial0_0, ..., img_serial0_5],[img_serial1_0, ..., img_serial1_5], ...]
    step_serial_loc : [[loc_serial0], [loc_serial1], ...]  # loc_serial = midpoint(x), midpoint(y), w, h, angle_deg
    """
    return step_serial_images, step_serial_loc



def mult_detect(self, step_num):
    """
    :return: loc : (x, y, w, h, group_index) list of location of mult_num
    """
    SAVE_MULT = self.opt.save_mult
    SAVE_MULT_NPY = self.opt.save_mult_npy
    SAVE_MULT_BLACK = True #self.opt.save_mult_black
    # refresh_folder(self.opt.mult_path)
    # refresh_folder(self.opt.mult_black_path)
    image_list = [] # legacy
    step_mult_loc = []
    # bring template ('X shape')
    template = plt.imread('./function/utilities/template0.png')
    template = np.around(template * 255).astype(np.uint8)
    template_cut = template[8:32, 8:30]
    w, h = template.shape
    # step img
    step_img = self.steps[step_num]
    img = np.copy(step_img)
    h_, w_ = img.shape[:2]
    # connected components of cut
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img_gray = cv.threshold(img_gray, 140, 255, cv.THRESH_BINARY_INV)[1]
    ret, labels, stats, c = cv.connectedComponentsWithStats(img_gray, connectivity=8)
    remain_index = []
    remain_index_1 = []
    delete_index = [0]
    # remove unprobable components
    for r in range(1, ret):
        remain_condition = True
        remain_condition_1 = True
        x, y, w, h, area = stats[r]
        if h > 100 or w > 100:
            # too long
            remain_condition = False
            remain_condition_1 = False
            delete_index.append(r)
        if h < 20 and w < 20:
            # too short
            remain_condition = False
            remain_condition_1 = False
            delete_index.append(r)
        if h / w > 1.2 or h / w < 0.8:
            # ratio not proper
            remain_condition = False
        if remain_condition:
            remain_index.append(r)  # no number remain
        if remain_condition_1:
            remain_index_1.append(r)  # number remain
    step_mult_loc = []
    img_cc = (np.isin(labels, remain_index) * 255).astype(np.uint8)
    img_cc_with_num = (np.isin(labels, remain_index_1) * 255).astype(np.uint8)
    img_cc_with_num = cv.cvtColor(img_cc_with_num, cv.COLOR_GRAY2RGB)

    stats = np.array(stats)
    stats = np.delete(stats, delete_index, 0)
    c = np.array(c)
    c = np.delete(c, delete_index, 0)
    ret = c.shape[0]
    if SAVE_MULT_BLACK:
        cv.imwrite(os.path.join(self.opt.mult_black_path, self.opt.assembly_name + '_' + str(step_num).zfill(2) + '_00.png'), img_cc_with_num)
    for index in range(ret):
        cv.circle(img_cc_with_num, tuple(c[index].astype(np.int)), 10, (255, 0, 0), 3)  # blue # centroids of all components

    methods = ['cv.TM_CCORR']
    for meth in methods:
        img1 = img_cc.copy()
        method = eval(meth)
        res = cv.matchTemplate(img1, template, method)
        threshold = 5000000
        res = ((res > threshold) * 255).astype(np.uint8)
        if SAVE_MULT_BLACK:
            cv.imwrite(os.path.join(self.opt.mult_black_path, self.opt.assembly_name + '_' + str(step_num).zfill(2) + '_01.png'), res)
        # connected components of template matching map
        ret1, labels1, stats1, c1 = cv.connectedComponentsWithStats(res, connectivity=8)
        match_index = []
        mult_num_order = 0
        already_chosen_index = []
        taken = []
        mult_images_dict = {}
        for r1 in range(1, ret1):
            mult_images_dict[mult_num_order] = []
            points = stats[:, :2]
            x = stats1[r1, 0] + stats1[r1, 2]
            y = stats1[r1, 1] + stats1[r1, 3]
            point1 = np.array([x, y])
            # distance between cc points of template matching map & img_cc_with_num
            diff = points - point1
            diff = np.sum(np.absolute(diff), axis=1)
            index = np.argmin(diff)
            if diff[index] < 40:
                cv.circle(img_cc_with_num, tuple(c[index].astype(np.int)), 7, (0, 0, 255), -1)  # red # centroid of nearest component to template matching result 
                if SAVE_MULT_BLACK:
                    cv.imwrite(os.path.join(self.opt.mult_black_path, self.opt.assembly_name + '_' + str(step_num).zfill(2) + '_02.png'), img_cc_with_num)
                x, y, w, h = stats[index, :4]
                bottom_left_point = (x, y + h)  # bottom_left point of X
                bottom_right_point = (x + w, y + h)
                up_right_point = (x + w, y)

                # check similarity of x_cut with template_cut
                x_cut = ((labels[y:y + h, x:x + w] > 0) * 255).astype(np.uint8)
                fx = template_cut.shape[1] / x_cut.shape[1]
                fy = template_cut.shape[0] / x_cut.shape[0]
                x_cut = cv.resize(x_cut, None, fx=fx, fy=fy, interpolation=cv.INTER_NEAREST)
                sim = template_cut * x_cut
                if np.sum(sim) < 100:
                    continue

                # check if one / two components are in approx region of mult_num
                rect_w = 2 * w
                rect_h = 1.1 * h
                cv.rectangle(img_cc_with_num, (int(bottom_left_point[0] - rect_w), int(bottom_left_point[1] - rect_h)), (int(bottom_left_point[0]), int(bottom_left_point[1])), (0, 255, 0), 5)  # green
                inside_rect_index = []
                for r in range(ret):
                    x_q, y_q = c[r, :]
                    if x_q > bottom_left_point[0] - rect_w and x_q < bottom_left_point[0]:
                        if y_q > bottom_left_point[1] - rect_h and y_q < bottom_left_point[1]:
                            if r not in already_chosen_index:
                                inside_rect_index.append(r)
                                already_chosen_index.append(r)
                points = []
                for r in inside_rect_index:
                    x, y, w, h = stats[r, :4]
                    points.append([x, y])
                    points.append([x + w, y])
                    points.append([x, y + h])
                    points.append([x + w, y + h])
                points = np.array(points)
                try:
                    x_min = int(points[:, 0].min())
                    x_max = int(points[:, 0].max())
                    y_min = int(points[:, 1].min())
                    y_max = int(points[:, 1].max())
                    cv.rectangle(img_cc_with_num, (x_min, y_min), (x_max, y_max), (255, 0, 255), 5)  # magenta
                    if SAVE_MULT_BLACK:
                        cv.imwrite(os.path.join(self.opt.mult_black_path, self.opt.assembly_name + '_' + str(step_num).zfill(2) + '_03.png'), img_cc_with_num)
                    if points.shape[0] == 8:
                        x_sort = np.sort(points[:, 0])
                        x_mid = int((x_sort[3] + x_sort[4]) / 2)

                        cropped_region_0 = img[y_min:y_max, x_min:x_mid]
                        ratio_0 = 24 / cropped_region_0.shape[0]
                        resized_0 = cv.resize(cropped_region_0, None, fx=ratio_0, fy=ratio_0, interpolation=cv.INTER_CUBIC)
                        left_pad_0 = int((32 - resized_0.shape[1]) / 2)
                        right_pad_0 = 32 - resized_0.shape[1] - left_pad_0
                        padded_0 = np.pad(resized_0, ((4, 4), (left_pad_0, right_pad_0), (0, 0)), 'constant', constant_values=255)
                        if SAVE_MULT:
                            cv.imwrite(os.path.join(self.opt.mult_path, self.opt.assembly_name + '_' + str(step_num).zfill(2) + '_' + str(mult_num_order).zfill(2) + '_'
                                                + '00.png'), padded_0)
                        padded_0 = cv.cvtColor(padded_0, cv.COLOR_BGR2GRAY)
                        padded_0 = 255 - padded_0
                        # if step_num == 5:
                        #     print(mult_num_order)
                        #     show(padded_0, 'padded_0')
                        image_list.append(padded_0)
                        mult_images_dict[mult_num_order].append(padded_0)

                        cropped_region_1 = img[y_min:y_max, x_mid:x_max]
                        ratio_1 = 24 / cropped_region_1.shape[0]
                        resized_1 = cv.resize(cropped_region_1, None, fx=ratio_1, fy=ratio_1, interpolation=cv.INTER_CUBIC)
                        left_pad_1 = int((32 - resized_1.shape[1]) / 2)
                        right_pad_1 = 32 - resized_1.shape[1] - left_pad_1
                        padded_1 = np.pad(resized_1, ((4, 4), (left_pad_1, right_pad_1), (0, 0)), 'constant', constant_values=255)
                        if SAVE_MULT:
                            cv.imwrite(os.path.join(self.opt.mult_path, self.opt.assembly_name + '_' + str(step_num).zfill(2) + '_' + str(mult_num_order).zfill(2) + '_'
                                                + '01.png'), padded_1)
                        padded_1 = cv.cvtColor(padded_1, cv.COLOR_BGR2GRAY)
                        padded_1 = 255 - padded_1
                        # if step_num == 5:
                        #     print(mult_num_order)
                        #     show(padded_1, 'padded_1')
                        image_list.append(padded_1)
                        mult_images_dict[mult_num_order].append(padded_1)
                        mult_num_order += 1
                    else:
                        cropped_region = img[y_min:y_max, x_min:x_max]
                        ratio = 24 / cropped_region.shape[0]
                        resized = cv.resize(cropped_region, None, fx=ratio, fy=ratio, interpolation=cv.INTER_CUBIC)
                        left_pad = int((32 - resized.shape[1]) / 2)
                        right_pad = 32 - resized.shape[1] - left_pad
                        padded = np.pad(resized, ((4, 4), (left_pad, right_pad), (0, 0)), 'constant', constant_values=255)
                        if SAVE_MULT:
                            cv.imwrite(os.path.join(self.opt.mult_path, self.opt.assembly_name + '_' + str(step_num).zfill(2) + '_' + str(mult_num_order).zfill(2) + '_'
                                                + '00.png'), padded)
                        padded = cv.cvtColor(padded, cv.COLOR_BGR2GRAY)
                        padded = 255 - padded
                        # if step_num == 5:
                        #     print(mult_num_order)
                        #     show(padded, 'padded')
                        image_list.append(padded)
                        mult_images_dict[mult_num_order].append(padded)
                        mult_num_order += 1
                    # x, y, w, h = stats[index, :4]
                    x, y, w, h = x_min, y_min, x_max-x_min, y_max-y_min
                    step_mult_loc.append([int(x), int(y), int(w), int(h)])
                except:
                    pass


    if SAVE_MULT_NPY:
        image_npy = np.array(image_list) # black background
        image_npy = image_npy.reshape(-1, 32, 32, 1)
        print('npy shape : {}'.format(image_npy.shape))
        np.save(os.path.join(self.opt.mult_path, self.opt.assembly_name + '_mult.npy'), image_npy)

    # return images
    step_mult_images = []
    for mult_idx in list(mult_images_dict):
        if len(mult_images_dict[mult_idx]) != 0:
            step_mult_images.append(mult_images_dict[mult_idx])

    """
    step_mult_images: [[img_mult0_0, ...],[img_mult1_0, ...], ...]
    step_mult_loc : [[loc_mult0], [loc_mult1], ...]  # loc_mult = x, y, w, h (of 'X' sign)
    """
    return step_mult_images, step_mult_loc
