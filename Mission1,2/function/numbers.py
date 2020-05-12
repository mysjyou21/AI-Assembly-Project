import sys
sys.path.append('./function/utilities')
from function.utilities.utils import *


def serial_detect(self):
    """
    aiming to leave only serial_numbers in images
    uses connected components
    numbers in rectangle region, material instruction region are ignored
    numbers are gouped to nearest circle
    : return: self.serial_loc: serial_loc(i) is appended list of[x(mid_point), y(mid_point), group_index, angle, widht, height]
                     of serial_numbers(if not exist, [])
             self.is_pair_matrices: is_pair_matrices(i) show pair relation between n number components with nxn
             matrix. If is_pair_matrices[i][j, k] == 1, jth number is right above kth number
    """
    '''
    SAVE_INTERMEDIATE_BLACK : Whether to save intermediate results with black background
    SAVE_INTERMEDIATE_WHITE : Whether to save intermediate results with white background
    SAVE_SERIAL_WHOLE : Whether to save whole 6 digit number
    SAVE_SERIAL_INDIV_NUMPY : Whether to save NUMPY file of individual number
    PRINT_PROGRESS : Whether to print progress of this function
    '''
    SAVE_SERIAL_WHOLE = self.opt.save_serial_whole
    SAVE_INTERMEDIATE_BLACK = self.opt.save_serial_black
    SAVE_INTERMEDIATE_WHITE = self.opt.save_serial_white
    PRINT_PROGRESS = self.opt.print_serial_progress
    SAVE_SERIAL_INDIV_NUMPY = True
    RECTANGLE = True  # True

    refresh_folder(self.opt.serial_whole_path)
    refresh_folder(self.opt.serialpath)
    refresh_folder(self.opt.serial_black_path)
    refresh_folder(self.opt.serial_white_path)

    image_list = []
    self.is_pair_matrices = []
    total_serial = len(self.cut)
    for i, cut in enumerate(self.cut):
        img = np.copy(cut)

        # remove rectangle area
        if RECTANGLE:
            if self.rectangle_check[i] > 0:
                for rect_loc in self.rectangle_loc[i]:
                    x, y, w, h = rect_loc
                    cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = cv.threshold(img, 140, 255, cv.THRESH_BINARY_INV)[1]

        # remove material instruction area
        horizontal_line = np.ones([1, img.shape[1]], dtype=np.float32) / img.shape[1]
        result = np.matmul(img, horizontal_line.T)
        maximum_avg_intensity = result.max()
        if maximum_avg_intensity > 200:
            line_idx = int(np.argmax(result, axis=0))
            img[:line_idx, :] = 0

        # show(img)

        # CONNECTED COMPONENTS
        ret, labels, stats, c = cv.connectedComponentsWithStats(img, connectivity=8)

        '''
        ret : number of components
        labels : number pixel values of each component
        stats : left(x), top(y), w, h, area
        c : center positions of each component
        '''
        remain_index = []
        h_, w_ = img.shape[:2]
        for r in range(1, ret):

            # CONDITION WHETHER TO KEEP COMPONENT FROM IMAGE
            remain_condition = True
            cc_x, cc_y, cc_w, cc_h, cc_area = stats[r]

            # RESOLUTION DEPENDENT
            # if cc_y > 0.94 * h_:
            #     # at bottom
            #     remain_condition = False
            # if cc_area > 6.36945e-5 * h_ * w_:  # yellow
            #     # too big
            #     remain_condition = False
            # if cc_area < 4.88326e-6 * h_ * w_:  # yellow
            #     # too small
            #     remain_condition = False
            # if cc_w > 2.2695035e-2 * w_ or cc_h > 2.2695035e-2 * w_:  # yellow
            #     # too long
            #     remain_condition = False
            # if cc_w < 5.673759e-3 * w_ and cc_h < 5.673759e-3 * w_:  # yellow
            #     # too short
            #     remain_condition = False

            # if self.opt.assembly_name not in ['input', 'mission1', 'mission2']:
            #     if cc_y > 0.94 * h_:
            #         # at bottom
            #         remain_condition = False

            if cc_area > 500:  # yellow
                # too big
                # (if not removed, occasionally centroid of large objects are place on serial number spot)
                remain_condition = False
            if cc_area < 23:  # yellow
                # too small
                remain_condition = False
            if cc_w > 40 or cc_h > 40 * w_:  # yellow
                # too long
                remain_condition = False
            if cc_w < 10 and cc_h < 10:  # yellow
                # too short
                remain_condition = False

            # LEAVE VALID COMPONENT FROM IMAGE
            if remain_condition:
                remain_index.append(r)

        if len(remain_index) > 500:
            # if too much, pass (might be instruction cut)
            remain_index = []

        if SAVE_INTERMEDIATE_BLACK or SAVE_INTERMEDIATE_WHITE:
            img = (np.isin(labels, remain_index) * 255).astype(np.uint8)
            # show(img, False)

        # VISUALIZE REMAINING COMPONENTS
        if SAVE_INTERMEDIATE_BLACK:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            for l in remain_index:
                img = cv.circle(img, (int(c[l, 0]), int(c[l, 1])), 5, (0, 0, 255), -1)  # red
                pass

        # FIND END POINTS OF SERIAL NUMBER
        end_points = []

        for n, v in enumerate(remain_index):
            for w in remain_index[n + 1:]:
                # vector v - w
                vector_base = c[w] - c[v]
                mag_base, angle_base = cv.cartToPolar(vector_base[0], vector_base[1], angleInDegrees=1)
                mag_base, angle_base = mag_base[0], angle_base[0]
                if mag_base > img.shape[0] / 4:
                    # if too long, pass
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
                    # even spaced check
                    # if np.sum(component_mags < (mag_base / 2)) == 3:
                    if True:
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
                            # if len(np.unique(stats[components, 4])) > 2:
                            if True:
                                # not same pixels
                                components_are_different = False
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
                                # if im_err_sum / im_sum > 10:
                                if True:
                                    # not same shape
                                    end_points.append([c[v], c[w], mag_angle_info, overlap_flag])

        # CHOOSE START POINT, END POINT
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

        # CROP SERIAL NUMBER REGIONS
        im = Image.fromarray(cut)
        imim = np.copy(im)
        end_points = np.array(end_points)

        # MANAGE MORE THAN 6 DIGITS CASE

        for j, x in enumerate(end_points):
            mid_point, angle_deg, width, height = get_serial_detect_values(x)
            ul, ur, lr, ll = get_serial_detect_corners(mid_point, angle_deg, width, height)
            # cv.circle(imim, (int(ul[0]), int(ul[1])), 4, (0, 0, 255), -1)
            # cv.circle(imim, (int(ur[0]), int(ur[1])), 4, (0, 0, 255), -1)
            # cv.circle(imim, (int(lr[0]), int(lr[1])), 4, (0, 0, 255), -1)
            # cv.circle(imim, (int(ll[0]), int(ll[1])), 4, (0, 0, 255), -1)
            # show(imim)
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

        # print(i, len(end_points))
        if PRINT_PROGRESS:
            # print(f'progress : {str(i + 1).zfill(3)}/{total_serial},  # of left cc : {str(ret).zfill(3)}, # of end points : {str(len(end_points)).zfill(3)}', end='\r')
            print(f'progress : {str(i + 1).zfill(3)}/{total_serial}', end='\r')
        # CROP SERIAL NUMBER REGION
        delete_end_points_index = []
        print_flags = False
        if print_flags:
            print('image', i)
        for j, x in enumerate(end_points):
            mid_point, angle_deg, width, height = get_serial_detect_values(x)
            im_cropped = crop(im, mid_point, angle_deg, width, height)
            im_cropped = np.asarray(im_cropped)
            ratio = 32 / im_cropped.shape[0]
            im_cropped = cv.resize(im_cropped, None, fx=ratio, fy=ratio, interpolation=cv.INTER_CUBIC)
            if im_cropped.shape[1] > 160:
                delete_end_points_index.append(j)  # cyan
                if print_flags:
                    print('f1')
                continue
            zero_padding = np.full((32, 160 - im_cropped.shape[1], 3), 255.0)
            im_cropped = np.concatenate((im_cropped, zero_padding), axis=1).astype(np.uint8)

            # SAVE SERIAL INDIV
            is_screw = False
            if True:
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
                    is_screw = True  # screw FP handling
                    delete_end_points_index.append(j)  # cyan
                    if print_flags:
                        print('f2')
                    continue
                cut_index = cut_index_temp

                if False:
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

                # more than 10 digits case
                if ret >= 10:
                    is_screw = True
                    delete_end_points_index.append(j)  # cyan
                    if print_flags:
                        print('f3')
                        show(im_cropped)
                    continue

                # 6 digit but len(cut_index) > 7 error handling
                if len(cut_index) > 7:
                    if cut_index[-1] < 135:
                        if print_flags:
                            print('cut_index cropped')
                        delete_index = np.argmin(np.diff(np.array(cut_index)))
                        del cut_index[delete_index]

                # less than 6 digits case
                if len(cut_index) < 7:
                    if ret != 7:
                        if print_flags:
                            print('f4')  # cyan
                            # show(im_cropped)
                        is_screw = True
                        delete_end_points_index.append(j)
                        continue
                    else:
                        # handle '74'
                        diff = np.diff(np.array(cut_index))
                        diff_index = np.argmax(diff)
                        add_value = diff[diff_index] // 2
                        cut_index.insert(diff_index + 1, cut_index[diff_index] + add_value)

                # longer than 32 bits case
                diff = np.diff(np.array(cut_index))
                if np.sum(diff > 32) >= 1:
                    if print_flags:
                        print('f5')  # cyan
                        # show(im_cropped)
                    is_screw = True
                    delete_end_points_index.append(j)
                    continue

                # vertical sum of pixels min value too high
                if vert_min > 5000:
                    if print_flags:
                        print('f6')
                    is_screw = True
                    delete_end_points_index.append(j)
                    continue

                some_bool = False
                for k in range(len(cut_index) - 1):
                    im_serial_indiv = im_cropped[:, cut_index[k]:cut_index[k + 1], :]
                    if True:
                        cv.imwrite(os.path.join(self.opt.serialpath, self.opt.assembly_name + '_' + str(i).zfill(2) + '_' + str(j).zfill(2) + '_' + str(k).zfill(2) + '.png'), im_serial_indiv)
                    indiv = cv.cvtColor(im_serial_indiv, cv.COLOR_RGB2GRAY)
                    indiv = 255 - indiv
                    x_start = (32 - indiv.shape[1]) // 2
                    zero_padding = np.zeros((32, 32)).astype(np.uint8)
                    # print('x_start', x_start)
                    # print('indiv.shape[1]', indiv.shape[1])
                    try:
                        zero_padding[:, x_start:x_start + indiv.shape[1]] = indiv
                        img_indiv = zero_padding
                        image_list.append(img_indiv)
                        some_bool = True
                    except:
                        pass
                if some_bool:
                    # print(i, j, vert_min)
                    pass

            # for save intermediate white
            if SAVE_INTERMEDIATE_WHITE and is_screw == False:
                imim = drawBoundingBox(imim, mid_point, angle_deg, width, height)

            # SAVE SERIAL WHOLE
            if SAVE_SERIAL_WHOLE and is_screw == False:
                cv.imwrite(os.path.join(self.opt.serial_whole_path, self.opt.assembly_name + '_' + str(i).zfill(2) + '_' + str(j).zfill(2) + '.png'), im_cropped)

        # delete False Positives
        delete_end_points_index = np.unique(np.array(delete_end_points_index))
        end_points = np.delete(end_points, delete_end_points_index, 0)
        # SAVE INTERMEDIATE BLACK
        if SAVE_INTERMEDIATE_BLACK:
            for x in end_points:
                img1 = cv.circle(img, tuple(x[0].astype(np.int)), 10, (0, 255, 0), -1)  # green
                img1 = cv.circle(img, tuple(x[1].astype(np.int)), 10, (255, 0, 0), -1)  # blue
                pass
            cv.imwrite(os.path.join(self.opt.serial_black_path, self.opt.assembly_name + '_' + str(i).zfill(2) + '.png'), img)

        # SAVE INTERMEDIATE WHITE
        if SAVE_INTERMEDIATE_WHITE:
            cv.imwrite(os.path.join(self.opt.serial_white_path, self.opt.assembly_name + '_' + str(i).zfill(2) + '.png'), imim)  # red

        # return pair matrix
        N = len(end_points)
        if N != 0:
            is_pair_matrix = np.zeros((N, N))
            mid_points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2
            for e in range(N):
                for r in range(e, N):
                    is_pair_matrix[e][r] = np.linalg.norm(mid_points[e] - mid_points[r])
            is_pair_matrix_zeros = (is_pair_matrix <= 0) * 100
            is_pair_matrix += is_pair_matrix_zeros
            is_pair_matrix = (is_pair_matrix < 50) * 1
        else:
            is_pair_matrix = [[0]]

        self.is_pair_matrices.append(is_pair_matrix)

        # return location
        serial_loc_temp = []
        N = len(end_points)
        if N != 0:
            circle_loc = np.array(self.circle_loc[i])
            circle_group_loc = np.array(self.circle_group_loc[i])
            for x in end_points:
                mid_point, angle, width, height = get_serial_detect_values(x)
                index_n = 0
                for jj in range(circle_group_loc.shape[0]):
                    try:
                        # inside circle?
                        x_g, y_g, w_g, h_g, index_g = circle_group_loc[jj]
                        if mid_point[0] > x_g and mid_point[0] < x_g + w_g:
                            if mid_point[1] > y_g and mid_point[1] < y_g + h_g:
                                index_n = index_g
                    except:  # no circle in cut
                        index_n = 1
                if index_n == 0:  # still no group?
                    if len(circle_loc):
                        circle_center = np.zeros((circle_loc.shape[0], 2))
                        dist = np.zeros(circle_loc.shape[0])
                        circle_center[:, 0] = circle_loc[:, 0] + 0.5 * circle_loc[:, 2]
                        circle_center[:, 1] = circle_loc[:, 1] + 0.5 * circle_loc[:, 3]
                        for jj in range(circle_center.shape[0]):
                            dist[jj] = np.linalg.norm(circle_center[jj, :] - mid_point)
                        min_dist_index = np.argmin(dist)
                        index_n = circle_loc[min_dist_index, 4]
                if index_n == 0:  # still? no group?
                    index_n = 1
                mid_point = np.append(mid_point.astype(np.int), [index_n, angle, width, height])
                serial_loc_temp.append(mid_point)
        else:
            serial_loc_temp = []

        self.serial_loc.append(serial_loc_temp)

    # SAVE serial INDIV NUMPY
    if SAVE_SERIAL_INDIV_NUMPY:
        image_npy = np.array(image_list)
        image_npy = image_npy.reshape(-1, 32, 32, 1)
        print(f'\nnpy shape : {image_npy.shape}')
        np.save(os.path.join(self.opt.serialpath, self.opt.assembly_name + '_serial.npy'), image_npy)
    return self.serial_loc, self.is_pair_matrices


def mult_detect(self):
    """
    :return: loc : (x, y, w, h, group_index) list of location of mult_num
    """
    SAVE_MULT_INTERMEDIATE_BLACK = True
    refresh_folder(self.opt.mult_black_path)
    image_list = []
    template = plt.imread('./function/utilities/template0.png')
    template = np.around(template * 255).astype(np.uint8)
    template_cut = template[8:32, 8:30]
    # show(template_cut)
    w, h = template.shape
    for i, cut in enumerate(self.cut):
        h_, w_ = cut.shape[:2]
        circle_group_loc = np.array(self.circle_group_loc[i])
        img = np.copy(cut)
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img_gray = cv.threshold(img_gray, 140, 255, cv.THRESH_BINARY_INV)[1]
        # connected components of cut
        ret, labels, stats, c = cv.connectedComponentsWithStats(img_gray, connectivity=8)
        remain_index = []
        remain_index_1 = []
        delete_index = [0]
        # remove unprobable components
        for r in range(1, ret):
            remain_condition = True
            remain_condition_1 = True
            x, y, w, h, area = stats[r]
            if h > 100 or w > 100:  # yellow
                # too long
                remain_condition = False
                remain_condition_1 = False
                delete_index.append(r)
            if h < 20 and w < 20:  # yellow
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
        loc_temp = []
        # show(cut)
        img_cc = (np.isin(labels, remain_index) * 255).astype(np.uint8)
        # show(img_cc)
        img_cc_with_num = (np.isin(labels, remain_index_1) * 255).astype(np.uint8)
        # show(img_cc_with_num)
        img_cc_with_num = cv.cvtColor(img_cc_with_num, cv.COLOR_GRAY2RGB)

        stats = np.array(stats)
        stats = np.delete(stats, delete_index, 0)
        c = np.array(c)
        c = np.delete(c, delete_index, 0)
        ret = c.shape[0]
        if SAVE_MULT_INTERMEDIATE_BLACK:
            cv.imwrite(os.path.join(self.opt.mult_black_path, self.opt.assembly_name + '_' + str(i).zfill(2) + '00.png'), img_cc_with_num)
        for index in range(ret):
            cv.circle(img_cc_with_num, tuple(c[index].astype(np.int)), 10, (255, 0, 0), 3)  # blue
        # show(img_cc_with_num, False)

        methods = ['cv.TM_CCORR']
        for meth in methods:
            img1 = img_cc.copy()
            method = eval(meth)
            res = cv.matchTemplate(img1, template, method)
            threshold = 5000000
            res = ((res > threshold) * 255).astype(np.uint8)
            cv.imwrite(os.path.join(self.opt.mult_black_path, self.opt.assembly_name + '_' + str(i).zfill(2) + '01.png'), res)
            # show(res)
            # connected components of template matching map
            ret1, labels1, stats1, c1 = cv.connectedComponentsWithStats(res, connectivity=8)
            # print(ret1)
            match_index = []
            mult_num_order = 0
            already_chosen_index = []
            taken = []
            for r1 in range(1, ret1):
                points = stats[:, :2]
                x = stats1[r1, 0] + stats1[r1, 2]
                y = stats1[r1, 1] + stats1[r1, 3]
                point1 = np.array([x, y])
                # distance between cc points of template matching mpa & img_cc_with_num
                diff = points - point1
                diff = np.sum(np.absolute(diff), axis=1)
                index = np.argmin(diff)
                if diff[index] < 40:  # yellow
                    cv.circle(img_cc_with_num, tuple(c[index].astype(np.int)), 7, (0, 0, 255), -1)  # red
                    # show(img_cc_with_num, False)
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
                    # print(np.sum(sim))
                    # show(x_cut)
                    if np.sum(sim) < 100:
                        continue

                    # check if one / two components are in approx region of mult_num
                    rect_w = 2 * w
                    rect_h = 1.1 * h
                    cv.rectangle(img_cc_with_num, (int(bottom_left_point[0] - rect_w), int(bottom_left_point[1] - rect_h)), (int(bottom_left_point[0]), int(bottom_left_point[1])), (0, 255, 0), 5)  # green
                    # show(img_cc_with_num)
                    inside_rect_index = []
                    for r in range(ret):
                        x_q, y_q = c[r, :]
                        # cv.circle(img_cc_with_num, tuple(c[index].astype(np.int)), 10, (255, 0, 0), -1)  # blue
                        if x_q > bottom_left_point[0] - rect_w and x_q < bottom_left_point[0]:
                            if y_q > bottom_left_point[1] - rect_h and y_q < bottom_left_point[1]:
                                if r not in already_chosen_index:
                                    inside_rect_index.append(r)
                                    already_chosen_index.append(r)
                    # show(img_cc_with_num)
                    # print(inside_rect_index)
                    points = []
                    for r in inside_rect_index:
                        x, y, w, h = stats[r, :4]
                        points.append([x, y])
                        points.append([x + w, y])
                        points.append([x, y + h])
                        points.append([x + w, y + h])
                    points = np.array(points)
                    # print(points)
                    try:
                        index_x = 1
                        for j in range(circle_group_loc.shape[0]):
                            x_g, y_g, w_g, h_g, index_g = circle_group_loc[j]
                            if bottom_left_point[0] > x_g and bottom_left_point[0] < x_g + w_g:
                                if bottom_left_point[1] > y_g and bottom_left_point[1] < y_g + h_g:
                                    index_x = index_g
                        if index_x in taken:
                            index_x += 1
                        taken.append(index_x)

                        x_min = int(points[:, 0].min())
                        x_max = int(points[:, 0].max())
                        y_min = int(points[:, 1].min())
                        y_max = int(points[:, 1].max())
                        cv.rectangle(img_cc_with_num, (x_min, y_min), (x_max, y_max), (255, 0, 255), 5)  # magenta
                        # show(img_cc_with_num)
                        # print(points)

                        if points.shape[0] == 8:
                            x_sort = np.sort(points[:, 0])
                            x_mid = int((x_sort[3] + x_sort[4]) / 2)

                            cropped_region_0 = img[y_min:y_max, x_min:x_mid]
                            ratio_0 = 24 / cropped_region_0.shape[0]
                            resized_0 = cv.resize(cropped_region_0, None, fx=ratio_0, fy=ratio_0, interpolation=cv.INTER_CUBIC)
                            left_pad_0 = int((32 - resized_0.shape[1]) / 2)
                            right_pad_0 = 32 - resized_0.shape[1] - left_pad_0
                            padded_0 = np.pad(resized_0, ((4, 4), (left_pad_0, right_pad_0), (0, 0)), 'constant', constant_values=255)
                            cv.imwrite(os.path.join(self.opt.multpath, self.opt.assembly_name + '_' + str(i).zfill(2) + '_' + str(mult_num_order).zfill(2) + '_'
                                                    + '00' + '_' + f'{index_x}.png'), padded_0)
                            padded_0 = cv.cvtColor(padded_0, cv.COLOR_BGR2GRAY)
                            image_list.append(padded_0)

                            cropped_region_1 = img[y_min:y_max, x_mid:x_max]
                            ratio_1 = 24 / cropped_region_1.shape[0]
                            resized_1 = cv.resize(cropped_region_1, None, fx=ratio_1, fy=ratio_1, interpolation=cv.INTER_CUBIC)
                            left_pad_1 = int((32 - resized_1.shape[1]) / 2)
                            right_pad_1 = 32 - resized_1.shape[1] - left_pad_1
                            padded_1 = np.pad(resized_1, ((4, 4), (left_pad_1, right_pad_1), (0, 0)), 'constant', constant_values=255)
                            cv.imwrite(os.path.join(self.opt.multpath, self.opt.assembly_name + '_' + str(i).zfill(2) + '_' + str(mult_num_order).zfill(2) + '_'
                                                    + '01' + '_' + f'{index_x}.png'), padded_1)
                            padded_1 = cv.cvtColor(padded_1, cv.COLOR_BGR2GRAY)
                            image_list.append(padded_1)
                            mult_num_order += 1
                        else:
                            cropped_region = img[y_min:y_max, x_min:x_max]
                            ratio = 24 / cropped_region.shape[0]
                            resized = cv.resize(cropped_region, None, fx=ratio, fy=ratio, interpolation=cv.INTER_CUBIC)
                            left_pad = int((32 - resized.shape[1]) / 2)
                            right_pad = 32 - resized.shape[1] - left_pad
                            padded = np.pad(resized, ((4, 4), (left_pad, right_pad), (0, 0)), 'constant', constant_values=255)
                            cv.imwrite(os.path.join(self.opt.multpath, self.opt.assembly_name + '_' + str(i).zfill(2) + '_' + str(mult_num_order).zfill(2) + '_'
                                                    + '00' + '_' + f'{index_x}.png'), padded)
                            mult_num_order += 1
                            padded = cv.cvtColor(padded, cv.COLOR_BGR2GRAY)
                            image_list.append(padded)
                        x, y, w, h = stats[index, :4]
                        loc_temp.append([int(x), int(y), int(w), int(h), index_x])
                    except:
                        pass

            # show(img_cc_with_num, False)
        self.mult_loc.append(loc_temp)

    if True:
        image_npy = np.array(image_list)
        image_npy = image_npy.reshape(-1, 32, 32, 1)
        print(f'npy shape : {image_npy.shape}')
        np.save(os.path.join(self.opt.multpath, self.opt.assembly_name + '_mult.npy'), image_npy)

    return self.mult_loc
