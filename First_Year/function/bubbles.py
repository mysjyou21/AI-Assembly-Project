import sys
sys.path.append('./function/utilities')
from function.utilities.utils import *
# 준형
# def hough_circle
# def speech_rectangle
# def bubble_detector sub function


def circle_detect(self, I, i, param1=80, param2=83, ratio_threshold=0.2):
    # TODO: 준형
    """
    check whether speech bubble is in image I
    : param I: current stephan cut image
    : return: loc: list of locations(x, y, w, h, group_idx)
    """
    img = np.copy(I)
    Ig = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    Ig = 255 - Ig
    h, w = Ig.shape
    h_ = 1654
    scale = h / h_
    w_ = int(np.around(w / scale))
    # change size of image
    Ig_ = cv.resize(Ig, (w_, h_))
    minR, maxR = int(.09 * w_), int(.21 * w_)

    PARAM1 = param1
    PARAM2 = param2

    cir_coord_ = cv.HoughCircles(Ig_, cv.HOUGH_GRADIENT, 1, 100, param1=PARAM1, param2=PARAM2,
                                 minRadius=minR, maxRadius=maxR)
    if cir_coord_ is not None:
        cir_coord_ = cir_coord_[0]
        loc_ = np.zeros([len(cir_coord_), 4], dtype=int)
        for n in range(len(cir_coord_)):
            x_, y_, r_ = cir_coord_[n]
            x1_ = int(np.around(max(0, x_ - r_)))
            y1_ = int(np.around(max(0, y_ - r_)))
            w_ = int(np.around(2 * r_))
            h_ = w_
            loc_[n] = [x1_, y1_, w_, h_]

       # give constraint to circles
        detected_ = []
        circles_ = []
        for x1_, y1_, w_, h_ in loc_:
            img_ = Ig_[y1_: y1_ + h_, x1_: x1_ + w_] / 255.
            if img_.shape[0] == img_.shape[1]:
                detected_.append(Ig_[y1_: y1_ + h_, x1_: x1_ + w_] / 255.)
                circle_ = np.zeros([w_, w_], dtype=np.uint8)
                circle_ = cv.circle(circle_, (w_ // 2, w_ // 2), w_ // 2, (255, 255, 255), 4)
                circles_.append(circle_ / 255.)
        ratio = [np.sum(detected_[n] * circles_[n]) / np.sum(circles_[n]) for n in range(len(detected_))]
        RATIO_THRESHOLD = ratio_threshold

        to_be_saved_idx = []
        for n in range(len(ratio)):
            if ratio[n] > RATIO_THRESHOLD:
                to_be_saved_idx.append(n)
            '''
            else:
                print(ratio[n])
            '''

        cir_coord_temp = np.zeros([len(to_be_saved_idx), 3])
        loc_temp = np.zeros([len(to_be_saved_idx), 4], dtype=int)
        for k, idx in enumerate(to_be_saved_idx):
            cir_coord_temp[k] = cir_coord_[idx]
            loc_temp[k] = loc_[idx]
        cir_coord_ = cir_coord_temp
        loc_ = loc_temp
        cir_coord = np.zeros_like(cir_coord_)
        for n in range(len(cir_coord_)):
            x, y, r = cir_coord_[n] * scale
            cir_coord[n] = [x, y, r]

        for x, y, r in cir_coord:
            x1, y1 = int(x - r), int(y - r)
            x2, y2 = int(x + r), int(y + r)
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        loc = []
        for n in range(len(cir_coord)):
            x, y, r = cir_coord[n]
            x1 = int(np.around(max(0, x - r)))
            y1 = int(np.around(max(0, y - r)))
            w = int(np.around(2 * r))
            h = w
            loc.append([x1, y1, h, w])

        if len(loc) == 0:
            loc = []

    else:
        cir_coord = []
        loc = []

    # [grouping circles]
    # group index is appended to loc
    group_img = np.zeros_like(img)
    group_info = np.zeros(len(cir_coord))
    for n in range(len(cir_coord)):
        margin = 1.1
        cv.circle(group_img, tuple(cir_coord[n][:2].astype(np.int)), int(margin * cir_coord[n][2]), (255, 255, 255),
                  -1)
    group_img = cv.cvtColor(group_img, cv.COLOR_RGB2GRAY)
    _, labels, _, _ = cv.connectedComponentsWithStats(group_img.astype(np.uint8), connectivity=8)
    for n in range(len(cir_coord)):
        group_info[n] = labels[int(cir_coord[n][1]), int(cir_coord[n][0])]
    if len(group_info) != 0:
        for n in range(len(loc)):
            loc[n].append(int(group_info[n]))

    cv.imwrite(os.path.join(self.opt.circlepath, f'circle_{i}.png'), img)

    return loc


def rectangle_detect(self, I, i):
    """
    check whether speech rectangle is in image I
    : param I: current stefan cut image
    : return: loc: list of locations(x, y, w, h)
    """
    # make image which has vertical & horizontal component of original cut,
    # using erosion and dilation
    img = np.copy(I)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bw = 255 - gray
    hor = np.copy(bw)
    ver = np.copy(bw)
    # show(bw, 'bw')

    cols = hor.shape[1]
    hor_size = cols // 100  # <- changed
    hor_se = cv.getStructuringElement(cv.MORPH_RECT, (hor_size, 1))
    hor = cv.erode(hor, hor_se)
    hor = cv.dilate(hor, hor_se)
    # show(hor, 'hor')

    rows = ver.shape[0]
    ver_size = rows // 100  # <- changed
    ver_se = cv.getStructuringElement(cv.MORPH_RECT, (1, ver_size))
    ver = cv.erode(ver, ver_se)
    ver = cv.dilate(ver, ver_se)
    # show(ver, 'ver')

    merged = np.maximum(hor, ver)
    # show(merged, 'merged')

    # detect corners by using Harris corner detection
    harris = cv.cornerHarris(np.float32(merged), 10, 3, 0.04)
    harris = cv.dilate(harris, None)
    _, thresh = cv.threshold(harris, 0.5 * harris.max(), 255, 0)
    thresh = np.uint8(thresh)
    # show(thresh, 'thresh')

    _, _, _, centroids = cv.connectedComponentsWithStats(thresh)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(merged, np.float32(centroids), (5, 5), (-1, -1), criteria).astype(np.int0)
    corners_temp = [tuple(row) for row in corners]
    corners = np.unique(corners_temp, axis=0)

    while len(corners):
        x_list, y_list = corners[:, 0], corners[:, 1]
        temp = np.copy(corners)
        px_threshold = 1
        deleted = []
        for n in range(len(corners)):
            x, y = corners[n]
            x_idx = np.argwhere(np.abs(x_list - x) <= px_threshold)
            y_idx = np.argwhere(np.abs(y_list - y) <= px_threshold)
            if len(x_idx) == 1 or len(y_idx) == 1:
                deleted.append(n)
        temp = np.delete(temp, deleted, 0)
        if len(temp) < 4:
            temp = np.array([])
        if len(corners) != len(temp):
            corners = temp
        else:
            break
    # for corner in corners:
    #     x, y = corner
    #     cv.circle(img, (x, y), 10, (255, 0, 0), -1)
    loc = []
    iter = 0

    if len(corners) % 4 == 0:
        corners_array = np.copy(corners)
        corners_list = corners_array.tolist()
        while iter < 1000:
            if len(corners_list) == 0:
                break
            x_min = np.min(corners_array[:, 0])
            vertical_idx = np.squeeze(np.argwhere(np.abs(corners_array[:, 0] - x_min) <= 1))
            if vertical_idx.shape == ():  # <- changed
                break
            a = np.argwhere(np.abs(corners_array[:, 0] - x_min) <= 1)
            y_min = np.min(corners_array[vertical_idx][:, 1])
            horizontal_idx = np.squeeze(np.argwhere(np.abs(corners_array[:, 1] - y_min) <= 1))
            if horizontal_idx.shape == ():  # <- changed
                break
            up_left_idx = np.squeeze(np.intersect1d(vertical_idx, horizontal_idx))
            vertical_points = corners_array[vertical_idx]
            horizontal_points = corners_array[horizontal_idx]

            up_left = corners_array[up_left_idx]
            down = vertical_points[np.argsort(vertical_points[:, 1])[1]]
            right = horizontal_points[np.argsort(horizontal_points[:, 0])[1]]
            down_right = None

            x_temp = right[0]
            y_temp = down[1]

            is_break = False
            for point in [[x_temp, y_temp], [x_temp - 1, y_temp], [x_temp + 1, y_temp], [x_temp, y_temp - 1],
                          [x_temp, y_temp + 1]]:
                for n in range(len(corners_array)):
                    if np.all(corners_array[n] == np.array(point)):
                        down_right = np.asarray(point)
                        is_break = True
                        break
                if is_break:
                    break
            if down_right is not None:
                to_be_deleted = [point.tolist() for point in [up_left, down, right, down_right]]
                # for point in to_be_deleted:
                #     cv.circle(img, tuple(point), 50, (0, 0, 255), -1)
            else:
                to_be_deleted = [point.tolist() for point in [up_left, down, right]]  # <- changed
            for point in to_be_deleted:
                corners_list.remove(point)
            corners_array = np.asarray(corners_list)

            if down_right is not None:
                x, y = up_left
                h = down_right[1] - y
                w = down_right[0] - x
                rectangle = [x, y, w, h]
                for mult_loc in self.mult_loc[i]:
                    x_mult = mult_loc[0] + mult_loc[2]
                    y_mult = mult_loc[1] + mult_loc[3]
                    if not (x_mult > x and x_mult < x + w):
                        if not (y_mult > y and y_mult < y + h):
                            loc.append(rectangle)
                iter += 1

    elif len(corners) % 2 == 0:
        D_MIN = 1
        for corner in corners:
            LU = corner
            x, y = LU

            is_L = np.abs(corners[:, 0] - x) <= D_MIN
            is_low = corners[:, 1] > y
            L_idx = np.argwhere(is_L & is_low == True)
            if len(L_idx) == 0:
                continue

            is_U = np.abs(corners[:, 1] - y) <= D_MIN
            is_right = corners[:, 0] > x
            U_idx = np.argwhere(is_U & is_right == True)
            if len(U_idx) == 0:
                continue

            L_idx = np.squeeze(L_idx, axis=1)
            L = np.asarray([corners[i] for i in L_idx])
            LD = np.squeeze([L[i] for i in range(len(L)) if L[:, 1][i] == L[:, 1].min()])

            U_idx = np.squeeze(U_idx, axis=1)
            U = np.asarray([corners[i] for i in U_idx])
            RU = np.squeeze([U[i] for i in range(len(U)) if U[:, 0][i] == U[:, 0].min()])

            RD_x, RD_y = RU[0], LD[1]
            RD = np.squeeze([corners[i] for i in range(len(corners)) if np.abs(corners[i][0] - RD_x) <= D_MIN
                             and np.abs(corners[i][1] - RD_y) <= D_MIN])
            if len(RD) == 0:
                continue

            x, y = LU
            w, h = RD - LU
            rectangle = [x, y, w, h]
            for mult_loc in self.mult_loc[i]:
                x_mult = mult_loc[0] + mult_loc[2]
                y_mult = mult_loc[1] + mult_loc[3]
                if not (x_mult > x and x_mult < x + w):
                    if not (y_mult > y and y_mult < y + h):
                        loc.append(rectangle)
            iter += 1

    # exclude rectangle whose h/w ratio is too small(e.g., 1/3) or big(e.g., 3)
    to_be_deleted = []
    for rectangle in loc:
        h, w = rectangle[2], rectangle[3]
        ratio = h / w
        if ratio < 1 / 3 or ratio > 3:
            to_be_deleted.append(rectangle)
    for rectangle in to_be_deleted:
        loc.remove(rectangle)

    # exclude rectangle whose size is too small
    to_be_deleted = []
    for rectangle in loc:
        h, w = rectangle[2], rectangle[3]
        if min(h, w) < 0.05 * min(bw.shape):
            to_be_deleted.append(rectangle)
    for rectangle in to_be_deleted:
        loc.remove(rectangle)

    # save images
    for rectangle in loc:
        x1, y1, w, h = rectangle
        x2, y2 = x1 + w, y1 + h
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv.imwrite(os.path.join(self.opt.rectanglepath, f'rectangle_{i}.png'), img)

    return loc
