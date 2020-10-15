from function.utilities.utils import *


def OCR_serial(session, img_pl, logit_tf, assembly_name, serialpath, ocr_modelpath, N_I, serial_loc):

    BATCH_SIZE = 64

    assembly = assembly_name
    image_paths = sorted(glob.glob(os.path.join(serialpath, '*.png')))
    img_test = []
    for path in image_paths:
        I = np.clip(255. * plt.imread(path), 0, 255)[:, :, :3].astype(np.uint8)
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
        I = 255 - I
        x_start = (32 - I.shape[1]) // 2
        zero_padding = np.zeros((32, 32), dtype=np.uint8)
        zero_padding[:, x_start:x_start + I.shape[1]] = I
        img = zero_padding
        img_test.append(img)
    img_test = np.asarray(img_test)
    img_test = np.reshape(img_test, [-1, 32, 32, 1])

    # tf.reset_default_graph()
    # img_pl = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
    # logit_tf = digit_recognizer(img_pl, is_train=False)

    modelpath = ocr_modelpath
    ckpt = tf.train.latest_checkpoint(modelpath)
    saver = tf.train.Saver()

    saver.restore(session, ckpt)
    num_batches = int(np.ceil(len(img_test) / BATCH_SIZE))

    digit = []
    for idx in range(num_batches):
        img_batch = img_test[BATCH_SIZE*idx: BATCH_SIZE*(idx+1)]
        logit_batch = session.run(logit_tf, feed_dict={img_pl: img_batch})
        digit_batch = np.argmax(logit_batch, axis=1)
        for item in digit_batch:
            digit.append(item)
    digit = np.asarray(digit)

    image_names = [os.path.basename(path) for path in image_paths]
    codes = []
    for name in image_names:
        replace_t1 = name.replace(assembly, '')
        replace_t2 = replace_t1.replace('.png', '')
        replace_t3 = replace_t2.replace('_', '')
        codes.append(replace_t3)

    first_cut_number = int(codes[0][:-4])    # error when 'codes' is empty list
    cut_number_temp = first_cut_number
    large_code_result = []
    large_path_result = []
    large_digit_result = []
    current_code_list = []
    current_path_list = []
    current_digit_list = []
    for i, code in enumerate(codes):
        cut_number = int(code[:2])
        if cut_number == cut_number_temp:
            current_code_list.append(code)
            current_path_list.append(image_paths[i])
            current_digit_list.append(digit[i])
        else:
            cut_number_temp = cut_number
            large_code_result.append(current_code_list)
            large_path_result.append(current_path_list)
            large_digit_result.append(current_digit_list)
            current_code_list = [code]
            current_path_list = [image_paths[i]]
            current_digit_list = [digit[i]]
    large_code_result.append(current_code_list)
    large_path_result.append(current_path_list)
    large_digit_result.append(current_digit_list)

    path_result = []
    digit_result = []
    for i, code_group in enumerate(large_code_result):
        first_serial_idx = int(code_group[0][-4:-2])
        small_path_result = []
        small_digit_result = []
        current_path_list = []
        current_digit_list = []
        serial_idx_temp = first_serial_idx
        for j, code in enumerate(code_group):
            serial_idx = int(code[-4:-2])
            if serial_idx == serial_idx_temp:
                current_path_list.append(large_path_result[i][j])
                current_digit_list.append(large_digit_result[i][j])
            else:
                serial_idx_temp = serial_idx
                small_path_result.append(current_path_list)
                small_digit_result.append(tostring(current_digit_list))
                current_path_list = [large_path_result[i][j]]
                current_digit_list = [large_digit_result[i][j]]
        small_path_result.append(current_path_list)
        small_digit_result.append(tostring(current_digit_list))
        path_result.append(small_path_result)
        digit_result.append(small_digit_result)

    digit_with_null_temp = []
    j = 0
    for i in range(N_I):
        if j < len(digit_result):
            path_group = path_result[j]
            first_path = os.path.basename(path_group[0][0])
            temp = first_path.replace(assembly, '')
            temp = temp.replace('.png', '')
            code = temp.replace('_', '')
            cut_idx = int(code[:-4])
            if i == cut_idx:
                digit_with_null_temp.append(digit_result[j])
                j += 1
            else:
                digit_with_null_temp.append([])
        else:
            digit_with_null_temp.append([])

    digit_with_null = []
    for n, digits_in_cut_temp in enumerate(digit_with_null_temp):
        if digits_in_cut_temp == []:
            digit_with_null.append([])
        else:
            grp_idx_list = []
            for i in range(len(digits_in_cut_temp)):
                grp_idx = int(serial_loc[n][i][2])
                grp_idx_list.append(grp_idx)
            grp_idx_list = np.asarray(grp_idx_list)
            max_grp_idx = max(grp_idx_list)
            digits_in_cut = []
            for i in range(1, max_grp_idx + 1, 1):
                current_idx_where = np.asarray(np.where(grp_idx_list == i))[0]
                temp = []
                for idx in current_idx_where:
                    temp.append(digits_in_cut_temp[idx])
                digits_in_cut.append(temp)
            digit_with_null.append(digits_in_cut)

    OCR_serial_result = digit_with_null
    OCR_mult_intermidiate_result = []

    for digits_in_cut in OCR_serial_result:
        mult_in_cut = []
        for _ in digits_in_cut:
            mult_in_circle = ['1']
            mult_in_cut.append(mult_in_circle)
        OCR_mult_intermidiate_result.append(mult_in_cut)

    return OCR_serial_result, OCR_mult_intermidiate_result


def OCR_mult(session, img_pl, logit_tf, assembly_name, multpath, ocr_modelpath, OCR_serial_result, OCR_mult_intermidiate_result):

    BATCH_SIZE = 64

    assembly = assembly_name
    image_paths = sorted(glob.glob(os.path.join(multpath, '*.png')))
    img_test = []
    for path in image_paths:
        I = np.clip(255. * plt.imread(path), 0, 255)[:, :, :3].astype(np.uint8)
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
        I = 255 - I
        x_start = (32 - I.shape[1]) // 2
        zero_padding = np.zeros((32, 32), dtype=np.uint8)
        zero_padding[:, x_start:x_start + I.shape[1]] = I
        img = zero_padding
        img_test.append(img)
    img_test = np.asarray(img_test)
    img_test = np.reshape(img_test, [-1, 32, 32, 1])

    # tf.reset_default_graph()
    # img_pl = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
    # logit_tf = digit_recognizer(img_pl, is_train=False, reuse=tf.AUTO_REUSE)

    model_path = ocr_modelpath
    ckpt = tf.train.latest_checkpoint(model_path)
    saver = tf.train.Saver()

    saver.restore(session, ckpt)
    num_batches = int(np.ceil(len(img_test) / BATCH_SIZE))
    digit = []
    for idx in range(num_batches):
        img_batch = img_test[BATCH_SIZE*idx: BATCH_SIZE*(idx+1)]
        logit_batch = session.run(logit_tf, feed_dict={img_pl: img_batch})
        digit_batch = np.argmax(logit_batch, axis=1)
        for item in digit_batch:
            digit.append(item)
    digit = np.asarray(digit)

    image_names = [os.path.basename(path) for path in image_paths]
    codes = []
    for name in image_names:
        replace_t1 = name.replace(assembly, '')
        replace_t2 = replace_t1.replace('.png', '')
        replace_t3 = replace_t2.replace('_', '')
        codes.append(replace_t3)

    last_cut = max([int(code[:-5]) for code in codes])
    last_loc = max([int(code[-5:-3]) for code in codes])
    last_circle = max([int(code[-1]) for code in codes])

    digit_np = np.zeros([last_cut + 1, last_circle + 1, last_loc + 1], dtype=int)
    for n, code in enumerate(codes):
        cut_num = int(code[:-5])
        loc_num = int(code[-5:-3])
        circle_num = int(code[-1])
        if digit_np[cut_num, circle_num, loc_num] == 0:
            digit_np[cut_num, circle_num, loc_num] = digit[n]
        else:
            digit_temp = digit_np[cut_num, circle_num, loc_num]
            digit_result = int(str(digit_temp) + str(digit[n]))
            digit_np[cut_num, circle_num, loc_num] = digit_result
    digit_np_temp = np.zeros([last_cut + 1, last_circle + 1], dtype=int)
    for i in range(last_cut + 1):
        for j in range(last_circle + 1):
            digit_with_cut_circle = digit_np[i, j]
            idx_not_zero = np.where(digit_with_cut_circle != 0)[0]
            if len(idx_not_zero) >= 2:
                digit_np_temp[i, j] = np.around(np.mean(digit_with_cut_circle[idx_not_zero])).astype(int)
            else:
                digit_np_temp[i, j] = np.sum(digit_with_cut_circle)

    OCR_mult_result = OCR_mult_intermidiate_result
    for code in codes:
        cut_num = int(code[:-5])
        circle_num = int(code[-1])
        if circle_num <= len(OCR_serial_result[cut_num]):
            OCR_mult_result[cut_num][circle_num - 1] = [str(digit_np_temp[cut_num, circle_num])]

    return OCR_mult_result

