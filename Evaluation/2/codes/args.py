import tensorflow as tf
import os
import copy


def define_args():
    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_integer('testmode', 0, """testmode, 0:matching trans-view, 1: classification image, 2: classification view, 3: classification trans""")
    tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint', """checkpoint directory""")
    tf.app.flags.DEFINE_string('gpu', '0', """gpu number to be used""")

    tf.app.flags.DEFINE_string('data_dir', '../data', """data directory""")
    tf.app.flags.DEFINE_string('view_dir', 'VIEWS', """VIEWS, VIEWS_BLACK, VIEWS_GRAY, VIEWS_GRAY_BLACK""")
    tf.app.flags.DEFINE_string('img_size', '224,224', """expected input image size, HEIGHT, WIDTH""")
    tf.app.flags.DEFINE_integer('feature_channel', 64, """feature size of the image, view""")
    tf.app.flags.DEFINE_integer('C', 11, """batch size of Class""")
    tf.app.flags.DEFINE_integer('view_num', 12, """number of views for each model""")

    tf.app.flags.DEFINE_integer('max_epoch', 1, """maximum number of epoch""")

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    # print(FLAGS)
    with open(os.path.join(FLAGS.checkpoint_dir, 'config.txt'), 'w') as config:
        d = copy.copy(FLAGS.flag_values_dict())
        delete_list = ['logtostderr', 'alsologtostderr', 'log_dir', 'v', 'verbosity', 'stderrthreshold', 'showprefixforinfo',
                       'run_with_pdb', 'pdb_post_mortem', 'run_with_profiling', 'profile_file', 'use_cprofile_for_profiling',
                       'only_check_args', 'op_conversion_fallback_to_while_loop', 'test_random_seed', 'test_srcdir', 'test_tmpdir',
                       'test_randomize_ordering_seed', 'xml_output_file']
        for dl in delete_list:
            try:
                del d[dl]
            except:
                pass
        config.write(str(d))

    return FLAGS
