'''usage : python create_cla_files.py -- -folder_path ../../data/stefan_8pages'''

import argparse
import sys
import os
import glob
import numpy as np
from treelib import Node, Tree

# for inheriting args
if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser('')
    parser.add_argument('-folder_path')
    args = parser.parse_known_args(argv)[0]

# define various directories
assert args.folder_path.endswith('/') == False
folder_path = args.folder_path
folder_name = folder_path.split('/')[-1]
target_models_path = os.path.join(folder_path, 'TARGET_MODELS/models')
images_path = os.path.join(folder_path, 'IMAGES')
views_path = os.path.join(folder_path, 'VIEWS')
views_gray_path = os.path.join(folder_path, 'VIEWS_GRAY')


def write_cla(mode, cla_name, start_path, extension=None):
    assert mode in ['model', 'image']
    # build directory tree
    tree = Tree()
    root_name = os.path.basename(start_path)
    tree.create_node(root_name, root_name)
    for root, dirs, files in os.walk(start_path):
        dirs.sort()
        files.sort()
        if extension:
            files = [f for f in files if os.path.splitext(f)[1] == extension]
        files = [os.path.splitext(f)[0] for f in files]
        parent = os.path.basename(root)
        for _dir in dirs:
            tree.create_node(_dir, _dir, parent=parent)
        tree.update_node(parent, data=files)
    tree.show()

    with open(os.path.join(folder_path, cla_name + '.cla'), 'w') as f:
        # write header
        f.write(folder_name + ' 1\n')
        num_total_classes = tree.size() - 1
        num_total_files = 0
        for node in tree.all_nodes_itr():
            num_total_files += len(node.data)
        num_total_classes = num_total_files if mode == 'model' else num_total_classes
        f.write(f'{num_total_classes} {num_total_files}\n\n')

        # write parent classes
        for i, node in enumerate(tree.all_nodes_itr()):
            if mode == 'model':
                for file in node.data:
                    f.write(f'{file} 0 1\n{file}\n\n')
            else:
                if i == 0: 
                    continue
                class_name = node.tag
                class_parent = tree.parent(class_name).tag
                class_parent = '0' if class_parent == root_name else class_parent
                class_num_files = len(node.data)
                f.write(f'{class_name} {class_parent} {class_num_files}\n')
                for file in node.data:
                    f.write(f'{file}\n')
                f.write('\n')



write_cla('model', 'Model', target_models_path, extension='.obj')
write_cla('image', 'Image_Train', images_path)
write_cla('image', 'Image_Test', images_path)
