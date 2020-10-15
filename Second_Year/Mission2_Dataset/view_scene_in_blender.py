import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--cad_path', default='./models_cad', help='cad directory')
parser.add_argument('--json', default='scene_models', help='which scene_models.json to read')
parser.add_argument('--scene_name', default='2-3-4-6', help='for example, 2-5')
parser.add_argument('--blender', default='blender', help='set to your blender command')
parser.add_argument('--scale', type=int, default=100, help='cad_model = 1/Scale * cad_model (original cad is too big)')
args = parser.parse_args()

flag = ' -cad_path ' + args.cad_path + ' -scene_name ' + args.scene_name + ' -json ' + args.json + ' -view_scene_mode True'\
+ ' -scale ' + str(args.scale)

mute = False
# Run Blender with python script
if mute:
    os.system(args.blender + ' -P ./render_views.py > ./render_stdout.txt -- ' + flag + ' 2>&1')
else:
    os.system(args.blender + ' -P ./render_views.py -- ' + flag)