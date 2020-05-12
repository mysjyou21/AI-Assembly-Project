# file runs in freecad
import Part
import Mesh
import os
import sys
import glob
sys.path.append('./')
import argparse
if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser('')
    parser.add_argument('-cad_ext', help='Input CAD file format. Output is always obj')
    parser.add_argument('-target_models_path', help='cad input path. output path is the same.')
    args = parser.parse_known_args(argv)[0]

files = sorted(glob.glob(os.path.join(args.target_models_path, '*' + args.cad_ext)))
for file in files:
    filename = os.path.splitext(file)[0]
    shape = Part.Shape()
    shape.read(file)
    doc = App.newDocument('Doc')
    pf = doc.addObject("Part::Feature", "MyShape")
    pf.Shape = shape
    Mesh.export([pf], filename + '.obj')
    App.closeDocument("Doc")