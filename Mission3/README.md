# CAD_Object_Retrieval

## Environment

python 3

## Required packages

tensorflow 1.x (1.13 preferred)

opencv-python

blender 2.82

freecad

antiprism

## Install

### Ubuntu

##### blender 2.82 install
sudo snap install blender --classic
##### freecad install
sudo apt install freecad
##### antiprism install
sudo add-apt-repository ppa:antiprism//ppa

sudo apt-get update

sudo apt-get install antiprism

---

### Windows
after install, add directory to sys path
##### blender 2.82 install
download and install from blender.org/download/
##### freecad install
download and install form freecadweb.org/downloads.php
##### antiprism install
download and install form antiprism.com/download/index.html



## Run

#### create 2D projections

cd ./data/stefan

python render_run.py

## Dataset
1. download benchmark SHREC2013, SHREC2014 Data in [Sketch3Dtoolkit](https://github.com/garyzhao/Sketch3DToolkit)
2. STEFAN

## Model
based on [Deep Cross-modality Adaptation for Sketch-based 3D Shape Retrieval](http://openaccess.thecvf.com/content_ECCV_2018/html/Jiaxin_Chen_Deep_Cross-modality_Adaptation_ECCV_2018_paper.html)

## Train network
python codes/main.py --mode=train

4 modes, train_feature_img, train_feature_view, train_trans, train

## Test network
python codes/main.py --mode=test --max_epoch=1 --checkpoint_dir=./checkpoint --testmode=1

4 testmodes, 1: img classification, 2: view classification, 3: trans classification, 4: view matching, 0: view-trans matching

Matching results will be saved in checkpoint_dir/Test
