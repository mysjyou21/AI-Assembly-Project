#python main.py --mode=train_feature_img --lr=1e-4 --gpu=2 --C=3 --K=4 --view_num=1 --checkpoint_dir=./checkpoint/20200420_0.15 --max_epoch=30 --margin=0.15
#python main.py --mode=train_feature_view --lr=1e-4 --gpu=2 --C=3 --K=3 --view_num=6 --checkpoint_dir=./checkpoint/20200420_0.15 --max_epoch=30 --margin=0.15
#python main.py --mode=train_trans --lr=1e-4 --gpu=2 --C=3 --K=3 --view_num=6 --checkpoint_dir=./checkpoint/20200420_0.15 --max_epoch=40 --margin=0.15
python main.py --mode=train --lr=5e-5 --gpu=2 --C=3 --K=3 --view_num=6 --checkpoint_dir=./checkpoint/20200417_0.15 --max_epoch=300 --margin=0.15
