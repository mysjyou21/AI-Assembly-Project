# STEFAN
python main.py --mode=train_feature_img --lr=1e-4 --gpu=1 --C=5 --K=1 --view_num=1 --checkpoint_dir=./checkpoint/stefan_line --max_epoch=100 --margin=0.15 --data_dir=../data/stefan_8pages_aug/
python main.py --mode=train_feature_view --lr=1e-4 --gpu=1 --C=5 --K=1 --view_num=12 --checkpoint_dir=./checkpoint/stefan_line --max_epoch=200 --margin=0.15 --data_dir=../data/stefan_8pages_aug/
python main.py --mode=train_trans --lr=1e-4 --gpu=1 --C=5 --K=1 --view_num=6 --checkpoint_dir=./checkpoint/stefan_line --max_epoch=100 --margin=0.15 --data_dir=../data/stefan_8pages_aug/
python main.py --mode=train --lr=1e-5 --gpu=1 --C=5 --K=1 --view_num=12 --checkpoint_dir=./checkpoint/stefan_line --max_epoch=45 --margin=0.15 --data_dir=../data/stefan_8pages_aug
python main.py --mode=test --gpu=1 --C=5 --K=1 --view_num=12 --checkpoint_dir=./checkpoint/stefan_line --data_dir=../data/stefan_8pages_aug

