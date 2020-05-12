# stefan_8_pages, view
python main.py --gpu=2 --view_dir=views --checkpoint_dir=./checkpoint/stefan_8pages_views --data_dir=../data/stefan_8pages/ --mode=train_feature_img --lr=1e-4 --C=5 --K=1 --view_num=1 --max_epoch=100 --margin=0.15
python main.py --gpu=2 --view_dir=views --checkpoint_dir=./checkpoint/stefan_8pages_views --data_dir=../data/stefan_8pages/ --mode=train_feature_view --lr=1e-4 --C=5 --K=1 --view_num=12 --max_epoch=200 --margin=0.15
python main.py --gpu=2 --view_dir=views --checkpoint_dir=./checkpoint/stefan_8pages_views --data_dir=../data/stefan_8pages/ --mode=train_trans --lr=1e-4 --C=5 --K=1 --view_num=6 --max_epoch=100 --margin=0.15
python main.py --gpu=2 --view_dir=views --checkpoint_dir=./checkpoint/stefan_8pages_views --data_dir=../data/stefan_8pages/ --mode=train --lr=1e-5 --C=5 --K=1 --view_num=12 --max_epoch=45 --margin=0.15
# stefan_8_pages, view_black
python main.py --gpu=2 --view_dir=views_black --checkpoint_dir=./checkpoint/stefan_8pages_views_black --data_dir=../data/stefan_8pages/ --mode=train_feature_view --lr=1e-4 --C=5 --K=1 --view_num=12 --max_epoch=200 --margin=0.15
python main.py --gpu=2 --view_dir=views_black --checkpoint_dir=./checkpoint/stefan_8pages_views_black --data_dir=../data/stefan_8pages/ --mode=train_feature_img --lr=1e-4 --C=5 --K=1 --view_num=1 --max_epoch=100 --margin=0.15
python main.py --gpu=2 --view_dir=views_black --checkpoint_dir=./checkpoint/stefan_8pages_views_black --data_dir=../data/stefan_8pages/ --mode=train_trans --lr=1e-4 --C=5 --K=1 --view_num=6 --max_epoch=100 --margin=0.15
python main.py --gpu=2 --view_dir=views_black --checkpoint_dir=./checkpoint/stefan_8pages_views_black --data_dir=../data/stefan_8pages/ --mode=train --lr=1e-5 --C=5 --K=1 --view_num=12 --max_epoch=45 --margin=0.15
# # stefan_8_pages, view_gray
# python main.py --gpu=3 --view_dir=views_gray --checkpoint_dir=./checkpoint/stefan_8pages_views_gray --data_dir=../data/stefan_8pages/ --mode=train_feature_img --lr=1e-4 --C=5 --K=1 --view_num=1 --max_epoch=100 --margin=0.15
# python main.py --gpu=3 --view_dir=views_gray --checkpoint_dir=./checkpoint/stefan_8pages_views_gray --data_dir=../data/stefan_8pages/ --mode=train_feature_view --lr=1e-4 --C=5 --K=1 --view_num=12 --max_epoch=200 --margin=0.15
# python main.py --gpu=3 --view_dir=views_gray --checkpoint_dir=./checkpoint/stefan_8pages_views_gray --data_dir=../data/stefan_8pages/ --mode=train_trans --lr=1e-4 --C=5 --K=1 --view_num=6 --max_epoch=100 --margin=0.15
# python main.py --gpu=3 --view_dir=views_gray --checkpoint_dir=./checkpoint/stefan_8pages_views_gray --data_dir=../data/stefan_8pages/ --mode=train --lr=1e-5 --C=5 --K=1 --view_num=12 --max_epoch=45 --margin=0.15
# # stefan_8_pages, view_gray_black
# python main.py --gpu=3 --view_dir=views_gray_black --checkpoint_dir=./checkpoint/stefan_8pages_views_gray_black --data_dir=../data/stefan_8pages/ --mode=train_feature_img --lr=1e-4 --C=5 --K=1 --view_num=1 --max_epoch=100 --margin=0.15
# python main.py --gpu=3 --view_dir=views_gray_black --checkpoint_dir=./checkpoint/stefan_8pages_views_gray_black --data_dir=../data/stefan_8pages/ --mode=train_feature_view --lr=1e-4 --C=5 --K=1 --view_num=12 --max_epoch=200 --margin=0.15
# python main.py --gpu=3 --view_dir=views_gray_black --checkpoint_dir=./checkpoint/stefan_8pages_views_gray_black --data_dir=../data/stefan_8pages/ --mode=train_trans --lr=1e-4 --C=5 --K=1 --view_num=6 --max_epoch=100 --margin=0.15
# python main.py --gpu=3 --view_dir=views_gray_black --checkpoint_dir=./checkpoint/stefan_8pages_views_gray_black --data_dir=../data/stefan_8pages/ --mode=train --lr=1e-5 --C=5 --K=1 --view_num=12 --max_epoch=45 --margin=0.15
