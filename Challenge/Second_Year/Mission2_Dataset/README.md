# Mission2_Dataset
로봇과제 미션 2 데이터셋 생성 코드 (이삭)

## Settings

### CloudCompare

CloudCompare 설치 후, CLI에서 CloudCompare 명령어로 해당 프로그램이 동작하도록 한다. 또는, CloudCompare가 실행되는 명령어로 args.py의 --cloudcompare의 default 값을 변경한다.

### Blender

Blender2.82a 버전 설치 후, CLI에서 blender 명령어로 해당 프로그램이 동작하도록 한다. 또는, blender가 실행되는 명령어로 args.py의 --blender의 default 값을 변경한다.

library 설치 : render_views.py line3의 출력을 통해 blender의 python path를 확인한 후, 여기에 python -m pip install --ugrade [library] 로 opencv-python, matplotlib, scipy 설치

## Excecution

### Mode 1 : scene generation
*WARNING : This step overwrites scene_models.json !!!*

몇장의 이미지를 생성할 것이며, 각 이미지에 CAD 모델들을 어떻게 랜덤하게 배치할 것인지 scene_models.json에 저장

render_views.py의 -moving_camera_mode의 default=''로 해놓고 사용

--num_scene : number of scenes

--num_cad_range : 'min,max' number of cad in single scene

### Mode 2 : rendering
Blender로 views, views_gray_black, views_gray_black_occlu 생성
### Mode 3 : post-processing
Blender 생성물로 rgb, mask, mask_occlu 생성 
### Mode 4 : label generation
label 생성

{scene_name:{rgb_name: [obj0, obj1, ...]}}

obj# = {bbox_obj : mask의 xywh, bbox_visib : mask_occlu의 xywh, cam_K : camera intrinsic, cam_R_m2c : cad2cam R, cam_t_m2c : cad2cam t, obj_id : models_cad에서의 cad index, obj_type : New or Mid}
### Mode 5 : add non-part components
rgb에 말풍선, 도구 등의 non-part components 추가하여 synth_result 생성


## Test your own scene_models.json

overwrite scene_models.json to your own file. Then run python main.py [--mode 234]

## Other functions

#### view_scene_in_blender.py
view scene by scene_name in scene_models.json or other json

#### merge_scene_models_json.py
merge [eunji, isaac, jaewoo, minwoo]'s scene_models.json to scene_model.json
