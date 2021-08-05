# Robot

## Folders

assemblies_300dpi/ 폴더별로 분류된 input

output/ 결과 csv for Mission1 & Mission2

results/ 폴더별로 분류된 중간산출물

    circle, mult, rectangle, serial

    group_image / 같은 그룹은 같은 색상으로 표시

    serial_black/ 번호 검출 중간산출물 검은 배경
    
    serial_white/ 번호 검출 중간산출물 하얀 배경
    
    serial/ 번호 검출 6자리 숫자 통째로

    mult_black/ 수량 검출 중간 산출물

    results_*.txt 갯수의 추정치 - 참값 cut 별 정리

function/ 함수
    OCR/ 번호 및 수량 검출 OCR 모델

    utilities/

        csv_to_lab.py : csv에서 lab.npy 만드는 파일 
    
        template0.png : '4x' 검출 때 필요
    
        utils.py : label matrix 여러가지, 유틸리티

        material_label.csv : material label 파일

        action_label.csv : action label 파일
    
main.py


## Usage

### single input folder

1. copy-paste query images in ./assemblies_300dpi/input

2. python main.py

   or

   python main.py

3. python main.py --assembly_name=(folder_name)

### multiple input folders(all folders in assemblies_300dpi)

1. bash test.sh
