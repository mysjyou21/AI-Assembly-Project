# Mission 1, 2
## Folders

### Input
assemblies_300dpi : 폴더별로 분류된 input

### Output
output : 결과 csv for Mission1 & Mission2

### Intermediate results
intermediate_results : 폴더별로 분류된 중간산출물

    circle : 원형 말풍선 검출 결과

    group_image  : 같은 그룹은 같은 색상으로 표시된 결과

    mult : 수량 검출 결과

    mult_black : 수량 검출 중간 산출물

    rectangle : 사각 말풍선 검출 결과

    serial : 부품번호 검출 번호 결과

    serial_black : 부품번호 검출 중간산출물 검은 배경
    
    serial_white : 부품번호 검출 중간산출물 하얀 배경
    
    serial_whole : 번호 검출 6자리 숫자 통째로

### Functions
function : 함수들이 저장된 폴더

    OCR : 부품번호 및 수량의 검출된 결과에 대한 OCR 모델

    utilities :

        csv_to_lab.py : csv에서 lab.npy 만드는 파일 
    
        template0.png : 수량 검출을 위해 필요
    
        utils.py : label matrix 여러가지, 유틸리티

        material_label.csv : material label 파일

        action_label.csv : action label 파일
    

config.py : configuration

main.py

test.sh : input folder 내에 있는 모든 assemblies에 대해 코드 실행


## Usage

### single input folder

1. put query images in ./assemblies_300dpi/input/cuts

2. python main.py


or

1. put query images in ./assemblies_300dpi/[assembly_name]/cuts

2. python main.py --assembly_name=[assembly_name]

### multiple input folders(all folders in assemblies_300dpi)

1. bash test.sh
