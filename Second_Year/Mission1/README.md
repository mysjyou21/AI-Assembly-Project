# IKEArobot
final year challenge

conda 4.8.3

python 3.5.6

Ubuntu 20.04 LTS

## Run SNU project only
python main_snu.py

## Run the entire system (example)
python main.py
python snu.py
python windown.py

to run SNU codes automatically(automatic CAD input),
set AUTO = True in main.snu.py and snu.py

to run SNU codes from certain step, add flag start_step
python main_snu.py --start_step=STEP
if there isn't the STEP's CAD file in the input/cad folder, add flag add_cad
python main_snu.py --start_step=STEP --add_cad=1
