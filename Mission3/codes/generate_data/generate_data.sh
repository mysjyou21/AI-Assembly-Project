assembly_name=VRep
# assembly_name=stefan_8pages
python render_run.py -n ${assembly_name} -r views -c stl
# python render_run.py -n ${assembly_name} -r views_black -c stl
python render_run.py -n ${assembly_name} -r views_gray -c stl
python render_run.py -n ${assembly_name} -r views_gray_black -c stl
