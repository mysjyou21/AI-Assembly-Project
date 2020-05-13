assembly_name=VRep
cad_type=ply
# assembly_name=stefan_8pages
python render_run.py -n ${assembly_name} -r views -c ${cad_type}
python render_run.py -n ${assembly_name} -r views_black -c ${cad_type}
python render_run.py -n ${assembly_name} -r views_gray -c ${cad_type}
python render_run.py -n ${assembly_name} -r views_gray_black -c ${cad_type}
