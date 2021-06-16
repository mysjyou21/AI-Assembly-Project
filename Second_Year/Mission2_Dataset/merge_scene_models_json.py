import json

scene_dicts = []


for i, name in enumerate(['eunji', 'isaac', 'jaewoo', 'minwoo']):
	json_filename = './scene_models_' + name + '.json'
	with open(json_filename, 'r') as f:
		scene_dict = json.load(f)
		scene_dicts.append(scene_dict)

merged_scene_dict = {}
for scene_dict in scene_dicts:
	merged_scene_dict.update(scene_dict)

with open('./scene_models.json', 'w') as f:
	json.dump(merged_scene_dict, f, indent=2, sort_keys=True)