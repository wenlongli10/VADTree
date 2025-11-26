import json, pickle

root_path = 'data/TAPOS'
split = 'train'

tapos_json_path = f'{root_path}/tapos_annotation.json'
anno_json = json.load(open(tapos_json_path, 'r'))

vids = anno_json.keys()
dict_data = dict()

for vid in vids:
    for instance in anno_json[vid]:
        if anno_json[vid][instance]['subset'] == split:
            pass
        else:
            continue
        dict_data[f'{vid}_{instance}'] = dict()

        tmp = instance.split('_')
        instance_start = float(tmp[1] + '.' + tmp[2])  # start at this shot
        instance_end = float(tmp[3] + '.' + tmp[4])  # end at this shot
        duration = instance_end - instance_start
        n_frames = int(anno_json[vid][instance]['total_frames'])
        bnds = anno_json[vid][instance]['substages'][1:-1]  # excluded the first and the end
        fps = float(anno_json[vid][instance]['total_frames']) / duration
        bnds_sec = [bnd / fps for bnd in bnds]

        dict_data[f'{vid}_{instance}']['num_frames'] = n_frames
        dict_data[f'{vid}_{instance}']['path_video'] = f'{root_path}/videos/{vid}.mp4'
        dict_data[f'{vid}_{instance}']['fps'] = fps
        dict_data[f'{vid}_{instance}']['video_duration'] = duration
        dict_data[f'{vid}_{instance}']['path_frame'] = f'{root_path}/images/{split}/{vid}_{instance}'
        dict_data[f'{vid}_{instance}']['f1_consis'] = [1.]
        dict_data[f'{vid}_{instance}']['f1_consis_avg'] = 1.
        dict_data[f'{vid}_{instance}']['substages_myframeidx'] = [bnds]
        dict_data[f'{vid}_{instance}']['substages_timestamps'] = [bnds_sec]

with open(f'{root_path}/tapos_{split}.pkl', 'wb') as f:
    pickle.dump(dict_data, f)
