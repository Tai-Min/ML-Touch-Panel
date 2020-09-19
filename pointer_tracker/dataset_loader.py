
import random
import os
import json

def dataset_loader(data_path, annotation_path):
    data_path = data_path.decode('utf-8')
    annotation_path = annotation_path.decode('utf-8')

    with open(os.path.join(data_path, 'structure.json')) as fp:
        dataset_indexed = json.load(fp)

    with open(os.path.join(data_path, 'weights.json')) as fp:
        dataset_weights = json.load(fp)

    while dataset_indexed:
        video_folder_name = random.choice(list(dataset_indexed.keys()))

        # check if video folder is empty
        if not dataset_indexed[video_folder_name]:
            print("Folder %s is empty, skipping." % video_folder_name)
            dataset_indexed.pop(video_folder_name, None)
            continue

        group_folder_name = random.choice(list(dataset_indexed[video_folder_name].keys()))

        # check if group folder is empty
        if not dataset_indexed[video_folder_name][group_folder_name]:
            print("Folder %s/%s is empty, skipping." % (video_folder_name, group_folder_name))
            dataset_indexed[video_folder_name].pop(group_folder_name, None)
            continue

        # remove exemplar file from list
        if 'exemplar.JPEG' in dataset_indexed[video_folder_name][group_folder_name]:
            dataset_indexed[video_folder_name][group_folder_name].remove('exemplar.JPEG')

        # check if there is at least one element in folder excluding exemplar
        if len(dataset_indexed[video_folder_name][group_folder_name]) == 0:
            print("No source frames in %s/%s, skipping." % (video_folder_name, group_folder_name))
            dataset_indexed[video_folder_name].pop(group_folder_name, None)
            continue

        exemplar_data_path = data_path + '/' + video_folder_name + '/' + group_folder_name + '/exemplar.JPEG'
        exemplar_annotation_path = annotation_path + '/' + video_folder_name + '/' + group_folder_name + '/exemplar.txt'

        # check if there is exemplar in group folder
        if not os.path.isfile(exemplar_data_path) or not os.path.isfile(exemplar_annotation_path):
            print("No exemplar in %s/%s, skipping." % (dataset_indexed[video_folder_name], dataset_indexed[video_folder_name][group_folder_name]))
            dataset_indexed[video_folder_name].pop(group_folder_name, None)
            continue

        selected_frame = random.choice(dataset_indexed[video_folder_name][group_folder_name])

        # remove it from list so it won't be used again in given epoch
        dataset_indexed[video_folder_name][group_folder_name].remove(selected_frame)

        frame_data_path = data_path + '/' + video_folder_name + '/' + group_folder_name + '/' + selected_frame
        frame_annotation_path = annotation_path + '/' + video_folder_name + '/' + group_folder_name + '/' + selected_frame[0:-5] + '.txt'

        # check if selected frame really exists in dataset
        if not os.path.isfile(frame_data_path) or not os.path.isfile(frame_annotation_path):
            print("Frame %s in folder %s/%s does not exists, skipping." % (selected_frame, video_folder_name, group_folder_name))
            continue

        yield ((exemplar_data_path, frame_data_path),(exemplar_annotation_path, frame_annotation_path), dataset_weights[video_folder_name])