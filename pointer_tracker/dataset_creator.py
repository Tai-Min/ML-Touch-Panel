import os
import json
import shutil
import threading
import xml.dom.minidom
import cv2
import random 

src_path = 'D:/Pobrane/ILSVRC2017_VID/ILSVRC2017_VID.tar/ILSVRC2017_VID/ILSVRC' # path to ILSVRC folder
dst_path = './dataset' # path to dataset folder
max_frames_apart = 50
exemplar_step = 15
search_image_size = 255
exemplar_image_size = 127

# Load annotation and data, center image around bounding box, fill some space with mean RGB value if necessary and resize image 
# then save it and radius of obiect's bounding box
# src_annotation_path - full path to annotation file
# src_data_path - full path to image file
# dst_video_folder_annotation_path - where to save new annotation
# dst_video_folder_path - where to save new image
# desired_name - new filename
# desired_size - size of new image
# padding_variable - bigger - less padding around image, use 4 for exemplar to make exemplars like in original whitepaper
def transform_save_image(src_annotation_path, src_data_path, dst_video_folder_annotation_path, dst_video_folder_path, desired_name, desired_size, padding_variable):
    # get info about tracked object
    annotation = xml.dom.minidom.parse(src_annotation_path)
    obj = annotation.getElementsByTagName("annotation")[0].getElementsByTagName("object")

    # empty object so ignore
    if not len(obj):
        return 0
    bndbox = obj[0].getElementsByTagName("bndbox")

    # get bounding box, height, padding and center
    bndbox = bndbox[0]
    xmax = int(bndbox.getElementsByTagName("xmax")[0].firstChild.nodeValue)
    xmin = int(bndbox.getElementsByTagName("xmin")[0].firstChild.nodeValue)
    ymax = int(bndbox.getElementsByTagName("ymax")[0].firstChild.nodeValue)
    ymin = int(bndbox.getElementsByTagName("ymin")[0].firstChild.nodeValue)
    width = xmax - xmin
    height = ymax - ymin
    padding = (width + height) / padding_variable
    xcenter = xmin + width/2
    ycenter = ymin + height/2

    # get image
    img = cv2.imread(src_data_path)

    # get mean RGB to fill empty spaces
    mean = cv2.mean(img)
    
    # add mean RGB colored padding just in case cropping area is out of image's area
    img = cv2.copyMakeBorder(img, int(padding), int(padding), int(padding), int(padding), cv2.BORDER_CONSTANT, value=mean)

    # crop image
    xcenter += padding
    ycenter += padding
    xmin += padding
    ymin += padding
    img = img[int(ycenter - height/2 - padding) : int(ycenter + height/2 + padding), int(xcenter - width/2 - padding) : int(xcenter + width/2 + padding)]

    # resize to desired size keep aspect ratio
    scaling_factor = desired_size / float(img.shape[0])
    if desired_size / float(img.shape[1]) < scaling_factor:
        scaling_factor = desired_size / float(img.shape[1])
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)

    # add padding again onto resized image
    ypadding = 0
    if img.shape[0] < desired_size:
        ypadding = (desired_size - img.shape[0])/2
    xpadding = 0
    if img.shape[1] < desired_size:
        xpadding = (desired_size - img.shape[1])/2
    img = cv2.copyMakeBorder(img, int(ypadding), int(ypadding), int(xpadding), int(xpadding), cv2.BORDER_CONSTANT, value=mean)

    # make sure it's size x size after floating point operations
    if img.shape[0] < desired_size:
        img = cv2.copyMakeBorder(img, int(desired_size - img.shape[0]), int(0), int(0), int(0), cv2.BORDER_CONSTANT, value=mean)
    if img.shape[1] < desired_size:
        img = cv2.copyMakeBorder(img, int(0), int(0), int(desired_size - img.shape[1]), int(0), cv2.BORDER_CONSTANT, value=mean)

    if img.shape[0] != desired_size or img.shape[1] != desired_size:
        return 0

    # save image
    image_file_path = os.path.join(dst_video_folder_path, desired_name + '.JPEG')
    cv2.imwrite(image_file_path, img)

    # get radius of tracked object in the image 
    xcenter *= scaling_factor
    ycenter *= scaling_factor
    xmin *= scaling_factor
    ymin *= scaling_factor
    radius = xcenter - xmin
    if ycenter - ymin > radius:
        radius = ycenter - ymin
    
    # save radius
    annotation_file_path = os.path.join(dst_video_folder_annotation_path, desired_name + '.txt')
    ifile = open(annotation_file_path, 'w')
    ifile.write(str(int(radius)))
    ifile.close()
    return radius

# Performs transformations exemplar and it's source images and save the result
# Params:
# src_video_folder_annotations_path - folder where annotation data (.xml) is stored
# src_video_folder_data_path - folder where image data (.jpg) is stored
# exemplar_filename - filename of exemplar with .xml extension
# source_image_filenames - list of source image filenames with .xml extensions
# dst_group_folder_annotation_path - where to save annotations
# dst_group_folder_data_path - where to save image data
def transform_save_exemplars_images(src_video_folder_annotations_path, src_video_folder_data_path, exemplar_filename, source_image_filenames, dst_group_folder_annotation_path, dst_group_folder_data_path):
    # no images for exemplar so remove group{n} folder
    if not len(source_image_filenames):
        shutil.rmtree(dst_group_folder_annotation_path)
        shutil.rmtree(dst_group_folder_data_path)
        return

    exemplar_annotation_path = os.path.join(src_video_folder_annotations_path, exemplar_filename)
    exemplar_data_path = os.path.join(src_video_folder_data_path, os.path.splitext(exemplar_filename)[0] + '.JPEG')

    radius = transform_save_image(exemplar_annotation_path, exemplar_data_path, dst_group_folder_annotation_path, dst_group_folder_data_path, 'exemplar', exemplar_image_size, 4)

    # exemplar invalid for some reason so remove group{n} folder
    if radius == 0:
        shutil.rmtree(dst_group_folder_annotation_path)
        shutil.rmtree(dst_group_folder_data_path)
        return

    # process every image in folder
    cntr = 0
    for image in source_image_filenames:
        image_annotation_path = os.path.join(src_video_folder_annotations_path, image)
        image_data_path = os.path.join(src_video_folder_data_path, os.path.splitext(image)[0] + '.JPEG')
        radius = transform_save_image(image_annotation_path, image_data_path, dst_group_folder_annotation_path, dst_group_folder_data_path, str(cntr), search_image_size, random.uniform(0.5, 4.1))
        if radius != 0:
            cntr+=1     

    # no valid images for exemplar so remove group{n} folder
    if cntr == 0:
        shutil.rmtree(dst_group_folder_annotation_path)
        shutil.rmtree(dst_group_folder_data_path)


# Create group that contains one exemplar and at most max_frames_apart number of source images
# Params:
# src_video_folder_annotation_path - path to video annotations that contains .xmls i.e ILSVRC/Annotations/val/VIDEO_FOLDER_NAME
# src_video_folder_data_path - path to video data folder that contains jpgs i.e ILSVRC/Data/val/VIDEO_FOLDER_NAME
# dst_video_folder_annotations_path - where to save group folder with annotations for exemplar and it's frames i.e dst_path/Annotations/dataset_type/VIDEO_FOLDER_NAME
# dst_video_folder_data_path - where to save group folder with exemplar and it's frames i.e dst_path/Data/dataset_type/VIDEO_FOLDER_NAME
def create_group(src_video_folder_annotation_path, src_video_folder_data_path, dst_video_folder_annotations_path, dst_video_folder_data_path):

    # get all frames of given video
    frames = os.listdir(src_video_folder_annotation_path)

    # split frames into batches of one exemplar and max_frames_apart source frames
    cntr = 0
    for i in range(0, len(frames), exemplar_step):
        current_exemplar = frames[i]
        current_frames = frames[i+1:i+1+max_frames_apart]

        # find next available group name
        while os.path.exists(os.path.join(dst_video_folder_annotations_path, 'group' + str(cntr))):
            cntr += 1

        # create group for current batch
        group_folder_annotation_path = os.path.join(dst_video_folder_annotations_path, 'group' + str(cntr))
        group_folder_data_path = os.path.join(dst_video_folder_data_path, 'group' + str(cntr))

        os.mkdir(group_folder_annotation_path)
        os.mkdir(group_folder_data_path)

        # transform images, annotate them and save
        transform_save_exemplars_images(src_video_folder_annotation_path, src_video_folder_data_path, current_exemplar, current_frames, group_folder_annotation_path, group_folder_data_path)

    # no groups were created so delete VIDEO_FOLDER_NAME folder
    if len(os.listdir(dst_video_folder_data_path)) == 0:
        shutil.rmtree(dst_video_folder_annotations_path)
        shutil.rmtree(dst_video_folder_data_path)

# Process given list of video folders
# Params:
# src_annotations_path - folder where video folders for annotations are stored i.e ILSVRC/Annotations/dataset_type
# src_data_path - folder where video folders for images are stored i.e ILSVRC/Data/dataset_type
# video_folder_names - list of videos folder names to process
# dst_annotations_path - where to store processed video folders with annotations i.e dst_path/Annotations/dataset_type
# dst_data_path - where to store processed video folders with images i.e dst_path/Data/dataset_type
# dataset_type - either train or val
def process_video_folders(src_annotations_path, src_data_path, video_folder_names, dst_annotations_path, dst_data_path, dataset_type):
    # for each video folder in current list
    cntr = 0
    folder_cntr = 0
    for video_folder in video_folder_names:
        folder_cntr+=1

        # append train precedensing folder name to this video folder name
        dst_video_folder = video_folder
        if dataset_type == 'train':
            dst_video_folder = os.path.basename(os.path.normpath(src_annotations_path)) + '_' + video_folder

        # some videos were processed so skip these
        if os.path.exists(os.path.join(dst_annotations_path, dst_video_folder)) or os.path.exists(os.path.join(dst_data_path, dst_video_folder)):
            print("Video folder %s exists, skipping." % dst_video_folder)
            continue

        src_video_folder_annotation_path = os.path.join(src_annotations_path, video_folder)
        src_video_folder_data_path = os.path.join(src_data_path, video_folder)
            
        dst_video_folder_annotation_path = os.path.join(dst_annotations_path, dst_video_folder)
        dst_video_folder_data_path = os.path.join(dst_data_path, dst_video_folder)

        os.mkdir(dst_video_folder_annotation_path)
        os.mkdir(dst_video_folder_data_path)

        # process video into groups that each contain one exemplar image and max_frames_apart source images
        create_group(src_video_folder_annotation_path, src_video_folder_data_path, dst_video_folder_annotation_path, dst_video_folder_data_path)
        
        print("Processed %d out of %d video folders in thread." % (folder_cntr, len(video_folder_names)))

# Process all folders that contains labels and images
# Params:
# src_annotations_path - path to folder where source annotations are stored i.e ILSVRC/Annotations/dataset_type
# src_data_path - path to folder where source data is stored i.e ILSVRC/Data/dataset_type
# dst_annotations_path - where to save annotations i.e dst_path/Annotations/dataset_type
# dst_data_path - where to save data i.e dst_path/Data/dataset_type
# dataset_type - either train or val
def process_all_video_folders(src_annotations_path, src_data_path, dst_annotations_path, dst_data_path, dataset_type):
    #print("Processing: %s" % src_data_path)

    # Split video list in n smaller lists
    # used for threading
    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)) 
    
    videos = os.listdir(src_annotations_path)

    # split folder into threads
    videos = list(split(videos, 2))

    threads = []
    for video_list in videos:
        threads.append(threading.Thread(target=thread_process, args=(src_annotations_path, src_data_path, video_list, dst_annotations_path, dst_data_path, dataset_type)))
        print("Starting thread for list of videos to process.")
        threads[-1].start()
    for thrd in threads:
        thrd.join()
        print("Thread Videos in list processed.")

# Process one dataset train or valid of ILSVRC2017_VID dataset
# Save results in dst_path/Annotations/dataset_type
# and dst_path/Data/dataset_type
# Params:
# src_annotations_path - path to folder where source annotations (or precedent folders in case of 'train' dataset type) are stored for this dataset part i.e ILSVRC/Annotations/dataset_type
# src_data_path - path to folder where source data (or precedent folders in case of 'train' dataset type) is stored for this dataset part i.e ILSVRC/Data/dataset_type
# dataset_type - either train or val
def process_dataset(src_annotations_path, src_data_path, dataset_type):
    dst_annotations_path = os.path.join(dst_path, 'Annotations')
    dst_annotations_path = os.path.join(dst_annotations_path, dataset_type)

    dst_data_path = os.path.join(dst_path, 'Data')
    dst_data_path = os.path.join(dst_data_path, dataset_type)

    # process every folder in train folder
    # in parallel
    if dataset_type == 'train':
        dirs = os.listdir(src_annotations_path)
        threads = []
        for train_folder in dirs:
            inner_src_annotations_path = os.path.join(src_annotations_path, train_folder)
            inner_src_data_path = os.path.join(src_data_path, train_folder)
            threads.append(threading.Thread(target=process_all_video_folders, args=(inner_src_annotations_path, inner_src_data_path, dst_annotations_path, dst_data_path, dataset_type)))
            print("Starting thread for %s." % train_folder)
            threads[-1].start()
        for thrd in threads:
            thrd.join()
            print("%s processed." % train_folder)

    # process every sample in valid folder
    else:
        process_all_video_folders(src_annotations_path, src_data_path, dst_annotations_path, dst_data_path, dataset_type)

# This function parses given directory tree
# which is either path to Data/train or path to Data/val
# and saves result as json file 'structure.json' for dataset's structure
# and 'weights.json' that contains weights for each video folder
def create_dataset_descriptor_files(path):
    dataset_tree = {}
    dataset_weights = {}
    max_video_frames = 0
    total_frames = 0

    # parse each video folder in dataset
    video_folder_list = os.listdir(path)
    for video_folder in video_folder_list:
        video_folder_path = os.path.join(path, video_folder)
        dataset_tree[video_folder] = {}

        video_frames = 0

        # parse each group in dataset
        for group_folder in os.listdir(video_folder_path):
            file_list = os.listdir(os.path.join(video_folder_path, group_folder))
            dataset_tree[video_folder][group_folder] = file_list

            total_frames += len(file_list)
            video_frames += len(file_list) 

        dataset_weights[video_folder] = video_frames

        if video_frames > max_video_frames:
            max_video_frames = video_frames

        print("Files parsed %d" % total_frames)

    # change number of frames into weights
    for video_folder in dataset_weights:
        dataset_weights[video_folder] = total_frames / (len(dataset_weights) * dataset_weights[video_folder]) 

    print("Total files for %s: %d" % (path, total_frames)) 

    with open(os.path.join(path, 'structure.json'), 'w') as fp:
        json.dump(dataset_tree, fp)

    with open(os.path.join(path, 'weights.json'), 'w') as fp:
        json.dump(dataset_weights, fp)

# script starts here
src_annotations_path = os.path.join(src_path, 'Annotations/VID')
src_data_path = os.path.join(src_path, 'DATA/VID')

dst_annotations_train_path = os.path.join(dst_path, 'Annotations/train')
if not os.path.exists(dst_annotations_train_path):
    os.makedirs(dst_annotations_train_path)

dst_annotations_val_path = os.path.join(dst_path, 'Annotations/val')
if not os.path.exists(dst_annotations_val_path):
    os.makedirs(dst_annotations_val_path)

dst_data_train_path = os.path.join(dst_path, 'Data/train')
if not os.path.exists(dst_data_train_path):
    os.makedirs(dst_data_train_path)

dst_data_val_path = os.path.join(dst_path, 'Data/val')
if not os.path.exists(dst_data_val_path):
    os.makedirs(dst_data_val_path)

src_annotations_train_path = os.path.join(src_annotations_path, 'train')
src_annotations_val_path = os.path.join(src_annotations_path, 'val')
src_data_train_path = os.path.join(src_data_path, 'train')
src_data_val_path = os.path.join(src_data_path, 'val')

process_dataset(src_annotations_train_path, src_data_train_path, 'train')
process_dataset(src_annotations_val_path, src_data_val_path, 'val')
create_dataset_descriptor_files(os.path.join(dst_path, 'Data/train'))
create_dataset_descriptor_files(os.path.join(dst_path, 'Data/val'))