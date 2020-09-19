import os
import numpy as np

def load_label(label_path, label_index):
    files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    print('%s samples found in %s' % (len(files), label_path))

    # in case of empty labels folder
    if not len(files):
        return {'x_data' : np.zeros(0), 'y_data' : np.zeros(0)}

    # get shape of input numpy array
    file_path = os.path.join(label_path, files[0])
    sample = np.loadtxt(file_path)
    input_shape = ((len(files),) + sample.shape,)
    print('Input shape: %s' % (input_shape,))

    x_data = np.zeros(*input_shape, dtype='float32')
    y_data = np.zeros(len(files), dtype='u1')
    for i, file in zip(range(len(files)), files):
        file_path = os.path.join(label_path, file)
        sample = np.loadtxt(file_path)
        x_data[i] = sample
        y_data[i] = label_index
    
    return {'x_data' : x_data, 'y_data' : y_data}

def load_dataset(dataset_path):
    print('Dataset to load: %s' % dataset_path)

    labels = os.listdir(dataset_path)
    print('Labels found: %s' % labels)

    dataset_label_dict = {}
    first_data = True

    for i, label in zip(range(len(labels)), labels):
        label_path = os.path.join(dataset_path, label)
        data = load_label(label_path, i)

        if(first_data):
            first_data = False
            dataset_x = data['x_data']
            dataset_y = data['y_data']
        else:
            dataset_x = np.concatenate([dataset_x, data['x_data']])
            dataset_y = np.concatenate([dataset_y, data['y_data']])
        dataset_label_dict[i] = label
    return {'features' : dataset_x, 'labels' : dataset_y, 'labels_str' : dataset_label_dict}
