import os
import numpy as np
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

def load_data(data_path_resize):
    # Load data
    file_ori = os.path.join(data_path_resize, 'Done.pickle') 
    file_cnp = os.path.join(data_path_resize, 'NP.pickle')
    file_vld = os.path.join(data_path_resize, 'Valid.pickle')
    file_gt = os.path.join(data_path_resize, 'GroundTruth.pickle')
    
    with open(file_ori, 'rb') as f:
        load_ori = pickle.load(f)
    with open(file_cnp, 'rb') as f:
        load_cnp = pickle.load(f)
    with open(file_vld, 'rb') as f:
        load_vld = pickle.load(f)
    with open(file_gt, 'rb') as f:
        load_gt = pickle.load(f)
        
    print("Loaded data:")
    print("Image data size:", load_ori.shape)
    print("CNP data size:", load_cnp.shape, ' - Labels:', np.unique(load_cnp))
    print("Valid data size:", load_vld.shape, ' - Labels:', np.unique(load_vld))
    print("GT data size:", load_gt.shape, ' - Labels:', np.unique(load_gt))
    
    try:   
        assert(load_ori.shape == load_cnp.shape and
              load_ori.shape == load_vld.shape and
              load_ori.shape == load_gt.shape)
    except Exception as e:
        print('Error: Unconsistent data')
        
#     image_height = load_ori.shape[1] 
#     image_width = load_ori.shape[2]
    
    # ppp = np.array([xx for xx in load_vld if xx.sum()>0], np.float32)
    indexes = [ii for ii, sample in enumerate(load_vld) if sample.sum()>0]
    load_ori = np.array(load_ori[indexes])
    load_cnp = np.array(load_cnp[indexes])
    load_vld = np.array(load_vld[indexes])
    load_gt = np.array(load_gt[indexes])
    
    print("Selected data:")
    print("Image data size:", load_ori.shape)
    print("CNP data size:", load_cnp.shape, ' - Labels:', np.unique(load_cnp))
    print("Valid data size:", load_vld.shape, ' - Labels:', np.unique(load_vld))
    print("GT data size:", load_gt.shape, ' - Labels:', np.unique(load_gt))
    
    return load_ori, load_cnp, load_vld, load_gt

def show_sample(data, index):
    imgplot = plt.imshow(data[index,:,:])
    plt.show()
    
def build_datasets(data, labels, train_fraction, valid_fraction):
    stop_train = np.int32(data.shape[0]*train_fraction)
    stop_valid = np.int32(data.shape[0]*(train_fraction+valid_fraction))
    train_data = data[0:stop_train, :, :]
    valid_data = data[stop_train:stop_valid, :, :]
    test_data = data[stop_valid:-1, :, :]
    train_labels = labels[0:stop_train, :, :]
    valid_labels = labels[stop_train:stop_valid, :, :]
    test_labels = labels[stop_valid:-1, :, :]
    return train_data, valid_data, test_data, train_labels, valid_labels, test_labels    
    
def split_data(data, labels, train_fraction, valid_fraction):#, test_fraction=0):
    # Create train/test data sets
#     train_fraction = .999
#     valid_fraction = .0005
    
    # TODO: Randomize sample selection
#     stop_train = np.int32(data.shape[0]*train_fraction)
#     stop_valid = np.int32(data.shape[0]*(train_fraction+valid_fraction))
#     train_data = data[0:stop_train, :, :]
#     valid_data = data[stop_train:stop_valid, :, :]
#     test_data = data[stop_valid:-1, :, :]
            
    # train_data, valid_data, test_data, train_labels, valid_labels, test_labels = build_datasets(
    #     load_ori, load_gt, train_fraction, valid_fraction)    
    train_data, valid_data, test_data, train_labels, valid_labels, test_labels = build_datasets(
        data, labels, train_fraction, valid_fraction)   
    print("Train data:", train_data.shape)
    print("Validation data:", valid_data.shape)
    print("Test data:", test_data.shape)
    print("Train labels:", train_labels.shape)
    print("Validation labels:", valid_labels.shape)
    print("Test labels:", test_labels.shape)    
    
    return train_data, valid_data, test_data, train_labels, valid_labels, test_labels
    