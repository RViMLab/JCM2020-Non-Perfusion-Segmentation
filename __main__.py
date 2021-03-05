# %matplotlib inline

import os
from os.path import join
import parameters as param
from dataset import DataSet
from trainer import Trainer
from tensorflow.python.client import device_lib

from loader import inference

from util_ml import get_metrics, get_metrics_roc_fold

import argparse

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def wellcome():
    print("CNP tool by Joan M. Nunez do Rio")
#     print("Available GPUs:", get_available_gpus())
def set_visible_gpus(visible_gpus):
    print("GPU settings:", param.gpu_list)
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', default=None)
    print("CUDA_VISIBLE_DEVICES:", cuda_visible_devices)   
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus

    print("Available GPU's:", get_available_gpus())
    
def train_feeding(train_data_folder):
    print("Training (feeding)...")
    dataset = DataSet()
    load_ori, _, _, load_gth = dataset.load_pickle(train_data_folder)
       
    trainer = Trainer(param.batch_size, '')
    
    print('load_ori.shape', load_ori.shape)
    trainer.train_model_feeding(load_ori, load_gth)

    print("Training finished.")

    
def format_data(data_path, data_folder, kfolds=0):
    print("Formating data...")
    dataset = DataSet()
    input_path = os.path.join(data_path, data_folder)

    # schannel_path = os.path.join(data_path, data_folder + r'_schannel')
    # dataset.make_schannel(input_path, schannel_path)

    crop_path = os.path.join(data_path, data_folder + r'_crop')
    dataset.make_crop(input_path,#schannel_path,
                      crop_path,
                      param.crop_height,
                      param.crop_width)

    resize_path = os.path.join(data_path, data_folder + r'_resize')
    dataset.make_resize(crop_path,
                        resize_path,
                        param.resize_height,
                        param.resize_width)

    preproc_mode = param.preproc_mode
    if param.preproc_mode != '':
        preproc_folder = dataset.make_preproc(resize_path,
                                              preproc_mode,
                                              param.resize_height,
                                              param.resize_width)

        cv_path = os.path.join(data_path,
                               data_folder + r'_cv' + str(kfolds) + '_' + preproc_mode)
    else:
        preproc_folder = None
        cv_path = os.path.join(data_path,
                               data_folder + r'_cv' + str(kfolds))

    # dataset.split_data(resize_path, split_path, param.train_fraction, preproc_folder)
    dataset.crossval(resize_path, cv_path, kfolds, preproc_folder)
    # dataset.crossval(input_path, cv_path, kfolds, preproc_folder)

    print('split_path', cv_path)
    for kk in range(kfolds):
        print(kfolds, 'cross-validation. Fold ', kk)
        # TODO check when tfrec_valid is None as it meant No images available to save on TFrecord
        dataset.write_tfrecord(cv_path, 'fold' + str(kk))

    # print("OUT format_data")
    return cv_path#, tfrec_train, tfrec_valid


def format_data_cv(data_path, data_folder, kfolds=0):
    print("Formating data...")
    dataset = DataSet()
    cv_path = os.path.join(data_path,
                           data_folder + r'_cv' + str(kfolds) + '_uncropped')


    print('split_path', cv_path)
    for kk in range(kfolds):
        print(kfolds, 'cross-validation. Fold ', kk)
        # check when tfrec_valid is None as it meant No images available to save on TFrecord
        dataset.write_tfrecord(cv_path, 'fold' + str(kk))

    # print("OUT format_data")
    return cv_path#, tfrec_train, tfrec_valid


def format_topcom(data_path, data_folder):
    '''
    Formats Topcom data
    '''
    print("Formatting Topcom data...")
    dataset = DataSet()

    preproc_folder = dataset.make_preproc_kaggle(data_path, data_folder)
    
    tfrec_train = dataset.write_tfrecord(join(data_path,preproc_folder),
                                         join(data_path,data_folder),
                                         'train')
    tfrec_valid = dataset.write_tfrecord(join(data_path,preproc_folder),
                                         join(data_path,data_folder),
                                         'test')
    
    preproc_path = os.path.join(data_path, preproc_folder)

    return preproc_path, tfrec_train, tfrec_valid


def has_valid_data(data_path, tfrec_valid):
    return os.path.isfile(os.path.join(data_path, tfrec_valid))


def train(train_batch_size, train_data_folder, kcrossval):#, premodel_path=None):
    print("Training...")
    trainer = Trainer(train_batch_size,
                      train_data_folder,
                      kcrossval)
    trainer.train_model()
    print("Training finished.")


def predict(param):
    # print("Loading model from:", model_path)
    inference(param.inf_model_folder,
              param.inf_data_folder,
              param.inf_out_folder,
              param.inf_cv_folder)


def metrics(param, flag_roc):
    if not flag_roc:
        get_metrics(param.pred_folder, param.gt_folder, param.weights_folder)
    else:
        get_metrics_roc_fold(param.pred_folder, param.gt_folder, param.weights_folder, param.cv_folder)

def main():
    wellcome()
    set_visible_gpus(param.gpu_list)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='Select mode: train, inference, metrics [default: train], e.g. '
                             'python main.py --mode train')
    args = parser.parse_args()

    if args.mode == 'train':
        # Format data and train model   
        if param.preproc_path is '':
            data_path = format_data(param.data_path, param.data_folder, param.kcrossval)
            # data_path = format_data_cv(param.data_path, param.data_folder, param.kcrossval)
        else:
            data_path = param.preproc_path
            tfrec_train = 'train_data.tfrecords'
            tfrec_valid = 'test_data.tfrecords'

        train(param.train_batch_size, data_path, param.kcrossval)

    elif args.mode == 'inference':
        # Predict        
        predict(param)
    elif args.mode == 'metrics':
        # Performance
        metrics(param, True)
    
if __name__ == '__main__':
    main()
    
    
    