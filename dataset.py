import numpy as np
import os
from os.path import join
import matplotlib.image as mpimg
import scipy as sp
from scipy import ndimage

import random
from skimage import transform
from random import getrandbits
from six.moves import cPickle as pickle

import tensorflow as tf

import skimage.transform as sktr
import matplotlib.pyplot as plt
import shutil

import cv2

# Creates dataset from raw data (images or TFrecord)
class DataSet(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        
        self.data_path = None
        self.schannel_path = None
        self.augm_path = None
        self.crop_path = None
        self.resize_path = None
        
    def get_data_path(self):
        return self.__data_path 
    
    def __recursive_img_schannel(self, src, dst, ignore=None):
        '''
        Format data to single channel images (keeps first channel)
        '''
        if os.path.isdir(src):
            if not os.path.isdir(dst):
                os.makedirs(dst)
            files = os.listdir(src)
            if ignore is not None:
                ignored = ignore(src, files)
            else:
                ignored = set()
            for f in files:
                if f not in ignored:
                    self.__recursive_img_schannel(os.path.join(src, f), 
                                                  os.path.join(dst, f), 
                                                  ignore)
        else:
            try:
                image_data = (ndimage.imread(src).astype(float))
                image_dims = len(image_data.shape)
                if image_dims == 2:
                    image_gray = image_data.copy()
                elif image_dims == 3:
                    image_gray = image_data[:,:,0].copy()
    
                sp.misc.toimage(image_gray, cmin=0.0, cmax=255).save(dst[:-3] + 'png')
                #Image.fromarray(image_gray).save(dst)
            except IOError as e:
                print('Could not read:\n', src, '\nError', e, '- Skipping file.')
                
    def make_schannel(self, src, dst, ignore=None):
        print('Converting data to gray...')
        self.schannel_path = dst
        if os.path.isdir(dst):
            print('Removing gray data folder:', dst)
            shutil.rmtree(dst)
        self.__recursive_img_schannel(src, dst, ignore)
        
    def __bounding_box(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
    
        return rmin, cmin, rmax, cmax
        """ Data augmentation prior to pipeline.
            DEPRECATED: Data augmentation is now part of the pipeline"""
        
    def __extract_data_sample(self,
                            img,
                            cnp,
                            vld,
                            sample_size=(1024, 1024),
                            rotation_range=(0, 180), 
                            shear_range=(0, 30),
                            scale_range=(0.5, 1),
                            flip=(True, True),
                            gaus_sigmax=(0, 6),
                            gaus_sigmay=(0, 6)):
        """ Data augmentation prior to pipeline.
        DEPRECATED: Data augmentation is now part of the pipeline"""
        sample_height = sample_size[0]
        sample_width = sample_size[1]
        img_out = img
        cnp_out = cnp
        vld_out = vld
        img_height = img_out.shape[0]
        img_width = img_out.shape[1]
        center = np.round(np.array((img.shape[1], img.shape[0])) / 2. - 0.5)
    
        rotation_angle = random.randint(rotation_range[0], rotation_range[1])
        shear_angle = random.randint(shear_range[0], shear_range[1])
        scale_ratio = np.round(random.uniform(scale_range[0], scale_range[1]), 2)   
    
        tform1 = transform.SimilarityTransform(translation=center)
        tform2 = transform.AffineTransform( 
            rotation=np.deg2rad(rotation_angle), 
            shear=np.deg2rad(shear_angle))#scale=(scaley_ratio, scalex_ratio), translation=center) #, preserve_range=True)
        tform3 = transform.SimilarityTransform(scale=scale_ratio)
        tform4 = transform.SimilarityTransform(translation=-center)
        tform = tform4 + tform3 + tform2 + tform1 # 
    
    #     output_shape = img.shape[0]*scaley_ratio, img.shape[1]*scalex_ratio
    #     output_shape = np.round(output_shape)
        img_out = transform.warp(img, tform, preserve_range=True)#, output_shape=output_shape)
        cnp_out = transform.warp(cnp, tform, preserve_range=True)#, output_shape=output_shape)
        vld_out = transform.warp(vld, tform, preserve_range=True)#, output_shape=output_shape)
    
        # Flip X
        flipx_enabled = (not getrandbits(1))
        if flip[1] & flipx_enabled:
            img_out = img_out[:,::-1]
            cnp_out = cnp_out[:,::-1]
            vld_out = vld_out[:,::-1]
    
        # Flip Y
        flipy_enabled = (not getrandbits(1))
        if flip[0] & flipy_enabled:
            img_out = img_out[::-1,:]
            cnp_out = cnp_out[::-1,:]
            vld_out = vld_out[::-1,:]
    
        # Blurring
        sigmax = random.randint(gaus_sigmax[0], gaus_sigmax[1])
        sigmay = random.randint(gaus_sigmay[0], gaus_sigmay[1])
        img_out = ndimage.filters.gaussian_filter(img_out, (sigmay, sigmax))
    
        bbox = np.array(self.__bounding_box(vld_out))
#         bbox_ctr = (bbox[0]+bbox[2]) // 2, (bbox[1]+bbox[3]) // 2
        bbox_height = bbox[2] - bbox[0]
        bbox_width = bbox[3] - bbox[1]
        height_ratio = bbox_height / img_height
        width_ratio = bbox_width / img_width
    
        bbox_ext = (np.amax([bbox[0] - (img_height-bbox_height) * height_ratio // (4), 0]),
                    np.amax([bbox[1] - (img_width-bbox_width)   * width_ratio  // (4), 0]),
                    np.amin([bbox[2] + (img_height-bbox_height) * height_ratio // (4), img_height]),
                    np.amin([bbox[3] + (img_width-bbox_width)   * width_ratio  // (4), img_width]))
        bbox_ext = np.int32(bbox_ext)
    
        # Extract sample
        try:
            sample_y = random.randint(bbox_ext[0], bbox_ext[2]-sample_height)
            sample_x = random.randint(bbox_ext[1], bbox_ext[3]-sample_width)
        except IOError as e:
                print('Could extract sample:\n', '\nError', e, '- Skipping file.')    
    
        sample_img = img_out[sample_y:sample_y+sample_height, 
                             sample_x:sample_x+sample_width]
        sample_cnp = cnp_out[sample_y:sample_y+sample_height, 
                             sample_x:sample_x+sample_width]
        sample_vld = vld_out[sample_y:sample_y+sample_height, 
                             sample_x:sample_x+sample_width]
            
        return sample_img, sample_cnp, sample_vld    
        
    def data_augmentation(self, src, dst, ext):
        """ Data augmentation prior to pipeline.
        DEPRECATED: Data augmentation is now part of the pipeline"""
        dst_img = os.path.join(dst, 'Done')
        dst_cnp = os.path.join(dst, 'NP')
        dst_vld = os.path.join(dst, 'Valid')
        
        if not os.path.isdir(dst):   
            os.makedirs(dst_img)
            os.makedirs(dst_cnp)
            os.makedirs(dst_vld)
        files = os.listdir(os.path.join(src, 'Done'))      
        for f in files:
            try:
                src_img = os.path.join(src, 'Done', f)
                src_cnp = os.path.join(src, 'NP', f)
                src_vld = os.path.join(src, 'Valid', f)
                print(src_img)
                img = (ndimage.imread(src_img).astype(float))
                cnp = (ndimage.imread(src_cnp).astype(float))
                vld = (ndimage.imread(src_vld).astype(float))
    
                sample_img, sample_cnp, sample_vld = self.__extract_data_sample(img, cnp, vld)
    
                dst_img = os.path.join(dst, 'Done', f[:-4] + '_' + str(ext) + '.png')
                dst_cnp = os.path.join(dst, 'NP', f[:-4] + '_' + str(ext) + '.png')
                dst_vld = os.path.join(dst, 'Valid', f[:-4] + '_' + str(ext) + '.png')
                sp.misc.toimage(sample_img, cmin=0.0, cmax=255).save(dst_img)
                sp.misc.toimage(sample_cnp, cmin=0.0, cmax=255).save(dst_cnp)
                sp.misc.toimage(sample_vld, cmin=0.0, cmax=255).save(dst_vld)
            except IOError as e:
                print('Could not read:\n', f, '\nError', e, '- Skipping file.')
                
    def make_data_augm(self, src, dst, data_augm_rate):
        """ Data augmentation prior to pipeline.
        DEPRECATED: Data augmentation is now part of the pipeline"""
        print('Data augmentation...')
        self.augm_path = dst
        if os.path.isdir(dst):
            print('Removing data augmentation folder:', dst)
            shutil.rmtree(dst)
        for ii in range(data_augm_rate):
            print('Processing data folder. Iteration:', ii)
            self.data_augmentation(src, dst, ii)
                
    def __recursive_img_crop(self, src, dst, height, width, ignore=None):
        if os.path.isdir(src):
            if not os.path.isdir(dst):
                os.makedirs(dst)
            files = os.listdir(src)
            if ignore is not None:
                ignored = ignore(src, files)
            else:
                ignored = set()
            for f in files:
                if f not in ignored:
                    self.__recursive_img_crop(os.path.join(src, f), 
                                              os.path.join(dst, f), 
                                              height, 
                                              width, 
                                              ignore)
        else:
            try:
                image_data = (ndimage.imread(src).astype(float))
                image_copy = image_data.copy()
                [iheight, iwidth] = image_data.shape
                im_center_h = int(iheight/2)
                im_center_w = int(iwidth/2)
                image_copy = image_copy[
                    im_center_h-int(height/2):im_center_h+int(height/2),
                    im_center_w-int(width/2):im_center_w+int(width/2)]
    
                sp.misc.toimage(image_copy, cmin=0.0, cmax=255).save(dst)
            except IOError as e:
                print('Could not read:\n', src, '\nError', e, '- Skipping file.')
            #else:
            #    shutil.copyfile(src, dst)
            
    def make_crop(self, src, dst, height, width, ignore=None):
        print('Cropping data...')
        self.crop_path = dst
        if os.path.isdir(dst):
            print('Removing cropped data folder:', dst)
            shutil.rmtree(dst)
        self.__recursive_img_crop(src, dst, height, width, ignore)
            
    def __resize_data(self, src, dst, resize_height, resize_width, interpolation):
        '''
        Resize data array
        If one dimension size is 0 it will be computed to preserve size fraction
        '''
        print('src', src)
        if not os.path.isdir(dst):
            os.makedirs(dst)     
        files = os.listdir(src)      
        for f in files:
            try:
                file_src = os.path.join(src, f)
                file_dst = os.path.join(dst, f)
                image_data = (ndimage.imread(file_src).astype(float))
                if resize_height == 0 & resize_width == 0:
                    print('Error: At least one dimension must not be 0')
                elif resize_width == 0:
                    image_ratio = resize_height/image_data.shape[0]
                elif resize_height == 0:
                    image_ratio = resize_width/image_data.shape[1]
                else:
                    image_ratio = resize_height/image_data.shape[0]
                resized_data = np.round(sktr.resize(image_data,[resize_height, resize_width]))
   
                sp.misc.toimage(resized_data, cmin=0.0, cmax=255).save(file_dst)
            except IOError as e:
                print('Could not read:\n', file_src, '\nError', e, '- Skipping file.')
                
    def make_resize(self, src, dst, height, width):
        print('Resizing data...')
        self.resize_path = dst
        if os.path.isdir(dst):
            print('Removing resized data folder:', dst)
            shutil.rmtree(dst)
        self.__resize_data(join(src, 'Done'), 
                           join(dst, 'Done'), 
                           height, 
                           width, 
                           'bicubic')
        self.__resize_data(join(src, 'NP'), 
                           join(dst, 'NP'), 
                           height, 
                           width, 
                           'nearest')
        print('Resize valid')
        self.__resize_data(join(src, 'Valid'), 
                           join(dst, 'Valid'), 
                           height, 
                           width, 
                           'nearest')
        
    def make_preproc(self, src, mode, height, width):
        print('Preprocessing data...')
        dst_folder =  'Done_' + mode
        if not os.path.isdir(src):
            return None    
        else:
            os.makedirs(os.path.join(src, dst_folder))
            
        files = os.listdir(os.path.join(src, 'Done'))      
        for f in files:
            try:
                file_src = os.path.join(src, 'Done', f)
                file_dst = os.path.join(src, dst_folder, f)
                image_data = (ndimage.imread(file_src).astype(float))
                
                if mode=='clahe':
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    preproc_data = clahe.apply(image_data.astype(np.uint8))
                elif mode=='local_avg':
                    preproc_data = cv2.addWeighted(image_data, 4, cv2.GaussianBlur(image_data ,(0, 0), 10), -4, 128)
                elif mode=='eq':
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    preproc_data = resized_data = cv2.equalizeHist(image_data.astype(np.uint8))
                else:
                    preproc_data = image_data
   
                sp.misc.toimage(preproc_data, cmin=0.0, cmax=255).save(file_dst)
            except IOError as e:
                print('Could not read:\n', file_src, '\nError', e, '- Skipping file.')
                
        return dst_folder

    def __save_selection(self, src, dst, files, idx, preproc_folder=None):
        data_folder = 'Done'
        if preproc_folder is not None:
            data_folder = preproc_folder   
            
        img_dst_folder = join(dst, 'Done')
        cnp_dst_folder = join(dst, 'NP')
        val_dst_folder = join(dst, 'Valid')
        if not os.path.isdir(img_dst_folder):
            os.makedirs(img_dst_folder)
        if not os.path.isdir(cnp_dst_folder):
            os.makedirs(cnp_dst_folder)
        if not os.path.isdir(val_dst_folder):
            os.makedirs(val_dst_folder)
        for i in idx:
            image_name = files[i]
            img_path = join(src, data_folder, image_name)
            cnp_path = join(src, 'NP', image_name)
            val_path = join(src, 'Valid', image_name)
            try:
                img_data = ndimage.imread(img_path).astype(float)
                cnp_data = ndimage.imread(cnp_path).astype(float)
                val_data = ndimage.imread(val_path).astype(float)
                
                img_dst_file = join(img_dst_folder, image_name)
                cnp_dst_file = join(cnp_dst_folder, image_name)
                val_dst_file = join(val_dst_folder, image_name)
                
                sp.misc.toimage(img_data, cmin=0.0, cmax=255).save(img_dst_file)
                sp.misc.toimage(cnp_data, cmin=0.0, cmax=255).save(cnp_dst_file)
                sp.misc.toimage(val_data, cmin=0.0, cmax=255).save(val_dst_file)
            except IOError as e:
                print('Could not read:\n', img_path, '\nError', e, '- Skipping file.')  
        
    def split_data(self, src, dst, train_fraction, preproc_folder=None):
        if os.path.isdir(dst):
            print('Removing existing train/test data:', dst)
            shutil.rmtree(dst)
        os.makedirs(dst)
        
        data_folder = 'Done'
        if preproc_folder is not None:
            data_folder = preproc_folder        
        files = os.listdir(join(src, data_folder))
        
        ntrain = int(np.around(train_fraction*len(files)))    
        train_idx = [i for i in sorted(random.sample(range(len(files)), k=ntrain))]
        test_idx = [value for value in range(len(files)) if value not in train_idx]
        
        dst_train_folder = join(dst,'train')
        dst_test_folder = join(dst,'test')
        
        print('Creating training data...')
        self.__save_selection(src, 
                              dst_train_folder, 
                              files, 
                              train_idx,
                              data_folder)
        print('Creating testing data...')
        self.__save_selection(src, 
                              dst_test_folder, 
                              files, 
                              test_idx,
                              data_folder)


    def crossval(self, src, dst, kfolds, preproc_folder=None):
        if os.path.isdir(dst):
            print('Removing existing cross-validation data:', dst)
            shutil.rmtree(dst)
        os.makedirs(dst)

        data_folder = 'Done'
        if preproc_folder is not None:
            data_folder = preproc_folder
        files = os.listdir(join(src, data_folder))

        fold_nfiles = int(np.around(len(files)/kfolds))
        files_idx = range(len(files))

        for kk in range(kfolds):
            print(kk)
            fold_idx = [i for i in sorted(random.sample(files_idx, k=fold_nfiles))]
            files_idx = [value for value in files_idx if value not in fold_idx]

            print('Creating training data...')
            self.__save_selection(src,
                                  join(dst, 'fold' + str(kk)),
                                  files,
                                  fold_idx,
                                  data_folder)

        
    def get_ground_truth(self, cnp_data, vld_data):
        vld_img_data = vld_data>0.5
        cnp_img_data = cnp_data>0.5
        image_gt = np.zeros(vld_img_data.shape, np.float32)
        image_gt[vld_img_data==1] = 1.
        image_gt[cnp_img_data==1] = 2.
        return image_gt
    
    def __norm(self, src_folder, dst_folder, mode='data', pixel_depth=256):
        # Saves images but should save binaries
        print("norm")
        image_files = os.listdir(src_folder)
        image_files.sort()
        type(pixel_depth)
        for image_name in image_files:
            print(image_name)
            image_path = join(src_folder, image_name)
            try:
                if mode == 'data':
                    image_data = (ndimage.imread(image_path).astype(float) - 
                                  pixel_depth / 2) / pixel_depth
                elif mode == 'labels':
                    image_data = ndimage.imread(image_path).astype(float) / pixel_depth
                    image_data = image_data==1

                dst_file = join(dst_folder, image_name)
                print("dst_file", dst_file)
                sp.misc.toimage(image_data, cmin=0.0, cmax=255).save(dst_file)
            except IOError as e:
                print('Could not read:\n', image_path, '\nError', e, '- Skipping file.')
                
    def __comb_and_norm(self, src_folder, dst_folder, pixel_depth=256):
        # Saves images but should save binaries
        print("comb_and_norm")
        vld_img_files = os.listdir(join(src_folder, 'Valid'))
        vld_img_files.sort()
        type(pixel_depth)
        for image_name in vld_img_files:
            vld_img_path = join(src_folder, 'Valid', image_name)
            cnp_img_path = join(src_folder, 'NP', image_name)
            try:
                cnp_img_data = ndimage.imread(cnp_img_path).astype(float) / pixel_depth
                vld_img_data = ndimage.imread(vld_img_path).astype(float) / pixel_depth
                image_data = self.get_ground_truth(cnp_img_data, vld_img_data)
                
            except IOError as e:
                print('Could not read:\n', cnp_img_path, '\n or', vld_img_path, '\nError', e, '- Skipping file.')
                
            dst_file = join(dst_folder, 'Groundtruth', image_name)
            sp.misc.toimage(image_data, cmin=0.0, cmax=255).save(dst_file)
                    
    def normalize(self, src_folder, dst_folder):
        """ Normalizes data and saves it in image format.
        DEPRECATED"""
        # Called methods save images but should save binaries
        try:
            os.makedirs(os.path.join(dst_folder, 'Done'))
            os.makedirs(os.path.join(dst_folder, 'NP'))
            os.makedirs(os.path.join(dst_folder, 'Valid'))
            os.makedirs(os.path.join(dst_folder, 'Groundtruth')) 
        except Exception as e:
            print('Can\'t create a file if it already exists.')
            
        
        print('Normalizing data...')
        self.__norm(os.path.join(src_folder, 'Done'), 
                    os.path.join(dst_folder, 'Done'), 
                    'data')
        self.__norm(os.path.join(src_folder, 'NP'), 
                    os.path.join(dst_folder, 'NP'), 
                    'labels')
        self.__norm(os.path.join(src_folder, 'Valid'), 
                    os.path.join(dst_folder, 'Valid'), 
                    'labels')
        self.__comb_and_norm(src_folder, dst_folder)

    def __int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def __bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def normalize_sample(self, img, pixel_depth=255, weight_map=None):
        ori_data = np.float32(img)
        if weight_map is not None:
            ori_data_masked = ori_data * (weight_map / np.max(weight_map))
            ori_data_norm = (ori_data - np.mean(ori_data_masked)) / (2*pixel_depth)

        else:
            ori_data_norm = (ori_data - np.mean(ori_data)) / pixel_depth
        return ori_data_norm
    
    def write_tfrecord_DEPR(self, data_path, name, pixel_depth=1, force=True):
        """Normalizes data and converts it to tfrecords."""
        print("Saving data to TFrecord file...")
        out_file = name + '.tfrecords'
        out_path = os.path.join(data_path, out_file)
        if os.path.exists(out_path) and not force:
            print('%s already exists - Skipping conversion.' % out_path)
            return
        
        ori_files = os.listdir(os.path.join(data_path, 'Done'))
        cnp_files = os.listdir(os.path.join(data_path, 'NP'))
        vld_files = os.listdir(os.path.join(data_path, 'Valid'))
        
        if len(ori_files) != len(cnp_files) or len(ori_files) != len(vld_files):
            raise ValueError('Number of images %i does not match labels (%i, %i).' %
                             (len(ori_files), len(cnp_files), len(vld_files)))
    
        print('Writing', out_path)
        with tf.python_io.TFRecordWriter(out_path) as writer:
            for image_name in ori_files:
                ori_path = os.path.join(data_path, 'Done', image_name)
                cnp_path = os.path.join(data_path, 'NP', image_name)
                vld_path = os.path.join(data_path, 'Valid', image_name)

                ori_data = mpimg.imread(ori_path)
                cnp_data = mpimg.imread(cnp_path)
                vld_data = mpimg.imread(vld_path)
                
                height = ori_data.shape[0]
                width = ori_data.shape[1]
                
                assert(height == cnp_data.shape[0] and width == cnp_data.shape[1] and
                      height == vld_data.shape[0] and width == vld_data.shape[1]), \
                    'Image size does not match label size'
                
                ori_data = (ori_data - pixel_depth / 2) / pixel_depth
                cnp_data = np.float32(cnp_data>0.5)
                vld_data = np.float32(vld_data>0.5)
                
                gth_data = self.get_ground_truth(cnp_data, vld_data)
    
                ori_raw = ori_data.tostring()
                cnp_raw = cnp_data.tostring()
                vld_raw = vld_data.tostring()
                gth_raw = gth_data.tostring()
    
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'height': self.__int64_feature(height),
                            'width': self.__int64_feature(width),
                            'img': self.__bytes_feature(ori_raw),
                            'cnp': self.__bytes_feature(cnp_raw),
                            'vld': self.__bytes_feature(vld_raw),
                            'gth': self.__bytes_feature(gth_raw)
                    }))
                writer.write(example.SerializeToString())
    
        return out_file
    
    def write_tfrecord(self, data_folder, mode, pixel_depth=255, force=True):
        """Normalizes data and converts it to tfrecords (1 image per 1 integer label).
        data_folder: dataset folder path where train and test folders are stored
        gt_path: folder path where ground truth csv files are stored
        mode: such as 'train' or 'test'
        """
        print("Saving data to TFrecord file...")      
        data_path = join(data_folder, mode)
        out_file = mode + '_data.tfrecords'
        out_path = join(data_folder, out_file)
        if os.path.exists(out_path) and not force:
            print('%s already exists - Skipping conversion.' % out_path)
            return
        
        ori_files = os.listdir(os.path.join(data_path, 'Done'))
        cnp_files = os.listdir(os.path.join(data_path, 'NP'))
        vld_files = os.listdir(os.path.join(data_path, 'Valid'))
        print('%i images found' % len(ori_files))
        
        if len(ori_files) != len(cnp_files) or len(ori_files) != len(vld_files):
            raise ValueError('Number of images %i does not match labels (%i, %i).' %
                             (len(ori_files), len(cnp_files), len(vld_files))) 
    
        if len(ori_files) > 0:
            print('Writing', out_path)
            with tf.python_io.TFRecordWriter(out_path) as writer:
                for image_name in ori_files:
                    ori_path = os.path.join(data_path, 'Done', image_name)
                    cnp_path = os.path.join(data_path, 'NP', image_name)
                    vld_path = os.path.join(data_path, 'Valid', image_name)
                    
                    ori_data = cv2.imread(ori_path, cv2.IMREAD_GRAYSCALE)
                    cnp_data = cv2.imread(cnp_path, cv2.IMREAD_GRAYSCALE)
                    vld_data = cv2.imread(vld_path, cv2.IMREAD_GRAYSCALE)
                    
                    ori_data = self.normalize_sample(ori_data, pixel_depth, weight_map=vld_data) # TODO X
                    cnp_data = np.float32(cnp_data>pixel_depth//2)
                    vld_data = np.float32(vld_data>pixel_depth//2)

                    gth_data = self.get_ground_truth(cnp_data, vld_data)
        
                    ori_raw = ori_data.tostring()
                    cnp_raw = cnp_data.tostring()
                    vld_raw = vld_data.tostring()
                    gth_raw = gth_data.tostring()
                    
                    height = ori_data.shape[0]
                    width = ori_data.shape[1]
        
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'height': self.__int64_feature(height),
                                'width': self.__int64_feature(width),
                                'img': self.__bytes_feature(ori_raw),
                                'cnp': self.__bytes_feature(cnp_raw),
                                'vld': self.__bytes_feature(vld_raw),
                                'gt': self.__bytes_feature(gth_raw)
                        }))
                    writer.write(example.SerializeToString())
            print('Image size: %i, %i' % (height, width))
            return out_file
        else:
            print('Warning: No images available to save on TFrecord')
            return None

    def load_pickle(self, data_path):
        file_ori = os.path.join(data_path, 'Done.pickle') 
        file_cnp = os.path.join(data_path, 'NP.pickle')
        file_vld = os.path.join(data_path, 'Valid.pickle')
        file_gt = os.path.join(data_path, 'GroundTruth.pickle')
        
        print(os.path.join(data_path, 'Done.pickle') )
        
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

        