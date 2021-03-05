import os
import tensorflow as tf

import numpy as np
from skimage import transform
import random
from random import getrandbits
from scipy import ndimage

import math
import parameters as param

# Transforms a scalar string `example_proto` into a pair of a scalar string and
# a scalar integer, representing an image and its label, respectively.
def parse_and_decode(example_proto, feature_info): 
    features = dict()
    for key in feature_info.keys():
        if feature_info[key] == tf.float32:
            feat_type = tf.string
        else:
            feat_type = feature_info[key]
        features[key] = tf.FixedLenFeature([], feat_type)#tf.string)
    
    features_raw = tf.parse_single_example(example_proto, features)
    features_raw = {k:( tf.decode_raw(v, feature_info[k]) 
                       if feature_info[k] == tf.float32 else v ) 
                    for (k, v) in features_raw.items()}
#     for component in features.keys():
#         if feature_info[component] == tf.float32:
#             features_raw[component] = tf.decode_raw(features_raw[component], feature_info[component])

    return features_raw

def preprocess(features_raw, feature_info):
    features = dict()
    features = features_raw
    height, width = features['height'], features['width']
    im_shape = tf.cast(tf.stack([height, width]), tf.int64)
    
    features['img'] = tf.reshape(features['img'], im_shape)
    features = {k:( tf.reshape(v, im_shape) 
                       if feature_info[k] == tf.float32 else v ) 
                    for (k, v) in features.items()}
    
    return features

def __bounding_box(self, img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, cmin, rmax, cmax

def __extract_data_sample_nots(features,
                      sample_size=(1024, 1024),
                      rotation_range=(0, 180), 
                      shear_range=(0, 30),
                      scale_range=(0.5, 1),
                      flip=(True, True),
                      gaus_sigmax=(0, 6),
                      gaus_sigmay=(0, 6)):

    sample_height = sample_size[0]
    sample_width = sample_size[1]
#         img_out = img
#         cnp_out = cnp
#         vld_out = vld

    print('AAAAA')
    img_height = features['height']
    img_width = features['width']
    center = np.round(np.array((img_height, img_width)) / 2. - 0.5)
    
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
    
    img_out = transform.warp(features['img'], tform, preserve_range=True)#, output_shape=output_shape)
    cnp_out = transform.warp(features['cnp'], tform, preserve_range=True)#, output_shape=output_shape)
    vld_out = transform.warp(features['vld'], tform, preserve_range=True)#, output_shape=output_shape)
    gth_out = transform.warp(features['gt'], tform, preserve_range=True)#, output_shape=output_shape)
  
    # Flip X
    flipx_enabled = (not getrandbits(1))
    if flip[1] & flipx_enabled:
        img_out = img_out[:,::-1]
        cnp_out = cnp_out[:,::-1]
        vld_out = vld_out[:,::-1]
        gth_out = gth_out[:,::-1]
 
    # Flip Y
    flipy_enabled = (not getrandbits(1))
    if flip[0] & flipy_enabled:
        img_out = img_out[::-1,:]
        cnp_out = cnp_out[::-1,:]
        vld_out = vld_out[::-1,:]
        gth_out = gth_out[::-1,:]

    # Blurring
    sigmax = random.randint(gaus_sigmax[0], gaus_sigmax[1])
    sigmay = random.randint(gaus_sigmay[0], gaus_sigmay[1])
    img_out = ndimage.filters.gaussian_filter(img_out, (sigmay, sigmax))

    bbox = np.array(__bounding_box(vld_out))
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

    features['img'] = img_out[sample_y:sample_y+sample_height, 
                         sample_x:sample_x+sample_width]
    features['cnp'] = cnp_out[sample_y:sample_y+sample_height, 
                         sample_x:sample_x+sample_width]
    features['vld'] = vld_out[sample_y:sample_y+sample_height, 
                         sample_x:sample_x+sample_width]
    features['gt'] = gth_out[sample_y:sample_y+sample_height, 
                         sample_x:sample_x+sample_width]
    features['height'] = sample_height
    features['width'] = sample_width
    
    return features

def custom_transform(img, shx, shy, rot_angle, img_shape, interpolation='BILINEAR'):
    # translate, rotate, shear, translate back
    tf_zero = tf.constant(0.0, dtype=tf.float32)
    tf_one = tf.constant(1.0, dtype=tf.float32)
    tfactor = tf.constant(-0.5, dtype=tf.float32)
    tx = tf.scalar_mul(tfactor, tf.cast(img_shape[0], tf.float32))
    ty = tf.scalar_mul(tfactor, tf.cast(img_shape[1], tf.float32))
    
    tform_trans = tf.stack([[tf_one, tf_zero, tx], 
                            [tf_zero, tf_one, ty], 
                            [tf_zero, tf_zero, tf_one]], 
                           axis=0, 
                           name='stack_tform_trans')
    tf_neg_sin = tf.scalar_mul(tf.constant(-1, dtype=tf.float32), 
                               tf.sin(rot_angle))
    tform_rotate = tf.stack([[tf.cos(rot_angle), tf_neg_sin, tf_zero], 
                             [tf.sin(rot_angle), tf.cos(rot_angle), tf_zero], 
                             [tf_zero, tf_zero, tf_one]], 
                            axis=0, 
                            name='stack_tform_rot')
    tform_shear = tf.stack([[tf_one, shx, tf_zero], 
                            [shy, tf_one, tf_zero], 
                            [tf_zero, tf_zero, tf_one]], 
                           axis=0, 
                           name='stack_tform_shear')

    tfactor2 = tf.constant(0.5, dtype=tf.float32)
    tx2 = tf.scalar_mul(tfactor2, tf.cast(img_shape[0], tf.float32))
    ty2 = tf.scalar_mul(tfactor2, tf.cast(img_shape[1], tf.float32))
    tform_itrans = tf.stack([[tf_one, tf_zero, tx2], 
                             [tf_zero, tf_one, ty2], 
                             [tf_zero, tf_zero, tf_one]], 
                            axis=0, 
                            name='stack_tform_itrans')
     
    tform_rot_tr = tf.matmul(tform_rotate, tform_trans)
    tform_sh_rot_tr = tf.matmul(tform_shear, tform_rot_tr)
    tform = tf.matmul(tform_itrans, tform_sh_rot_tr) 
    tform_inv = tf.matrix_inverse(tform)
    tform_inv_flat = tf.reshape(tform_inv, [1,9])
    tform_inv_flat8, _ =  tf.split(tform_inv_flat, [8, 1], 1)
    img_out = tf.contrib.image.transform(
        img,
        tform_inv_flat8,
        interpolation=interpolation,
        name='augm_transform')
    
    return img_out

def custom_elastic_transform(img, shx, shy, rot_angle, img_shape, interpolation='BILINEAR'):
    # translate, rotate, shear, translate back
    tf_zero = tf.constant(0.0, dtype=tf.float32)
    tf_one = tf.constant(1.0, dtype=tf.float32)
    tfactor = tf.constant(-0.5, dtype=tf.float32)
    tx = tf.scalar_mul(tfactor, tf.cast(img_shape[0], tf.float32))
    ty = tf.scalar_mul(tfactor, tf.cast(img_shape[1], tf.float32))
    
    tform_trans = tf.stack([[tf_one, tf_zero, tx], 
                            [tf_zero, tf_one, ty], 
                            [tf_zero, tf_zero, tf_one]], 
                           axis=0, 
                           name='stack_tform_trans')
    tf_neg_sin = tf.scalar_mul(tf.constant(-1, dtype=tf.float32), 
                               tf.sin(rot_angle))
    tform_rotate = tf.stack([[tf.cos(rot_angle), tf_neg_sin, tf_zero], 
                             [tf.sin(rot_angle), tf.cos(rot_angle), tf_zero], 
                             [tf_zero, tf_zero, tf_one]], 
                            axis=0, 
                            name='stack_tform_rot')
    tform_shear = tf.stack([[tf_one, shx, tf_zero], 
                            [shy, tf_one, tf_zero], 
                            [tf_zero, tf_zero, tf_one]], 
                           axis=0, 
                           name='stack_tform_shear')

    tfactor2 = tf.constant(0.5, dtype=tf.float32)
    tx2 = tf.scalar_mul(tfactor2, tf.cast(img_shape[0], tf.float32))
    ty2 = tf.scalar_mul(tfactor2, tf.cast(img_shape[1], tf.float32))
    tform_itrans = tf.stack([[tf_one, tf_zero, tx2], 
                             [tf_zero, tf_one, ty2], 
                             [tf_zero, tf_zero, tf_one]], 
                            axis=0, 
                            name='stack_tform_itrans')
     
    tform_rot_tr = tf.matmul(tform_rotate, tform_trans)
    tform_sh_rot_tr = tf.matmul(tform_shear, tform_rot_tr)
    tform = tf.matmul(tform_itrans, tform_sh_rot_tr) 
    tform_inv = tf.matrix_inverse(tform)
    tform_inv_flat = tf.reshape(tform_inv, [1,9])
    tform_inv_flat8, _ =  tf.split(tform_inv_flat, [8, 1], 1)
    img_out = tf.contrib.image.transform(
        img,
        tform_inv_flat8,
        interpolation=interpolation,
        name='augm_transform')
    
    return img_out

def crop_n_resize(img, crop_size, output_size, interpolation=tf.image.ResizeMethod.BILINEAR):
    img_out = tf.random_crop(
        img,
        crop_size,
        seed=None,
        name='augm_crop')

    img_out = tf.image.resize_images(
        img_out,
        output_size,
        method=interpolation,
        align_corners=False)
      
    return img_out

def __get_sample_fa(features,
                 max_delta=0.1,
                 contrast_range=[0.8, 1.2],
                 shear_range=[-0.25, 0.25],
                 crop=[0.7, 1]):
    img_out = features['img']
    cnp_out = features['cnp']
    vld_out = features['vld']
    gth_out = features['gt']
    input_height = features['height']
    input_width = features['width']
 
    # Bring back img values to [0,1]
    cons_shift = tf.constant(0.5, dtype=tf.float32)
    img_out = tf.add(img_out, cons_shift)
     
    img_out = tf.expand_dims(img_out, axis=2)

    ##################################################################################
    # Contrast, brightness & noise
    # The higher/lower a pixel is from the mean the more its value is increased/decreased
    img_out = tf.image.random_contrast(img_out, contrast_range[0], contrast_range[1])
    img_out = tf.image.random_brightness(img_out, max_delta=max_delta)

    noise = tf.truncated_normal(shape=tf.shape(img_out),
                                mean=0.0,
                                stddev=0.005, #0.05,
                                dtype=tf.float32,
                                name='data_aug_noise')
    img_out = tf.add(img_out, noise, name='augm_add_noise')
    ##################################################################################

    img_out = tf.squeeze(img_out, axis=2)
    feat_array = tf.stack([img_out, cnp_out, vld_out, gth_out],
                         axis=2,
                         name='stack_data_feat')
 
    # Flip both axis
    feat_array = tf.image.random_flip_up_down(feat_array)
    feat_array = tf.image.random_flip_left_right(feat_array)
  
    # Rotation & shear
    rotation_angle = tf.random_uniform([], -np.pi, np.pi, dtype=tf.float32)###### 
    shx = tf.random_uniform([], 
                            minval=shear_range[0], 
                            maxval=shear_range[1], 
                            dtype=tf.float32)
    shy = tf.random_uniform([], 
                            minval=shear_range[0], 
                            maxval=shear_range[1], 
                            dtype=tf.float32)

    feat_array = custom_transform(feat_array, shx, shy, rotation_angle, [input_width, input_height], 'BILINEAR')

    # Crop & resize
    crop_ratio_x = tf.random_uniform([], minval=crop[0], maxval=crop[1], dtype=tf.float32)
    crop_ratio_y = tf.random_uniform([], minval=crop[0], maxval=crop[1], dtype=tf.float32)
    height = tf.multiply(crop_ratio_x, tf.cast(input_height, dtype=tf.float32), name='augm_multiply_h')
    width = tf.multiply(crop_ratio_y, tf.cast(input_width, dtype=tf.float32), name='augm_multiply_w')
    height = tf.round(height, name='augm_round_h')
    width = tf.round(width, name='augm_round_w')
    crop_size = tf.cast(
        tf.stack([height, width, 4], axis=0, name='augm_stack_crop_size'), 
        tf.int32)
      
    data_size = tf.cast(
        tf.stack([input_height, input_width], axis=0, name='augm_stack_crop_size'), 
        tf.int32)
    feat_array = crop_n_resize(feat_array, crop_size, data_size, tf.image.ResizeMethod.BILINEAR)
      
    img_out = tf.clip_by_value(feat_array[:,:,0], clip_value_min=0, clip_value_max=1, name='augm_clip_img')
    cnp_out = tf.clip_by_value(feat_array[:,:,1], clip_value_min=0, clip_value_max=1, name='augm_clip_cnp')
    vld_out = tf.clip_by_value(feat_array[:,:,2], clip_value_min=0, clip_value_max=1, name='augm_clip_vld')
    gth_out = tf.clip_by_value(feat_array[:,:,3], clip_value_min=0, clip_value_max=2, name='augm_clip_gth')
    
    # img values back to [-0.5, 0.5]
    img_out = tf.subtract(img_out, cons_shift)    
    cnp_out = tf.round(cnp_out, name='augm_round_cnp')
    vld_out = tf.round(vld_out, name='augm_round_vld')
    gth_out = tf.round(gth_out, name='augm_round_gth')

    features['img'] = img_out
    features['cnp'] = cnp_out
    features['vld'] = vld_out
    features['gt'] = gth_out  
    features['height'] = input_height
    features['width'] = input_width

def data_augmentation(features):
    print('data aug IN')
#     __extract_data_sample(features)
    __get_sample_fa(features)
    print('data aug OUT')
    return features

def pipeline(example_proto):
    feature_info = dict(
        height=tf.int64,
        width=tf.int64,
        img=tf.float32,
        cnp=tf.float32,
        vld=tf.float32,
        gt=tf.float32)
    
    features_raw = parse_and_decode(example_proto, feature_info)
    features = preprocess(features_raw, feature_info)

    return features
