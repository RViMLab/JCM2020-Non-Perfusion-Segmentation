import tensorflow as tf
import numpy as np

import parameters as param

from model_util import variable_summaries
    
def init_random_uniform(shape, min_max=[-0.01, 0.01]):
    # shape=[filter_height, filter_width, in_channels, out_channels]
    return tf.random_uniform(shape, min_max[0], min_max[0])
    
def init_random_normal(shape, up_unit=False):
    # shape=[filter_height, filter_width, in_channels, out_channels]
    input_dim = np.where(up_unit, shape[3], shape[2])
    return tf.random_normal(shape, 0, np.sqrt(2/(shape[0]*shape[1]*input_dim)))

def down_unit(input_tensor, input_dim, output_dim, name, init_w=tf.random_uniform):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w1_1 = tf.get_variable("w1", 
                               initializer=init_w([3,3,input_dim,output_dim], -0.04, 0.04), #0,0),#
                               dtype=tf.float32)
        b1_1 = tf.get_variable("b1", 
                               initializer=tf.zeros(output_dim), 
                               dtype=tf.float32)
        conv1_1 = tf.nn.conv2d(input_tensor, w1_1, strides=[1,1,1,1], padding='SAME') + b1_1
        r1_1 = tf.nn.relu(conv1_1, name='r1')
        w1_2 = tf.get_variable("w2", 
                               initializer=init_w([3,3,output_dim,output_dim], -0.04, 0.04),
                                   dtype=tf.float32)
        b1_2 = tf.get_variable("b2", 
                               initializer=tf.zeros(output_dim), 
                               dtype=tf.float32)
        conv1_2 = tf.nn.conv2d(r1_1, w1_2, strides=[1,1,1,1], padding='SAME') + b1_2
        r1_2 = tf.nn.relu(conv1_2, name='r2')
        
        variable_summaries('w1', w1_1)
        variable_summaries('b1', b1_1)
        
        return r1_2
    
def up_unit(input_tensor, concat_tensor, input_dim, output_dim, name, init_w=tf.random_uniform):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w1_1 = tf.get_variable("w1", 
                               initializer=init_w(
                                   [2,2,output_dim,input_dim], -0.04, 0.04), #[height, width, out_channels, in_channels]
                               dtype=tf.float32) 
        b1_1 = tf.get_variable("b1", 
                               initializer=tf.zeros(output_dim), 
                               dtype=tf.float32)
        output_shape = [tf.shape(input_tensor)[0], 
                        2*tf.shape(input_tensor)[1], 
                        2*tf.shape(input_tensor)[2], 
                        output_dim]
        upconv = tf.nn.conv2d_transpose(input_tensor, 
                                         w1_1, 
                                         output_shape=output_shape, 
                                         strides=[1,2,2,1], 
                                         padding='SAME') + b1_1
        concat = tf.concat([concat_tensor, upconv],-1, name='concat_upconv_' + name)
        w1_2 = tf.get_variable("w2", 
                               initializer=init_w([3,3,input_dim,output_dim], -0.04, 0.04), 
                               dtype=tf.float32)
        b1_2 = tf.get_variable("b2", 
                               initializer=tf.zeros(output_dim), 
                               dtype=tf.float32)
        conv1_1 = tf.nn.conv2d(concat, w1_2, strides=[1,1,1,1], padding='SAME') + b1_2
        r1_1 = tf.nn.relu(conv1_1, name='r1')
        w1_3 = tf.get_variable("w3", 
                               initializer=init_w([3,3,output_dim,output_dim], -0.04, 0.04), 
                               dtype=tf.float32)
        b1_3 = tf.get_variable("b3", 
                               initializer=tf.zeros(output_dim), 
                               dtype=tf.float32)
        conv1_2 = tf.nn.conv2d(r1_1, w1_3, strides=[1,1,1,1], padding='SAME') + b1_3
        r1_2 = tf.nn.relu(conv1_2, name='r2')
        
        return r1_2

def output_unit(input_tensor, input_dim, output_dim, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w10_1 = tf.get_variable("w1", [1,1,input_dim,output_dim], dtype=tf.float32)
        conv10_1 = tf.nn.conv2d(input_tensor, w10_1, strides=[1,1,1,1], padding='SAME')
        
        variable_summaries('w1', w10_1)
        
        return conv10_1

def build_unet(x, unit_name, nclasses=3, dropout_keep=1, is_training=False): # TODO is_training not used
    tf.set_random_seed(1234)
    x_expand = tf.expand_dims(x, 3)
    r1_2 = down_unit(x_expand, 1, 64, unit_name + '1')
    pool1 = tf.nn.max_pool(
        r1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=unit_name + '_pool1')
    r2_2 = down_unit(pool1, 64, 128,  unit_name + '2')
    pool2 = tf.nn.max_pool(
        r2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=unit_name + '_pool2')
    r3_2 = down_unit(pool2, 128, 256,  unit_name + '3')
    pool3 = tf.nn.max_pool(
        r3_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=unit_name + '_pool3')
    r4_2 = down_unit(pool3, 256, 512,  unit_name + '4')
    pool4 = tf.nn.max_pool(
        r4_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=unit_name + '_pool4')
    r5_2 = down_unit(pool4, 512, 1024,  unit_name + '5')
    
    drop1 = tf.nn.dropout(r5_2, dropout_keep, name='drop1')
    
    r6_2 = up_unit(drop1, r4_2, 1024, 512, unit_name + '6')
    r7_2 = up_unit(r6_2, r3_2, 512, 256, unit_name + '7')
    r8_2 = up_unit(r7_2, r2_2, 256, 128, unit_name + '8')
    r9_2 = up_unit(r8_2, r1_2, 128, 64, unit_name + '9')
    conv10_1 = output_unit(r9_2, 64, nclasses, unit_name + '10')
    
    return conv10_1


def build_recurrent_unet(x, x_prev, unit_name, nclasses=3, dropout_keep=1):
    tf.set_random_seed(1234)
    x_expand = tf.expand_dims(x, 3)
    r1_2 = down_unit(x_expand, 1, 64, unit_name + '1')
    pool1 = tf.nn.max_pool(
        r1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=unit_name + '_pool1')
    r2_2 = down_unit(pool1, 64, 128, unit_name + '2')
    pool2 = tf.nn.max_pool(
        r2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=unit_name + '_pool2')
    r3_2 = down_unit(pool2, 128, 256, unit_name + '3')
    pool3 = tf.nn.max_pool(
        r3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=unit_name + '_pool3')
    r4_2 = down_unit(pool3, 256, 512, unit_name + '4')
    pool4 = tf.nn.max_pool(
        r4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=unit_name + '_pool4')
    r5_2 = down_unit(pool4, 512, 1024, unit_name + '5')

    drop1 = tf.nn.dropout(r5_2, dropout_keep, name='drop1')

    r6_2 = up_unit(drop1, r4_2, 1024, 512, unit_name + '6')
    r7_2 = up_unit(r6_2, r3_2, 512, 256, unit_name + '7')
    r8_2 = up_unit(r7_2, r2_2, 256, 128, unit_name + '8')
    r9_2 = up_unit(r8_2, r1_2, 128, 64, unit_name + '9')
    conv10_1 = output_unit(r9_2, 64, nclasses, unit_name + '10')

    return conv10_1


def down_unit_bnorm(input_tensor, input_dim, output_dim, name, init_w=tf.random_uniform):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w1_1 = tf.get_variable("w1",
                               initializer=init_w([3, 3, input_dim, output_dim], -0.04, 0.04),  # 0,0),#
                               dtype=tf.float32)
        b1_1 = tf.get_variable("b1",
                               initializer=tf.zeros(output_dim),
                               dtype=tf.float32)
        conv1_1 = tf.nn.conv2d(input_tensor, w1_1, strides=[1, 1, 1, 1], padding='SAME') + b1_1
        r1_1 = tf.nn.relu(conv1_1, name='r1')
        w1_2 = tf.get_variable("w2",
                               initializer=init_w([3, 3, output_dim, output_dim], -0.04, 0.04),
                               dtype=tf.float32)
        b1_2 = tf.get_variable("b2",
                               initializer=tf.zeros(output_dim),
                               dtype=tf.float32)
        conv1_2 = tf.nn.conv2d(r1_1, w1_2, strides=[1, 1, 1, 1], padding='SAME') + b1_2
        r1_2 = tf.nn.relu(conv1_2, name='r2')

        variable_summaries('w1', w1_1)
        variable_summaries('b1', b1_1)
        #         variable_summaries('w2', w1_2)
        #         variable_summaries('b2', b1_2)

        #         print(conv1_1.op.name)
        #         print(conv1_1.name)

        return r1_2

def build_unet_bnorm(x, unit_name, nclasses=3, dropout_keep=1):
    tf.set_random_seed(1234)
    x_expand = tf.expand_dims(x, 3)
    r1_2 = down_unit_bnorm(x_expand, 1, 64, unit_name + '1')
    pool1 = tf.nn.max_pool(
        r1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=unit_name + '_pool1')
    r2_2 = down_unit_bnorm(pool1, 64, 128, unit_name + '2')
    pool2 = tf.nn.max_pool(
        r2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=unit_name + '_pool2')
    r3_2 = down_unit_bnorm(pool2, 128, 256, unit_name + '3')
    pool3 = tf.nn.max_pool(
        r3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=unit_name + '_pool3')
    r4_2 = down_unit_bnorm(pool3, 256, 512, unit_name + '4')
    pool4 = tf.nn.max_pool(
        r4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=unit_name + '_pool4')
    r5_2 = down_unit_bnorm(pool4, 512, 1024, unit_name + '5')

    drop1 = tf.nn.dropout(r5_2, dropout_keep, name='drop1')

    r6_2 = up_unit(drop1, r4_2, 1024, 512, unit_name + '6')
    r7_2 = up_unit(r6_2, r3_2, 512, 256, unit_name + '7')
    r8_2 = up_unit(r7_2, r2_2, 256, 128, unit_name + '8')
    r9_2 = up_unit(r8_2, r1_2, 128, 64, unit_name + '9')
    conv10_1 = output_unit(r9_2, 64, nclasses, unit_name + '10')

    return conv10_1

def generalised_dice_loss(prediction, ground_truth, weight_map=None):
    # prediction: (batch_size, height, width, nchannels)
    # ground_truth: (batch_size, height, width)
    # weight_map: (batch_size, height, width), binary mask including regions to consider

    dice_nclasses = prediction.shape[3].value # value
    ground_truth = tf.cast(ground_truth, dtype=tf.int64, name='dice_cast_ground_truth')
    hot_labels = tf.one_hot(ground_truth, axis=-1, depth=dice_nclasses, name='dice_hot_labels')
    if weight_map is not None:
        weight_map_tile = tf.stack([weight_map, weight_map], axis=3)
        if dice_nclasses>2:
            for dim in range(dice_nclasses-2):
                weight_map_tile = tf.concat(
                    [weight_map_tile, tf.expand_dims(weight_map, 3)], axis=3, name='dice_concat_weight_map')
#                         hot_labels = tf.one_hot(ground_truth, axis=-1, depth=dice_nclasses, name='dice_hot_labels')
        hot_labels = hot_labels * weight_map_tile
        prediction = prediction * weight_map_tile#tf.cast(hot_labels, dtype=tf.float32, name='dice_cast_hot_labels')

    sum_labels = tf.reduce_sum(hot_labels, axis=(1,2), name='dice_sum_labels')
    weights = tf.reciprocal(tf.square(tf.add(sum_labels, 1e-6)), name='dice_weights')
    den_part = tf.add(prediction, hot_labels, name='dice_den_part')
    num_part = tf.multiply(prediction, hot_labels, name='dice_num_part')
    den_part_sum = tf.reduce_sum(den_part, axis=(1,2), name='dice_den_part_sum')
    num_part_sum = tf.reduce_sum(num_part, axis=(1,2), name='dice_num_part_sum')
    gdl_den = tf.reduce_sum(tf.add(tf.multiply(weights, den_part_sum), 1e-6, name='dice_add'), axis=1)
    gdl_num = tf.reduce_sum(tf.multiply(weights, num_part_sum), axis=1, name='dice_gdl_num')
    real_div = tf.realdiv(gdl_num ,gdl_den)
    gdl_compl = tf.scalar_mul(2, real_div)
    gdl_array = tf.subtract(1., gdl_compl)
    gdl = tf.reduce_mean(gdl_array)
    
    return gdl


