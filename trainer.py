

import tensorflow as tf
from tensorflow.python.client import device_lib
# from show_graph import show_graph
import math
import numpy as np

import matplotlib.pyplot as plt

import parameters as param
from model_util import trainable_model, gradient_summaries
import os

from input_pipeline import pipeline, data_augmentation
# from model_unet2 import build_unet, generalised_dice_loss, build_nonlocal_unet
from model_unet import build_unet, generalised_dice_loss, build_nonlocal_unet
from loader import load_model

import cv2
import scipy as sp

from iUNET_define import *

class Trainer(object):
    '''
    classdocs
    '''

    # def __init__(self, network, train_batch_size, train_path, valid_batch_size=None, valid_path=None, test_path=None):
    def __init__(self, network, train_batch_size, train_path, nkfolds=0):#, premodel_folder=None):
        '''
        Constructor
        '''
        self.train_batch_size = train_batch_size
        # self.valid_batch_size = valid_batch_size
        self.train_path = train_path  # TODO it will be data_path
        # self.valid_path = valid_path
        self.nkfolds = nkfolds
        self.filename = 'fold'
        # self.test_path = test_path
        # self.model_path = model_path
        
        # self.network = network # TODO check, commented as never used
        
        print('self.train_path', self.train_path)
        # TODO check train_nelem and valid_nelem
        # for kk in tf.python_io.tf_record_iterator(self.train_path):
        #     print(kk)
        
        # self.train_nelem = sum(1 for _ in tf.python_io.tf_record_iterator(self.train_path))
        # if self.valid_batch_size is None:
        #     self.valid_nelem = 0
        # else:
        #     self.valid_nelem = sum(1 for _ in tf.python_io.tf_record_iterator(self.valid_path))
        self.valid_nelem = 0 # TODO
        
        # set up TF environment
#         gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=param.visible_device_list)
#         self.config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

#         self.config.gpu_options.allow_growth = True
#         self.config.gpu_options.visible_device_list = param.visible_device_list        
        
#         os.environ['CUDA_VISIBLE_DEVICES'] = param.gpu_list
#         self.config = tf.ConfigProto(allow_soft_placement=True)
        
#         gpu_options = tf.GPUOptions(allow_growth=True)
#         self.config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
#         self.config = tf.ConfigProto(device_count={'GPU': param.gpu_list})

        # self.premodel_folder = premodel_folder
        
    def get_available_gpus(self):
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
    
    def plot_weights(self, units):
        filters = units.shape[3]
        plt.figure(1, figsize=(20,20))
        n_columns = 6
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            plt.title('Filter ' + str(i))
            plt.imshow(units[:,:,0,i], interpolation="nearest")#, cmap="gray")
        plt.show()
    
    def plotNNFilter(self, units):
        filters = units.shape[3]
        plt.figure(1, figsize=(20,20))
        n_columns = 6
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            plt.title('Filter ' + str(i))
            plt.imshow(units[0,:,:,i], interpolation="nearest")#, cmap="gray")
        plt.show()
    
#     def getActivations(self, sess, layer,stimuli):
#         imgplot = plt.imshow(np.squeeze(stimuli))
#         plt.show()
#         
#         activations = sess.run(layer,feed_dict={tf_train_data: stimuli})
#     #     imgplot = plt.imshow(np.squeeze(units[0,:,:,1]))
#     #     plt.show()
#         
#         self.plotNNFilter(activations)
#         return units

    def add_eval_step_BU(self, tf_prediction, tf_ground_truth):
#         with tf.name_scope('accuracy'):
#             with tf.name_scope('correct_prediction'):
        nclasses = tf_prediction.get_shape()
        pred_argmax = tf.argmax(tf_prediction, axis=3, name='pred_argmax')
        correct_prediction = tf.equal(pred_argmax, 
                                      tf.cast(tf_ground_truth, tf.int64))
#             with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy
    
    def add_eval_step(self, tf_prediction, tf_ground_truth):
#         with tf.name_scope('accuracy'):
#             with tf.name_scope('correct_prediction'):
        pred_argmax = tf.argmax(tf_prediction, axis=3, name='pred_argmax')
        correct_prediction = tf.equal(pred_argmax, 
                                      tf.cast(tf_ground_truth, tf.int64))
#             with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy
    
    def add_metrics_NO(self, tf_prediction, tf_ground_truth): # NOTEICE ORDER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Works ONLY for 2 classes
        pred_argmax = tf.argmax(tf_prediction, axis=3, name='pred_argmax')
        
        precision2 = tf.metrics.precision(tf_ground_truth, pred_argmax)
        recall2 = tf.metrics.recall(tf_ground_truth, pred_argmax)
        
        # when collecting both returned values then tf tn fp fn are always zero
        true_pos, tp_update = tf.metrics.true_positives(tf_ground_truth, pred_argmax, name='false_pos')
        true_neg, tn_update = tf.metrics.true_negatives(tf_ground_truth, pred_argmax, name='true_neg')
        false_pos, fp_update = tf.metrics.false_positives(tf_ground_truth, pred_argmax, name='false_pos')
        false_neg, fn_update = tf.metrics.false_negatives(tf_ground_truth, pred_argmax, name='false_neg')
        
        precision0 = tf.realdiv(true_pos, tf.add(true_pos, false_pos))
        recall0 = tf.realdiv(true_pos, tf.add(true_pos, false_neg))
        specificity0 = tf.realdiv(false_pos, tf.add(false_pos, true_neg))
        
#         precision1 = tf.realdiv(true_pos, tf.add(true_pos, false_neg))
#         recall1 = tf.realdiv(true_pos, tf.add(true_pos, false_pos))
#         specificity1 = tf.realdiv(false_pos, tf.add(false_neg, true_neg))
        
#         precision = tf.stack([precision0, precision1], 0)
#         recall = tf.stack([recall0, recall1], 0)
#         specificity = tf.stack([specificity0, specificity1], 0)
        precision = precision0
        recall = recall0
        specificity = specificity0
        
#         accuracy = tf.realdiv(false_positives, tf.add(false_positives, true_negatives))
        correct_prediction = tf.equal(pred_argmax, 
                                      tf.cast(tf_ground_truth, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
#         tf.summary.scalar('precision', precision )
        
        return precision, recall, specificity, accuracy, true_pos, true_neg, false_pos, false_neg, precision2, recall2, pred_argmax
    
#     def add_metrics(self, tf_prediction, tf_ground_truth, weights=None): # NOTICE ORDER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         pred_argmax = tf.argmax(tf_prediction, axis=3, name='pred_argmax')
#         
#         precision, prec_op = tf.metrics.precision(tf_ground_truth, pred_argmax, weights, name='precision')
#         recall, rec_op = tf.metrics.recall(tf_ground_truth, pred_argmax, weights, name='recall')
#         accuracy, acc_op = tf.metrics.accuracy(tf_ground_truth, pred_argmax, weights, name='accuracy')
#         
# #         correct_prediction = tf.equal(pred_argmax, 
# #                                       tf.cast(tf_ground_truth, tf.int64))
# #         accuracy2 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         
# #         tf.summary.scalar('accuracy', accuracy)
#         
#         scope_suffix = 'weighted' if weights is not None else '' 
# #         if weights is not None:
# #             scope_suffix = 'weighted'
# #         else:
# #             scope_suffix = ''
#         
#         with tf.name_scope('metrics_' + scope_suffix):#, reuse=tf.AUTO_REUSE):
#             tf.summary.scalar('accuracy', accuracy)
#             tf.summary.scalar('precision', precision)
#             tf.summary.scalar('recall', recall)
#         
#         return prec_op, rec_op, acc_op, pred_argmax#precision, recall, accuracy, pred_argmax#, accuracy2
#     
    def add_metrics(self, tf_prediction, tf_ground_truth, weights=None, threshold=0.5): # NOTICE ORDER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 2 class metrics

        #############################################################
        # if tf.rank(tf_prediction) is 4: # iunet
        # if tf_prediction.shape[4] is not None:  # iunet
        if param.network == 'iunet':  # iunet
            print('IF tf_prediction', tf_prediction.shape)
            # tf_prediction = tf.slice(tf_prediction, [2, 0, 0, 0, 0], [1, -1, -1, -1, -1])
            # tf_prediction = tf_prediction[2, :, :, :, :]

            tf_prediction0 = tf_prediction[-1, :, :, :, :] < threshold
            tf_prediction1 = tf_prediction[-1, :, :, :, :] > threshold
            tf_prediction_cc = tf.concat([tf_prediction0, tf_prediction1], 3)
            print('tf_prediction_cc', tf_prediction_cc.shape)
            # tf_prediction = tf.concat([tf_prediction, tf.ones(tf_prediction.get_shape()) - tf_prediction], 3)
            tf_prediction_cc = tf.cast(tf_prediction_cc, dtype=tf.float32)
            pred_argmax = tf.cast(
                tf.argmax(tf_prediction_cc, axis=3, name='pred_argmax'),
                dtype=tf.float32)
        else:
            pred_argmax = tf.cast(
                tf.argmax(tf_prediction, axis=3, name='pred_argmax'),
                dtype=tf.float32)
        # print('pred_argmax', pred_argmax.shape)
        # pred_argmax = tf.cast(
        #     tf.argmax(tf_prediction, axis=3, name='pred_argmax'),
        #     dtype=tf.float32)
        #############################################################

        true_pos = tf.multiply(pred_argmax, tf_ground_truth)
        true_neg = tf.multiply(pred_argmax - 1, tf_ground_truth - 1)
        false_pos = tf.multiply(pred_argmax, tf_ground_truth - 1)
        false_neg = tf.multiply(pred_argmax - 1, tf_ground_truth)
        
        scope_suffix = ''
        if weights is not None:
            scope_suffix = '_weighted'
            true_pos = tf.multiply(true_pos, weights)
            true_neg = tf.multiply(true_neg, weights)
            false_pos = tf.multiply(false_pos, weights)
            false_neg = tf.multiply(false_neg, weights)

        true_pos = tf.count_nonzero(true_pos, dtype=tf.float32, name='true_pos')
        true_neg = tf.count_nonzero(true_neg, dtype=tf.float32, name='true_neg')
        false_pos = tf.count_nonzero(false_pos, dtype=tf.float32, name='false_pos')
        false_neg = tf.count_nonzero(false_neg, dtype=tf.float32, name='false_neg')
        
        precison_den = tf.add(true_pos, false_pos, name='precison_den')
        recall_den = tf.add(true_pos, false_neg, name='recall_den')
        specificity_den = tf.add(false_pos, true_neg, name='specificity_den')
        
        precision = tf.realdiv(true_pos, tf.add(precison_den, 1e-6), name='precision')
        recall = tf.realdiv(true_pos, tf.add(recall_den, 1e-6), name='recall')
        specificity = tf.realdiv(false_pos, tf.add(specificity_den, 1e-6), name='specificity')
        
        f1_den = tf.add(precision, recall, name='f1_den')
        f1 = tf.realdiv(2*precision*recall, tf.add(f1_den, 1e-6), name='f1')
        
        acc_true = tf.add(true_pos, true_neg)
        acc_false = tf.add(false_pos, false_neg)
        accuracy = tf.realdiv(acc_true, tf.add(acc_true, acc_false)) 
        
        with tf.name_scope('metrics' + scope_suffix):#, reuse=tf.AUTO_REUSE):
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('precision', precision)
            tf.summary.scalar('recall', recall)
            tf.summary.scalar('specificity', specificity)
            tf.summary.scalar('f1', f1)
        
        # return precision, recall, specificity, f1, accuracy, pred_argmax#, accuracy2

        return true_pos, true_neg, false_pos, false_neg, precision, recall, specificity, f1, accuracy, pred_argmax
    
    def have_val_data(self):
        return self.valid_nelem > 0
    
    def get_data_batch_DEPR(self, next_element, nclasses):
        if nclasses == 2:
            return next_element['img'], next_element['cnp']
        elif nclasses == 3:
            return next_element['img'], next_element['gt']
        
    def get_data_batch(self, next_element, mode, nclasses):
        # returns image, ground truth, weight_map
        if mode == 'cnp' and nclasses == 2:
            return next_element['img'], next_element['cnp']
        elif mode == 'cnp' and nclasses == 3:
            return next_element['img'], next_element['gt']
        elif mode == 'vld' and nclasses == 2:
            return next_element['img'], next_element['vld']
        
    def get_data_weight_map(self, next_element, mode):
        if mode == 'vld':
            return next_element['vld']
        else:
            default_weight_map = tf.ones(
                tf.shape(next_element['img']), 
                dtype=tf.float32, 
                name='default_weight_map')
            return default_weight_map

    def get_data_paths(self, kfold):
        train_paths = [os.path.join(self.train_path, self.filename + str(ii) + '_data.tfrecords')
                     for ii in range(self.nkfolds) if ii is not kfold]
        test_paths = os.path.join(self.train_path, self.filename + str(kfold) + '_data.tfrecords')
        return train_paths, test_paths

    def get_tfrecord_size(self, datapaths):
        counter = tf.data.TFRecordDataset(datapaths)
        counter = counter.map(pipeline)
        counter = counter.repeat(1)
        counter = counter.batch(1)
        counter_it = counter.make_one_shot_iterator()
        cnt = 0
        with tf.Session() as ss:
            try:
                while True:
                    kk = ss.run(counter_it.get_next())
                    cnt += 1
            except tf.errors.OutOfRangeError:
                pass

        return cnt

    # def pretrained_model_exists(self):
    #     return self.model_path is not ''

    def print_names_in_graph(self):
        nodelist = [n.name for n in tf.get_default_graph().as_graph_def().node]
        for key in sorted(nodelist):
            print("tensor_name: ", key)

    def train_model(self):
        # for kfold in range(self.nkfolds): ################################################################
        #     self.train_model_fold(kfold)  ################################################################
        self.train_model_fold(param.nfold)

    def train_model_fold(self, kfold):
        with tf.Graph().as_default() as graph:
            # with tf.Session(config=self.config) as sess:
            with tf.Session() as sess:
                with tf.variable_scope('') as scope:
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    print("Training data path:", self.train_path)
                    print("Training fold:", kfold)
                    # print("Validation data path:", self.test_path)
                    # print("Test data path:", self.test_path)
                    # Define training and validation datasets with the same structure.
                    datapaths_train, datapaths_test = self.get_data_paths(kfold)
                    train_nelem = self.get_tfrecord_size(datapaths_train)
                    test_nelem = self.get_tfrecord_size(datapaths_test)
                    # train_nelem = sum(1 for _ in tf.python_io.tf_record_iterator(datapaths_train))

                    # train_dataset = tf.data.TFRecordDataset(self.train_path)#r'.\train.tfrecords')
                    train_dataset = tf.data.TFRecordDataset(datapaths_train)  # filename example: fold2_data.tfrecords
                    train_dataset = train_dataset.map(pipeline)
                    if param.train_augm:
                        train_dataset = train_dataset.map(data_augmentation)
                    if param.train_shuffle:
                        train_dataset = train_dataset.shuffle(
                            buffer_size=train_nelem, reshuffle_each_iteration=True)###########################################################################
#                     train_dataset = train_dataset.repeat(param.num_epochs)
                    train_dataset = train_dataset.batch(self.train_batch_size)
                    train_dataset = train_dataset.prefetch(1)
                    # You can use feedable iterators with a variety of different kinds of iterator
                    # (such as one-shot and initializable iterators).
#                     train_iterator = train_dataset.make_one_shot_iterator() 
                    train_iterator = train_dataset.make_initializable_iterator()
                    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
                    # and used to feed the `handle` placeholder.
                    train_handle = sess.run(train_iterator.string_handle())
                    
                    # A feedable iterator is defined by a handle placeholder and its structure. We
                    # could use the `output_types` and `output_shapes` properties of either
                    # `training_dataset` or `validation_dataset` here, because they have
                    # identical structure.
                    iter_handle = tf.placeholder(tf.string, shape=[])
                    iterator = tf.data.Iterator.from_string_handle(
                        iter_handle, train_dataset.output_types, train_dataset.output_shapes)
                    next_element = iterator.get_next()

                    # Test dataset
                    test_dataset = tf.data.TFRecordDataset(datapaths_test)
                    test_dataset = test_dataset.map(pipeline)
                    if param.test_shuffle:
                        test_dataset = test_dataset.shuffle(
                            buffer_size=test_nelem,
                            reshuffle_each_iteration=True)  ###################
                    test_dataset = test_dataset.batch(test_nelem)
                    test_dataset = test_dataset.prefetch(1)
                    test_iterator = test_dataset.make_initializable_iterator()
                    test_handle = sess.run(test_iterator.string_handle())
                    
                    if self.have_val_data():
                        valid_dataset = tf.data.TFRecordDataset(self.valid_path)
                        valid_dataset = valid_dataset.map(pipeline)
                        if param.valid_shuffle:
                            valid_dataset = valid_dataset.shuffle(
                                buffer_size=self.valid_nelem, reshuffle_each_iteration=True)########################
                        valid_dataset = valid_dataset.batch(self.valid_batch_size)
                        valid_iterator = valid_dataset.make_initializable_iterator()
                        valid_handle = sess.run(valid_iterator.string_handle())
        
#                     viz_train_iterator = train_dataset.make_one_shot_iterator()
#                     viz_train_handle = sess.run(viz_train_iterator.string_handle())

#                     next_images, next_labels = self.get_data_batch(next_element, param.nclasses)
                    next_images, next_labels = self.get_data_batch(next_element, param.input_mode, param.nclasses)
                    next_weight_map = self.get_data_weight_map(next_element, param.mask_mode)
#                     default_weight_map = tf.ones(
#                         tf.shape(next_images), 
#                         dtype=tf.float32, 
#                         name='default_weight_map')

                    input_data = tf.placeholder_with_default(
                        next_images, shape=[None, None, None], name='input_data')
                    input_labels = tf.placeholder_with_default(
                        next_labels, shape=[None, None, None], name='input_labels')
                    weight_map = tf.placeholder_with_default(
                        next_weight_map, shape=[None, None, None], name='weight_map')
                    dropout_keep = tf.placeholder_with_default(1.0, shape=None, name='dropout_keep')
                    is_training = tf.placeholder_with_default(False, shape=None, name='is_training')
#                     weight_map = tf.Variable(
#                         default_weight_map, trainable=False, expected_shape=tf.shape(default_weight_map), name='weight_map')
#                     dropout_keep = tf.placeholder_with_default(1.0, shape=None, name='input_data')

                    # Build network
                    # if self.pretrained_model_exists():
                    #     load_model(sess, self.model_path)
                    #     logits = graph.get_tensor_by_name("train10/Conv2D:0")
                    #     # self.print_names_in_graph()
                    # else:
                    #     logits = build_unet(input_data, 'train', param.nclasses, dropout_keep)
                    #     # self.print_names_in_graph()
                    #################################################################################
                    if param.network == 'unet':
                        logits = build_unet(input_data, 'train', param.nclasses, dropout_keep, is_training)
                    else:
                        n_iterations = 3
                        norm_type = 'bn'
                        # batch_norm_in_train_mode = True
                        num_layers = 4
                        feature_maps_root = 64
                        input_data_expand = tf.expand_dims(input_data, 3)
                        sig_logits, logits, encoder, decoder = iunet('iunet',
                                                                     input_data_expand,
                                                                     n_iterations,
                                                                     norm_type,
                                                                     is_training,
                                                                     num_layers,
                                                                     feature_maps_root,
                                                                     True)
                    print('logits: ', logits)
                    #################################################################################

                    # logits = build_nonlocal_unet(input_data, 'train', param.nclasses, dropout_keep)
#                     print("logits.shape", logits.get_shape())
                    #################################################################################
                    if param.activation == 'sigmoid':
                        if param.network == 'iunet':
                            sig_logits_ = [tf.expand_dims(sig_logit, 0) for sig_logit in sig_logits]
                            tf_prediction = tf.concat(sig_logits_, axis=0)
                        elif param.network == 'unet': #TODO unet and sigmoid, then metrics dont work
                            tf_prediction = tf.nn.sigmoid(logits, name='output')
                    else:
                        tf_prediction = tf.nn.softmax(logits, name='output')
                    # tf_prediction = tf.slice(tf_prediction, [2,0,0,0,0], [1,-1,-1,-1,-1])
                    #################################################################################
                    # Valid and test predictions
#                     tf_valid_prediction = tf.nn.softmax(build_unet(next_element['img'], 'valid', param.nclasses))

#                     # Loss
#                     tf_gt_hot = tf.one_hot(input_labels, param.nclasses)
#                     batch_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
#                         labels=tf_gt_hot,
#                         logits=tf_train_logits)
#                     batch_cross_entropy = tf.Print(batch_cross_entropy, [batch_cross_entropy], 'batch_cross_entropy')
#                     cross_entropy = tf.reduce_mean(batch_cross_entropy)
#                     smry_cross_entropyy = tf.summary.scalar('cross_entropy', cross_entropy)
#                     cross_entropy = tf.Print(cross_entropy, [cross_entropy], 'cross_entropy') #print to the console

#                     # Regularization
#                     reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#                     loss = cross_entropy + tf.reduce_sum(reg_variables)
#                     smry_loss = tf.summary.scalar('loss', loss)
#                     loss = tf.Print(loss, [loss], 'loss') #print to the console

# #                     tf_acc = self.add_eval_step(tf_prediction, input_labels)
#
# #                     tf_precision, tf_recall, tf_specificity, tf_accuracy, tf_true_positives, tf_true_negatives, tf_false_positives, tf_false_negatives, tf_precision2, tf_recall2, tf_pred_argmax = self.add_metrics(tf_prediction, input_labels)
                    #################################################################
                    # tf_precision, tf_recall, tf_specificity, tf_f1, tf_accuracy, _ = self.add_metrics(
                    #     tf_prediction, input_labels)
                    # tf_precision_w, tf_recall_w, tf_specificity_w, tf_f1_w, tf_accuracy_w, tf_pred_argmax = self.add_metrics(
                    #     tf_prediction, input_labels, weight_map)
                    true_pos, true_neg, false_pos, false_neg, tf_precision, tf_recall, tf_specificity, tf_f1, tf_accuracy, _ = self.add_metrics(
                        tf_prediction, input_labels)
                    tf_true_pos_w, tf_true_neg_w, tf_false_pos_w, tf_false_neg_w, tf_precision_w, tf_recall_w, tf_specificity_w, tf_f1_w, tf_accuracy_w, tf_pred_argmax = self.add_metrics(
                        tf_prediction, input_labels, weight_map)
                    #################################################################

                    #################################################################
                    # Loss
                    if param.loss == 'gdl':
                        loss = generalised_dice_loss(tf_prediction, input_labels, weight_map)
                        # smry_loss = tf.summary.scalar('loss', loss)
                        # loss = tf.Print(loss, [loss], 'loss') #print to the console
                    elif param.loss == 'iterative':
                        input_labels_expand = tf.expand_dims(input_labels, 3)
                        input_weightmap_expand = tf.expand_dims(weight_map, 3)
                        loss, loss_summaries = iterative_loss(sig_logits,
                                                              logits,
                                                              input_labels_expand,
                                                              n_iterations=n_iterations,
                                                              iteration_weighing='equal',
                                                              weight_map=input_weightmap_expand)

                    smry_loss = tf.summary.scalar('loss', loss)
                    loss = tf.Print(loss, [loss], 'loss')  # print to the console
                    #################################################################

#                     valid_loss = generalised_dice_loss(tf_valid_prediction,
#                                                       next_element['gth'])#,
#                     tf.summary.scalar('valid_loss', valid_loss)
#                     valid_loss = tf.Print(valid_loss, [valid_loss], 'valid_loss') #print to the console

                    accum_valid_loss = tf.placeholder(tf.float32, name='accum_valid_loss')
                    total_valid_loss = tf.identity(accum_valid_loss)
                    total_valid_loss = tf.Print(total_valid_loss, [total_valid_loss], 'total_valid_loss') #print to the console

#                     optimizer = trainable_model(loss)
                    optimizer, tf_gradients, tf_variables = trainable_model(loss, param.optimizer)

                    output_path = os.path.join(param.output_path, param.output_path + str(kfold))
                    merged_summary = tf.summary.merge_all()
                    train_writer = tf.summary.FileWriter(
                        os.path.join(output_path, param.summary_folder, param.train_sm_folder), sess.graph)
                    valid_writer = tf.summary.FileWriter(
                        os.path.join(output_path, param.summary_folder, param.valid_sm_folder))
                    test_writer = tf.summary.FileWriter(
                        os.path.join(output_path, param.summary_folder, param.test_sm_folder))

                    scope.reuse_variables()
                    init_op = tf.group(
                        tf.global_variables_initializer(),
                        tf.local_variables_initializer())
                    sess.run(init_op)
                    ###################################################

                    moving_variable_list = [v for v in tf.global_variables() if ('bnorm' in v.name and 'moving' in v.name)]
                    saver_variable_list = tf.trainable_variables() + moving_variable_list

                    # saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=param.models_to_keep)
                    saver = tf.train.Saver(var_list=saver_variable_list, max_to_keep=param.models_to_keep)
                    model_path = os.path.join(output_path, param.model_folder, param.model_name)
                    results_path = os.path.join(output_path, param.results_folder)
                    if os.path.exists(results_path):
                        os.rename(results_path, results_path + '_BU')
                    os.makedirs(results_path)

                    # if self.pretrained_model_exists(): # TODO not working
                    #     load_model(sess, self.model_path)
                    #     logits = graph.get_tensor_by_name("train10/Conv2D:0")
                    #     self.print_names_in_graph()

                    nsteps_per_epoch = np.int32(np.ceil(train_nelem / self.train_batch_size))
#                     nsteps_per_valid = self.valid_nelem // self.valid_batch_size
#                     mean_valid_loss = 1.
                    valid_save_data = False
                    test_save_data = False
                    print("Initialized")
                    print('nsteps_per_epoch', nsteps_per_epoch)
                    for epoch in range(param.num_epochs):
                        if param.verbose:
                            print("Epoch:", epoch)
                        sess.run(train_iterator.initializer)# comment to overfit one single image ##########################################
                        step = 0
                        while True:
                            try:
                                # os.write(1, str.encode('Training\n'))
#                                 sess.run(train_iterator.initializer)###########################################################################

                                input_img, true_pos_w, true_neg_w, false_pos_w, false_neg_w, mylogits, precision_w, recall_w, specificity_w, f1_w, accuracy_w, precision, recall, specificity, f1, accuracy, pred_argmax, _, l, predictions, summary, next_sample, batch_images, batch_labels, batch_weight_map = sess.run(
                                    [input_data, tf_true_pos_w, tf_true_neg_w, tf_false_pos_w, tf_false_neg_w, logits, tf_precision_w, tf_recall_w, tf_specificity_w, tf_f1_w, tf_accuracy_w, tf_precision, tf_recall, tf_specificity, tf_f1, tf_accuracy, tf_pred_argmax, optimizer, loss, tf_prediction, merged_summary, next_element, next_images, next_labels, next_weight_map],
                                    feed_dict={iter_handle: train_handle, dropout_keep: param.dropout_keep, is_training: param.batch_norm})#, accum_valid_loss: mean_valid_loss})

                                # movmean = graph.get_tensor_by_name("train9/bnorm2/moving_mean:0")
                                # print(sess.run(movmean))

                                # plt.imshow(input_img[0,:,:])
                                # plt.show()

                                overall_step = nsteps_per_epoch*epoch + step
                                train_writer.add_summary(summary, overall_step)
                                # saver.save(sess, model_path, global_step=overall_step)

                                if (overall_step % param.valid_visual_step == 0):
                                    valid_save_data = True
                                if (overall_step % param.test_visual_step == 0):
                                    test_save_data = True
                                
#                                 if (step % 78 == 0):#################################################################
                                if (overall_step % param.visualization_step == 0):#################################################################
                                    print('-------------------------------------------------------------')
                                    print('TRAINING ----------------------------------------------------')
#                                     print('Minibatch loss at epoch %d step %d (overall step %d): %f' % (epoch, step, overall_step, l))
                                    print('Minibatch loss at step %d (sub-step %d in epoch %d): %f' % (overall_step, step, epoch, l))
                                    
#                                     if (overall_step % (10*param.visualization_step) == 0):
                                    print('Saving training data.')
                                    # if not valid_save_data:
                                    #     valid_save_data = True
                                    
                                    step_res_path = os.path.join(results_path, str(overall_step))
                                    os.makedirs(step_res_path)

                                    #######################################################################
                                    # print('pred_argmax.shape', pred_argmax.shape)
                                    # print('mylogits len', len(mylogits))
                                    # print('mylogits 0', mylogits[2].shape)
                                    # # print('mylogits', mylogits)
                                    # plt.figure(1, figsize=(5, 5))
                                    # for kk in range(mylogits[2].shape[3]):
                                    #     plt.subplot(1, 3, kk+1)
                                    #     plt.imshow(mylogits[2][0,:,:,0])
                                    #     print('unique', np.unique(mylogits[kk]))
                                    # plt.show()
                                    #
                                    # plt.figure(3, figsize=(5, 5))
                                    # for kk in range(predictions.shape[0]):
                                    #     print('predictions', str(kk), ':', np.unique(predictions[kk, 0, :, :, 0]))
                                    #     plt.subplot(1, 3, kk + 1)
                                    #     plt.imshow(np.squeeze(
                                    #         predictions[kk, 0, :, :, 0]))
                                    # plt.show()
                                    if param.network == 'iunet':
                                        predictions = predictions[-1, :, :, :, :]
                                        print('im.shape', predictions.shape)
                                        predictions = np.concatenate(
                                            [predictions, np.ones(predictions.shape) - predictions], 3)
                                        print('im.shape', predictions.shape)
                                    #######################################################################

                                    for ii in range(next_sample['img'].shape[0]):
                                        im = next_sample['img'][ii,:,:]
                                        impath = os.path.join(step_res_path, str(ii) + '_input_B.png')
                                        sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                        impath = os.path.join(step_res_path, str(ii) + '_input.png')
                                        sp.misc.toimage(im, cmin=-0.5, cmax=0.5).save(impath)
                                        im = next_sample['gt'][ii,:,:]
                                        impath = os.path.join(step_res_path, str(ii) + '_gt.png')
                                        sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                        im = next_sample['cnp'][ii,:,:]
                                        impath = os.path.join(step_res_path, str(ii) + '_cnp.png')
                                        sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                        im = next_sample['vld'][ii,:,:]
                                        impath = os.path.join(step_res_path, str(ii) + '_vld.png')
                                        sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                        
                                        im = batch_images[ii,:,:]
                                        impath = os.path.join(step_res_path, str(ii) + '_images_B.png')
                                        sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                        impath = os.path.join(step_res_path, str(ii) + '_images.png')
                                        sp.misc.toimage(im, cmin=-0.5, cmax=0.5).save(impath)
                                        im = batch_labels[ii,:,:]
                                        impath = os.path.join(step_res_path, str(ii) + '_labels.png')
                                        sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                        im = batch_weight_map[ii,:,:]
                                        impath = os.path.join(step_res_path, str(ii) + '_weight_map.png')
                                        sp.misc.toimage(im, cmin=0, cmax=1).save(impath)

                                        for jj in range(predictions.shape[3]):
                                            im = predictions[ii,:,:,jj]
                                            print('im.shape', predictions.shape)
                                            print('im.shape', im.shape)
                                            impath = os.path.join(step_res_path, str(ii) + '_prediction_ch' + str(jj) + '.png')
                                            sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                            impath = os.path.join(step_res_path, str(ii) + '_prediction_ch' + str(jj) + '_B.png')
                                            sp.misc.toimage(im, cmin=0, cmax=1).save(impath)
                                            im = np.multiply(predictions[ii,:,:,jj], batch_weight_map[ii,:,:])
                                            impath = os.path.join(step_res_path, str(ii) + '_prediction_ch' + str(jj) + '_masked.png')
                                            sp.misc.toimage(im, cmin=0, cmax=1).save(impath)
#                                     
                                    if param.verbose:
                                        print('TRAINING next BATCH IN')
                                        for ii in range(next_sample['img'].shape[0]):
#                                         ii=0
                                            print('TRAINING next SAMPLE IN')
                                            plt.figure(1, figsize=(5,5))
                                            plt.subplot(1, 2, 1)
                                            imgplot = plt.imshow(next_sample['vld'][ii,:,:])
                                            plt.subplot(1, 2, 2)
                                            imgplot = plt.imshow(next_sample['gt'][ii,:,:])
                                            plt.show()
                                            print('batch_images.shape', batch_images.shape)
                                            plt.figure(2, figsize=(5,5))
                                            plt.subplot(1, 3, 1)
                                            imgplot = plt.imshow(batch_images[ii,:,:])
                                            plt.subplot(1, 3, 2)
                                            imgplot = plt.imshow(batch_labels[ii,:,:])
                                            plt.subplot(1, 3, 3)
                                            imgplot = plt.imshow(batch_weight_map[ii,:,:])
                                            plt.show()
#                                             print('gt unique: ', np.unique(next_sample['gt'][ii,:,:]))
#                                             print('vld unique: ', np.unique(next_sample['vld'][ii,:,:]))
#                                             print('cnp unique: ', np.unique(next_sample['cnp'][ii,:,:]))
#                                             plt.figure(2, figsize=(5,5))
#                                             plt.subplot(1, 2, 1)
#                                             imgplot = plt.imshow(next_sample['vld'][ii,:,:])
#                                             plt.subplot(1, 2, 2)
#                                             imgplot = plt.imshow(next_sample['cnp'][ii,:,:])
#                                             plt.show()
                                
                                            print('TRAINING predictions')
                                            plt.figure(3, figsize=(5,5))
                                            for jj in range(predictions.shape[3]):
                                                print('predictions', str(jj), ':', np.unique(predictions[ii,:,:,jj]))
                                                plt.subplot(1, 3, jj+1)
                                                plt.title('Filter ' + str(ii), fontsize=25)
                                                plt.imshow(np.squeeze(predictions[ii,:,:,jj]))#, interpolation="nearest")#, cmap="gray")
                                            plt.show()

                                        ##################################################################
                                        print('true_pos_w:', true_pos_w)
                                        print('true_neg_w:', true_neg_w)
                                        print('false_pos_w:', false_pos_w)
                                        print('false_neg_w:', false_neg_w)
                                        print('precision:', precision)
                                        print('recall:', recall)
                                        print('specificity:', specificity)
                                        print('f1:', f1)
                                        print('accuracy:', accuracy)
                                        print('precision_w:', precision_w)
                                        print('recall_w:', recall_w)
                                        print('specificity_w:', specificity_w)
                                        print('f1_w:', f1_w)
                                        print('accuracy_w:', accuracy_w)

                                        ##################################################################
                                        # comment if using iunet
                                        print('pred_argmax.shape', pred_argmax.shape)
                                        plt.figure(3, figsize=(5,5))
                                        for ii in range(predictions.shape[0]):
                                            print('pred_argmax', str(ii), ':', np.unique(pred_argmax[ii,:,:]))
                                            plt.subplot(1, 3, ii+1)
                                            plt.title('pred_argmax ' + str(ii), fontsize=25)
                                            plt.imshow(np.squeeze(pred_argmax[ii,:,:]))
                                        plt.show()
                                        ##################################################################                                            
                                
#                                 print('Increment step')    
                                step += 1 # In 'End-of-epoch computations' step is +1!!!
                            except tf.errors.OutOfRangeError:
                                step -= 1  # To correct last increment before training break
                                # if param.verbose:
                                print('TRAINING finished epoch %d' % (epoch))
                                if epoch % param.model_save_step == 0:
                                    # movmean = graph.get_tensor_by_name("train9/bnorm2/moving_mean:0")
                                    # print(sess.run(movmean))
                                    saver.save(sess, model_path, global_step=overall_step)
                                    test_save_data = True

                                break
                            
#                         print('step', step)
#                         print('overall_step', overall_step)
                                
#                         step -=1 # To correct last increment before training break
                        # End-of-epoch computations
                        if self.have_val_data():
                            os.write(1, str.encode('Validation\n'))
    #                         valid_step = 0
    #                         mean_valid_loss = 0 # as a reminder
                            sess.run(valid_iterator.initializer)
    #                         while True:
    #                             try:
        #                             sess.run(valid_iterator.initializer)
    #                         valid_l, valid_predictions, valid_summary, next_valid_sample = sess.run(
#                             valid_l, valid_predictions, valid_summary, next_valid_sample = sess.run(
#                                 [loss, tf_prediction, merged_summary, next_element], 
#                                 feed_dict={iter_handle: valid_handle})#, accum_valid_loss: mean_valid_loss})
                            
                            valid_precision_w, valid_recall_w, valid_specificity_w, valid_f1_w, valid_accuracy_w, valid_precision, valid_recall, valid_specificity, valid_f1, valid_accuracy, valid_l, valid_predictions, valid_summary, next_valid_sample, valid_batch_images, valid_batch_labels, valid_batch_weight_map = sess.run(
                                [tf_precision_w, tf_recall_w, tf_specificity_w, tf_f1_w, tf_accuracy_w, tf_precision, tf_recall, tf_specificity, tf_f1, tf_accuracy, loss, tf_prediction, merged_summary, next_element, next_images, next_labels, next_weight_map], 
                                feed_dict={iter_handle: valid_handle})#, accum_valid_loss: mean_valid_loss})
                            
    #                         if valid_step == 0:
    #                             mean_valid_loss = valid_l
    #                         else:
    #                             mean_valid_loss = np.mean([mean_valid_loss, valid_l])
    
                            valid_writer.add_summary(valid_summary, overall_step)
                            
#                             if (overall_step % param.visualization_step == 0):
                            print('VALIDATION --------------------------------------------------')
                            print('Minibatch loss at step %d: %f' % (overall_step, valid_l))
                                      
        #                         print('VALIDATION --------------------------------------------------')
        #                         print('Minibatch loss at validation step %d (overall step %d): %f' % (valid_step, overall_step, valid_l))
#                                 if (overall_step % (10*param.visualization_step) == 0):
                            if valid_save_data:
                                print('Saving validation data.')
                                valid_save_data = False
                                
                                step_res_path = os.path.join(results_path, 'val_' + str(overall_step))
                                os.makedirs(step_res_path)
                                for ii in range(next_valid_sample['img'].shape[0]):
                                    im = next_valid_sample['img'][ii,:,:]
                                    impath = os.path.join(step_res_path, str(ii) + '_input.png')
                                    sp.misc.toimage(im, cmin=-0.5, cmax=0.5).save(impath)
                                    im = next_valid_sample['gt'][ii,:,:]
                                    impath = os.path.join(step_res_path, str(ii) + '_gt.png')
                                    sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                    im = next_valid_sample['cnp'][ii,:,:]
                                    impath = os.path.join(step_res_path, str(ii) + '_cnp.png')
                                    sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                    im = next_valid_sample['vld'][ii,:,:]
                                    impath = os.path.join(step_res_path, str(ii) + '_vld.png')
                                    sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                    
                                    im = valid_batch_images[ii,:,:]
                                    impath = os.path.join(step_res_path, str(ii) + '_images_B.png')
                                    sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                    impath = os.path.join(step_res_path, str(ii) + '_images.png')
                                    sp.misc.toimage(im, cmin=-0.5, cmax=0.5).save(impath)
                                    im = valid_batch_labels[ii,:,:]
                                    impath = os.path.join(step_res_path, str(ii) + '_labels.png')
                                    sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                    im = valid_batch_weight_map[ii,:,:]
                                    impath = os.path.join(step_res_path, str(ii) + '_weight_map.png')
                                    sp.misc.toimage(im, cmin=0, cmax=1).save(impath)
                                
#                                             print('img', np.unique(next_sample['img'][ii,:,:]))
#                                             print('cnp', np.unique(next_sample['cnp'][ii,:,:]))
#                                             print('vld', np.unique(next_sample['vld'][ii,:,:]))
#                                             print('gt', np.unique(next_sample['gt'][ii,:,:]))
                                    
                                    for jj in range(valid_predictions.shape[3]):
                                        im = valid_predictions[ii,:,:,jj]
                                        impath = os.path.join(step_res_path, str(ii) + '_prediction_ch' + str(jj) + '.png')
                                        sp.misc.toimage(im, cmin=0, cmax=1).save(impath)
                                        impath = os.path.join(step_res_path, str(ii) + '_prediction_ch' + str(jj) + '_B.png')
                                        sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                        im = np.multiply(valid_predictions[ii,:,:,jj], valid_batch_weight_map[ii,:,:])
                                        impath = os.path.join(step_res_path, str(ii) + '_prediction_ch' + str(jj) + '_masked.png')
                                        sp.misc.toimage(im, cmin=0, cmax=1).save(impath)
                            if param.verbose:
                                print('VALIDATION next BATCH IN')
    #                                     for ii in range(self.valid_batch_size):
                                for ii in range(next_valid_sample['gt'].shape[0]):
                                    print('VALIDATION next SAMPLE IN')
                                    print(next_valid_sample['gt'].shape)
                                    plt.figure(1, figsize=(5,5))
                                    plt.subplot(1, 2, 1)
                                    plt.imshow(next_valid_sample['img'][ii,:,:])
                                    plt.subplot(1, 2, 2)
                                    plt.imshow(next_valid_sample['gt'][ii,:,:])
                                    plt.show()
                                    
                                    print(valid_batch_images.shape)
                                    plt.figure(1, figsize=(5,5))
                                    plt.subplot(1, 3, 1)
                                    plt.imshow(valid_batch_images[ii,:,:])
                                    plt.subplot(1, 3, 2)
                                    plt.imshow(valid_batch_labels[ii,:,:])
                                    plt.subplot(1, 3, 3)
                                    plt.imshow(valid_batch_weight_map[ii,:,:])
                                    plt.show()
                                    
                                    print('VALIDATION predictions')
                                    plt.figure(1, figsize=(5,5))
                                    for jj in range(valid_predictions.shape[3]):
                                        plt.subplot(1, 3, jj+1)
                                        plt.title('Filter ' + str(ii), fontsize=25)
                                        plt.imshow(np.squeeze(valid_predictions[ii,:,:,jj]))#, interpolation="nearest")#, cmap="gray")
                                    plt.show()
#                                         plt.figure(1, figsize=(5,5))
                                    for jj in range(valid_predictions.shape[3]):
                                        print('predictions_' + str(jj), np.unique(valid_predictions[ii,:,:,jj]))
                                        
                                    print('precision:', valid_precision)
                                    print('recall:', valid_recall)
                                    print('specificity:', valid_specificity)
                                    print('f1:', valid_f1)
                                    print('accuracy:', valid_accuracy)
                                    print('precision_w:', valid_precision_w)
                                    print('recall_w:', valid_recall_w)
                                    print('specificity_w:', valid_specificity_w)
                                    print('f1_w:', valid_f1_w)
                                    print('accuracy_w:', valid_accuracy_w)
                                        
                                #############################################################
#                                     print('xdropout_keep_valid:', xdropout_keep_valid)
                                #############################################################
                                    
#                         valid_step +=1 
# #                             except tf.errors.OutOfRangeError:
#                         print('VALIDATION finished -----------------------------------------')
# #                                 total_valid_l = sess.run(total_valid_loss, feed_dict={accum_valid_loss: mean_valid?_loss})
#                         total_valid_l = total_valid_loss.eval(feed_dict={accum_valid_loss: mean_valid_loss.astype(float)})
#                         valid_writer.add_summary(valid_summary, overall_step)
#                         print('Averaged validation loss at step %d (overall step %d): %f' % (valid_step, overall_step, total_valid_l))
#                         break

                        # TEST model
                        if test_nelem > 0:
                            os.write(1, str.encode('Test\n'))
                            sess.run(test_iterator.initializer)
                            test_precision_w, test_recall_w, test_specificity_w, test_f1_w, test_accuracy_w, test_precision, test_recall, test_specificity, test_f1, test_accuracy, test_l, test_predictions, test_summary, next_test_sample, test_batch_images, test_batch_labels, test_batch_weight_map = sess.run(
                                [tf_precision_w, tf_recall_w, tf_specificity_w, tf_f1_w, tf_accuracy_w,
                                 tf_precision, tf_recall, tf_specificity, tf_f1, tf_accuracy, loss, tf_prediction,
                                 merged_summary, next_element, next_images, next_labels, next_weight_map],
                                feed_dict={iter_handle: test_handle})  # , accum_valid_loss: mean_valid_loss})

                            test_writer.add_summary(test_summary, overall_step)

                            print('TEST --------------------------------------------------------')
                            print('Minibatch loss at step %d: %f' % (overall_step, test_l))

                            if test_save_data:
                                print('Saving validation data.')
                                test_save_data = False

                                #######################################################################
                                if param.network == 'iunet':
                                    test_predictions = test_predictions[-1, :, :, :, :]
                                    predictions = np.concatenate(
                                        [predictions, np.ones(predictions.shape) - predictions], 3)
                                #######################################################################

                                step_res_path_test = os.path.join(results_path, 'test_' + str(overall_step))
                                os.makedirs(step_res_path_test)
                                for ii in range(next_test_sample['img'].shape[0]):
                                    im = next_test_sample['img'][ii, :, :]
                                    impath = os.path.join(step_res_path_test, str(ii) + '_input.png')
                                    sp.misc.toimage(im, cmin=-0.5, cmax=0.5).save(impath)
                                    im = next_test_sample['gt'][ii, :, :]
                                    impath = os.path.join(step_res_path_test, str(ii) + '_gt.png')
                                    sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                    im = next_test_sample['cnp'][ii, :, :]
                                    impath = os.path.join(step_res_path_test, str(ii) + '_cnp.png')
                                    sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                    im = next_test_sample['vld'][ii, :, :]
                                    impath = os.path.join(step_res_path_test, str(ii) + '_vld.png')
                                    sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)

                                    im = test_batch_images[ii, :, :]
                                    impath = os.path.join(step_res_path_test, str(ii) + '_images_B.png')
                                    sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                    impath = os.path.join(step_res_path_test, str(ii) + '_images.png')
                                    sp.misc.toimage(im, cmin=-0.5, cmax=0.5).save(impath)
                                    im = test_batch_labels[ii, :, :]
                                    impath = os.path.join(step_res_path_test, str(ii) + '_labels.png')
                                    sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                    im = test_batch_weight_map[ii, :, :]
                                    impath = os.path.join(step_res_path_test, str(ii) + '_weight_map.png')
                                    sp.misc.toimage(im, cmin=0, cmax=1).save(impath)

                                    for jj in range(test_predictions.shape[3]):
                                        im = test_predictions[ii, :, :, jj]
                                        impath = os.path.join(step_res_path_test,
                                                              str(ii) + '_prediction_ch' + str(jj) + '.png')
                                        sp.misc.toimage(im, cmin=0, cmax=1).save(impath)
                                        impath = os.path.join(step_res_path_test,
                                                              str(ii) + '_prediction_ch' + str(jj) + '_B.png')
                                        sp.misc.toimage(im, cmin=np.amin(im), cmax=np.amax(im)).save(impath)
                                        im = np.multiply(test_predictions[ii, :, :, jj],
                                                         test_batch_weight_map[ii, :, :])
                                        impath = os.path.join(step_res_path_test,
                                                              str(ii) + '_prediction_ch' + str(jj) + '_masked.png')
                                        sp.misc.toimage(im, cmin=0, cmax=1).save(impath)
                            if param.verbose:
                                print('TEST next BATCH IN')
                                #                                     for ii in range(self.valid_batch_size):
                                for ii in range(next_test_sample['gt'].shape[0]):
                                    print('TEST next SAMPLE IN')
                                    print(next_test_sample['gt'].shape)
                                    plt.figure(1, figsize=(5, 5))
                                    plt.subplot(1, 2, 1)
                                    plt.imshow(next_test_sample['img'][ii, :, :])
                                    plt.subplot(1, 2, 2)
                                    plt.imshow(next_test_sample['gt'][ii, :, :])
                                    plt.show()

                                    print(test_batch_images.shape)
                                    plt.figure(1, figsize=(5, 5))
                                    plt.subplot(1, 3, 1)
                                    plt.imshow(test_batch_images[ii, :, :])
                                    plt.subplot(1, 3, 2)
                                    plt.imshow(test_batch_labels[ii, :, :])
                                    plt.subplot(1, 3, 3)
                                    plt.imshow(test_batch_weight_map[ii, :, :])
                                    plt.show()

                                    print('TEST predictions')
                                    plt.figure(1, figsize=(5, 5))
                                    for jj in range(test_predictions.shape[3]):
                                        plt.subplot(1, 3, jj + 1)
                                        plt.title('Filter ' + str(ii), fontsize=25)
                                        plt.imshow(np.squeeze(test_predictions[ii, :, :, jj]))
                                    plt.show()

                                    for jj in range(test_predictions.shape[3]):
                                        print('predictions_' + str(jj), np.unique(test_predictions[ii, :, :, jj]))

                                    print('precision:', test_precision)
                                    print('recall:', test_recall)
                                    print('specificity:', test_specificity)
                                    print('f1:', test_f1)
                                    print('accuracy:', test_accuracy)
                                    print('precision_w:', test_precision_w)
                                    print('recall_w:', test_recall_w)
                                    print('specificity_w:', test_specificity_w)
                                    print('f1_w:', test_f1_w)
                                    print('accuracy_w:', test_accuracy_w)

    def get_training_datasets(self, load_ori, load_gt, train_fraction = .999, valid_fraction = .0005):
        ''' 
        Create train/test data sets
        '''       
        print("Creating training datasets...")
        def build_datasets(data, labels, train_fraction, valid_fraction=0):
            stop_train = np.int32(data.shape[0]*train_fraction)
            stop_valid = np.int32(data.shape[0]*(train_fraction+valid_fraction))
            train_data = data[0:stop_train, :, :]
            valid_data = data[stop_train:stop_valid, :, :]
            test_data = data[stop_valid:-1, :, :]
            train_labels = labels[0:stop_train, :, :]
            valid_labels = labels[stop_train:stop_valid, :, :]
            test_labels = labels[stop_valid:-1, :, :]
            return train_data, valid_data, test_data, train_labels, valid_labels, test_labels
        
        train_data, valid_data, test_data, train_labels, valid_labels, test_labels = build_datasets(
            load_ori, load_gt, train_fraction, valid_fraction)   
        print("Train data:", train_data.shape)
        print("Validation data:", valid_data.shape)
        print("Test data:", test_data.shape)
        print("Train labels:", train_labels.shape)
        print("Validation labels:", valid_labels.shape)
        print("Test labels:", test_labels.shape)
        
        return train_data, valid_data, test_data, train_labels, valid_labels, test_labels 

