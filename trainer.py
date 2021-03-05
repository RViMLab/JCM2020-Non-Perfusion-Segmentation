

import tensorflow as tf
from tensorflow.python.client import device_lib
# from show_graph import show_graph
import math
import numpy as np

import matplotlib.pyplot as plt

import parameters as param
from model_util import trainable_model
import os

from input_pipeline import pipeline, data_augmentation
from model_unet import build_unet, generalised_dice_loss


# from iUNET_define import *

class Trainer(object):
    '''
    classdocs
    '''

    def __init__(self, train_batch_size, train_path, nkfolds=0):
        '''
        Constructor
        '''
        self.train_batch_size = train_batch_size
        self.train_path = train_path
        self.nkfolds = nkfolds
        self.filename = 'fold'
        print('self.train_path', self.train_path)
        self.valid_nelem = 0
        
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

        precision = precision0
        recall = recall0
        specificity = specificity0

        correct_prediction = tf.equal(pred_argmax, 
                                      tf.cast(tf_ground_truth, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        return precision, recall, specificity, accuracy, true_pos, true_neg, \
               false_pos, false_neg, precision2, recall2, pred_argmax


    def add_metrics(self, tf_prediction, tf_ground_truth, weights=None, threshold=0.5): # NOTICE ORDER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 2 class metrics

        if param.network == 'iunet':  # iunet
            print('IF tf_prediction', tf_prediction.shape)
            tf_prediction0 = tf_prediction[-1, :, :, :, :] < threshold
            tf_prediction1 = tf_prediction[-1, :, :, :, :] > threshold
            tf_prediction_cc = tf.concat([tf_prediction0, tf_prediction1], 3)
            print('tf_prediction_cc', tf_prediction_cc.shape)
            tf_prediction_cc = tf.cast(tf_prediction_cc, dtype=tf.float32)
            pred_argmax = tf.cast(
                tf.argmax(tf_prediction_cc, axis=3, name='pred_argmax'),
                dtype=tf.float32)
        else:
            pred_argmax = tf.cast(
                tf.argmax(tf_prediction, axis=3, name='pred_argmax'),
                dtype=tf.float32)

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
        if param.nfold is not None:
            self.train_model_fold(param.nfold)
        else:
            for kfold in range(self.nkfolds):
                self.train_model_fold(kfold)

    def train_model_fold(self, kfold):
        with tf.Graph().as_default() as graph:
            # with tf.Session(config=self.config) as sess:
            with tf.Session() as sess:
                with tf.variable_scope('') as scope:
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    print("Training data path:", self.train_path)
                    print("Training fold:", kfold)
                    datapaths_train, datapaths_test = self.get_data_paths(kfold)
                    train_nelem = self.get_tfrecord_size(datapaths_train)
                    test_nelem = self.get_tfrecord_size(datapaths_test)

                    train_dataset = tf.data.TFRecordDataset(datapaths_train)
                    train_dataset = train_dataset.map(pipeline)
                    if param.train_augm:
                        train_dataset = train_dataset.map(data_augmentation)
                    if param.train_shuffle:
                        train_dataset = train_dataset.shuffle(
                            buffer_size=train_nelem, reshuffle_each_iteration=True)
#                     train_dataset = train_dataset.repeat(param.num_epochs)
                    train_dataset = train_dataset.batch(self.train_batch_size)
                    train_dataset = train_dataset.prefetch(1)
                    train_iterator = train_dataset.make_initializable_iterator()
                    train_handle = sess.run(train_iterator.string_handle())

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
                            reshuffle_each_iteration=True)
                    test_dataset = test_dataset.batch(test_nelem)
                    test_dataset = test_dataset.prefetch(1)
                    test_iterator = test_dataset.make_initializable_iterator()
                    test_handle = sess.run(test_iterator.string_handle())
                    
                    if self.have_val_data():
                        valid_dataset = tf.data.TFRecordDataset(self.valid_path)
                        valid_dataset = valid_dataset.map(pipeline)
                        if param.valid_shuffle:
                            valid_dataset = valid_dataset.shuffle(
                                buffer_size=self.valid_nelem, reshuffle_each_iteration=True)
                        valid_dataset = valid_dataset.batch(self.valid_batch_size)
                        valid_iterator = valid_dataset.make_initializable_iterator()
                        valid_handle = sess.run(valid_iterator.string_handle())

                    next_images, next_labels = self.get_data_batch(next_element, param.input_mode, param.nclasses)
                    next_weight_map = self.get_data_weight_map(next_element, param.mask_mode)

                    input_data = tf.placeholder_with_default(
                        next_images, shape=[None, None, None], name='input_data')
                    input_labels = tf.placeholder_with_default(
                        next_labels, shape=[None, None, None], name='input_labels')
                    weight_map = tf.placeholder_with_default(
                        next_weight_map, shape=[None, None, None], name='weight_map')
                    dropout_keep = tf.placeholder_with_default(1.0, shape=None, name='dropout_keep')
                    is_training = tf.placeholder_with_default(False, shape=None, name='is_training')

                    if param.network == 'unet':
                        logits = build_unet(input_data, 'train', param.nclasses, dropout_keep, is_training)

                    if param.activation == 'softmax':
                        tf_prediction = tf.nn.softmax(logits, name='output')

#                     # Regularization
#                     reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#                     loss = cross_entropy + tf.reduce_sum(reg_variables)
#                     smry_loss = tf.summary.scalar('loss', loss)
#                     loss = tf.Print(loss, [loss], 'loss') #print to the console

                    true_pos, true_neg, false_pos, false_neg, tf_precision, \
                    tf_recall, tf_specificity, tf_f1, tf_accuracy, _ = self.add_metrics(
                        tf_prediction, input_labels)
                    tf_true_pos_w, tf_true_neg_w, tf_false_pos_w, tf_false_neg_w, tf_precision_w, \
                    tf_recall_w, tf_specificity_w, tf_f1_w, tf_accuracy_w, tf_pred_argmax = self.add_metrics(
                        tf_prediction, input_labels, weight_map)

                    # Loss
                    if param.loss == 'gdl':
                        loss = generalised_dice_loss(tf_prediction, input_labels, weight_map)
                        # smry_loss = tf.summary.scalar('loss', loss)
                        # loss = tf.Print(loss, [loss], 'loss') #print to the console

                    smry_loss = tf.summary.scalar('loss', loss)
                    loss = tf.Print(loss, [loss], 'loss')  # print to the console

                    accum_valid_loss = tf.placeholder(tf.float32, name='accum_valid_loss')
                    total_valid_loss = tf.identity(accum_valid_loss)
                    # print to the console
                    total_valid_loss = tf.Print(total_valid_loss, [total_valid_loss], 'total_valid_loss')


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

                    moving_variable_list = [v for v in tf.global_variables()
                                            if ('bnorm' in v.name and 'moving' in v.name)]
                    saver_variable_list = tf.trainable_variables() + moving_variable_list

                    saver = tf.train.Saver(var_list=saver_variable_list, max_to_keep=param.models_to_keep)
                    model_path = os.path.join(output_path, param.model_folder, param.model_name)
                    results_path = os.path.join(output_path, param.results_folder)
                    if os.path.exists(results_path):
                        os.rename(results_path, results_path + '_BU')
                    os.makedirs(results_path)

                    nsteps_per_epoch = np.int32(np.ceil(train_nelem / self.train_batch_size))
                    print("Initialized")
                    print('nsteps_per_epoch', nsteps_per_epoch)
                    for epoch in range(param.num_epochs):
                        if param.verbose:
                            print("Epoch:", epoch)
                        sess.run(train_iterator.initializer)# comment to overfit one single image
                        step = 0
                        while True:
                            try:
                                input_img, true_pos_w, true_neg_w, false_pos_w, false_neg_w, mylogits, precision_w, \
                                recall_w, specificity_w, f1_w, accuracy_w, precision, recall, specificity, f1, \
                                accuracy, pred_argmax, _, l, predictions, summary, next_sample, batch_images, \
                                batch_labels, batch_weight_map = sess.run(
                                    [input_data, tf_true_pos_w, tf_true_neg_w, tf_false_pos_w, tf_false_neg_w, logits,
                                     tf_precision_w, tf_recall_w, tf_specificity_w, tf_f1_w, tf_accuracy_w,
                                     tf_precision, tf_recall, tf_specificity, tf_f1, tf_accuracy, tf_pred_argmax,
                                     optimizer, loss, tf_prediction, merged_summary, next_element, next_images,
                                     next_labels, next_weight_map],
                                    feed_dict={iter_handle: train_handle,
                                               dropout_keep: param.dropout_keep,
                                               is_training: param.batch_norm})

                                overall_step = nsteps_per_epoch*epoch + step
                                train_writer.add_summary(summary, overall_step)

                                if param.verbose:
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
                                
#                                 print('Increment step')    
                                step += 1 # In 'End-of-epoch computations' step is +1!!!
                            except tf.errors.OutOfRangeError:
                                step -= 1  # To correct last increment before training break
                                # if param.verbose:
                                print('TRAINING finished epoch %d' % (epoch))
                                if epoch % param.model_save_step == 0:
                                    saver.save(sess, model_path, global_step=overall_step)
                                    test_save_data = True

                                break
                                
#                         step -=1 # To correct last increment before training break
                        # End-of-epoch computations
                        if self.have_val_data():
                            os.write(1, str.encode('Validation\n'))
    #                         valid_step = 0
    #                         mean_valid_loss = 0 # as a reminder
                            sess.run(valid_iterator.initializer)
                            
                            valid_precision_w, valid_recall_w, valid_specificity_w, valid_f1_w, valid_accuracy_w, \
                            valid_precision, valid_recall, valid_specificity, valid_f1, valid_accuracy, valid_l, \
                            valid_predictions, valid_summary, next_valid_sample, valid_batch_images, \
                            valid_batch_labels, valid_batch_weight_map = sess.run(
                                [tf_precision_w, tf_recall_w, tf_specificity_w, tf_f1_w, tf_accuracy_w, tf_precision,
                                 tf_recall, tf_specificity, tf_f1, tf_accuracy, loss, tf_prediction, merged_summary,
                                 next_element, next_images, next_labels, next_weight_map],
                                feed_dict={iter_handle: valid_handle})#, accum_valid_loss: mean_valid_loss})
    
                            valid_writer.add_summary(valid_summary, overall_step)

                            print('VALIDATION --------------------------------------------------')
                            print('Minibatch loss at step %d: %f' % (overall_step, valid_l))

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

                        # TEST model
                        if test_nelem > 0:
                            os.write(1, str.encode('Test\n'))
                            sess.run(test_iterator.initializer)
                            test_precision_w, test_recall_w, test_specificity_w, test_f1_w, test_accuracy_w, \
                            test_precision, test_recall, test_specificity, test_f1, test_accuracy, test_l, \
                            test_predictions, test_summary, next_test_sample, test_batch_images, test_batch_labels, \
                            test_batch_weight_map = sess.run(
                                [tf_precision_w, tf_recall_w, tf_specificity_w, tf_f1_w, tf_accuracy_w,
                                 tf_precision, tf_recall, tf_specificity, tf_f1, tf_accuracy, loss, tf_prediction,
                                 merged_summary, next_element, next_images, next_labels, next_weight_map],
                                feed_dict={iter_handle: test_handle})

                            test_writer.add_summary(test_summary, overall_step)

                            print('TEST --------------------------------------------------------')
                            print('Minibatch loss at step %d: %f' % (overall_step, test_l))

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

