import os
import numpy as np
from scipy import ndimage
import tensorflow as tf

import cv2

import skimage.transform as sktr
import matplotlib.pyplot as plt

# import parameters as param


def load_sample(data_folder, file_name):
    def get_ground_truth(cnp_data, vld_data):
        vld_img_data = vld_data > 0.5
        cnp_img_data = cnp_data > 0.5
        image_gt = np.zeros(vld_img_data.shape, np.float32)
        image_gt[vld_img_data == 1] = 1.
        image_gt[cnp_img_data == 1] = 2.
        return image_gt

    src_img = os.path.join(data_folder, 'Done', file_name)
    src_cnp = os.path.join(data_folder, 'NP', file_name)
    src_vld = os.path.join(data_folder, 'Valid', file_name)
    print(src_img)
    img = (ndimage.imread(src_img).astype(float))
    cnp = (ndimage.imread(src_cnp).astype(float))
    vld = (ndimage.imread(src_vld).astype(float))
    gth = get_ground_truth(cnp, vld)

    sample = {'img': img, 'cnp': cnp, 'vld': img, 'gth': gth}
    return sample

def load_model(session, model_path):
    last_checkpoint = tf.train.latest_checkpoint(model_path) # ...\Model folder
    print('Loading model checkpoint:', last_checkpoint)

    saver = tf.train.import_meta_graph(last_checkpoint + '.meta')
    saver.restore(session, last_checkpoint)

    # gph = tf.get_default_graph()

def normalize_sample(img, pixel_depth=255):
    ori_data = np.float32(img)
    ori_data = (ori_data - pixel_depth / 2) / pixel_depth
    return ori_data

def add_metrics(tf_prediction, tf_ground_truth, weights=None,
                threshold=0.5, network=None):  # NOTICE ORDER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # 2 class metrics

    #############################################################
    # if tf.rank(tf_prediction) is 4: # iunet
    # if tf_prediction.shape[4] is not None:  # iunet
    if network == 'iunet':  # iunet
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
    f1 = tf.realdiv(2 * precision * recall, tf.add(f1_den, 1e-6), name='f1')

    acc_true = tf.add(true_pos, true_neg)
    acc_false = tf.add(false_pos, false_neg)
    accuracy = tf.realdiv(acc_true, tf.add(acc_true, acc_false))

    # with tf.name_scope('metrics' + scope_suffix):  # , reuse=tf.AUTO_REUSE):
    #     tf.summary.scalar('accuracy', accuracy)
    #     tf.summary.scalar('precision', precision)
    #     tf.summary.scalar('recall', recall)
    #     tf.summary.scalar('specificity', specificity)
    #     tf.summary.scalar('f1', f1)

    return true_pos, true_neg, false_pos, false_neg, precision, recall, specificity, f1, accuracy, pred_argmax

def inference(model_folder, in_data_folder, out_folder, cv_folder=None):
    out_ch0_bin = os.path.join(out_folder, 'ch0_bin')
    out_ch1_bin = os.path.join(out_folder, 'ch1_bin')
    out_ch0 = os.path.join(out_folder, 'ch0')
    out_ch1 = os.path.join(out_folder, 'ch1')
    if os.path.isdir(out_folder):
        print('Output folder already exists:', out_folder)
    else:
        print('Creating output folder:', out_folder)
        os.makedirs(out_folder)
        os.makedirs(out_ch0_bin)
        os.makedirs(out_ch1_bin)
        os.makedirs(out_ch0)
        os.makedirs(out_ch1)

    tf.reset_default_graph()
    # imported_meta = tf.train.import_meta_graph(os.path.join(param.output_path, param.model_folder,'Model'))

    # tf.reset_default_graph()
    with tf.Session() as sess:
        model_folder = os.path.normpath(model_folder)
        print("Loading model from:", model_folder)
        # Import graph
        last_checkpoint = tf.train.latest_checkpoint(model_folder)
        # last_checkpoint = os.path.join(model_folder, 'model-97514')#################################################
        print("Loading checkpoint:", last_checkpoint)

        new_saver = tf.train.import_meta_graph(last_checkpoint + '.meta')
        new_saver.restore(sess, last_checkpoint)
        #     print(tf.train.latest_checkpoint(os.path.join(OUTPUT_DIR, MODEL_FOLDER)))

        gph = tf.get_default_graph()

        input_data = gph.get_tensor_by_name("input_data:0")
        # is_training = gph.get_tensor_by_name("is_training:0")
        conv10_1 = gph.get_tensor_by_name("train10/Conv2D:0")

        # movmean = gph.get_tensor_by_name("train9/bnorm2/moving_mean:0")
        # print(sess.run(movmean))

        # data_folder = os.path.join(pred_ref_folder, 'fold' + str(nfold))
        # files = os.listdir(os.path.join(data_folder, 'Done'))
        if cv_folder is None:
            files = os.listdir(in_data_folder)
        else:
            files = os.listdir(cv_folder)
        for f in files:
            try:
                filepath = os.path.join(in_data_folder, f)
                # sample = load_sample(pred_data_folder, f)
                # sample = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                print(filepath)
                sample = (ndimage.imread(filepath).astype(float))

                # maxdim = np.amax(sample.shape)
                # sample2 = np.zeros([maxdim, maxdim])
                # shift = np.asarray([np.round((maxdim-sample.shape[0])/2), np.round((maxdim-sample.shape[1])/2)])
                # shift = shift.astype(np.int)
                # sample2[shift[0]:shift[0]+sample.shape[0], shift[1]:shift[1]+sample.shape[1]] = sample
                # resized_sample = np.round(sktr.resize(sample2, [1024, 1024]))#[1792, 1792]))

                # plt.imshow(sample)
                # plt.show()

                norm_sample = normalize_sample(sample)
                norm_sample = np.expand_dims(norm_sample, 0)

                # plt.imshow(norm_sample[0,:,:])
                # plt.show()

                # init_op = tf.group(
                #     tf.global_variables_initializer(),
                #     tf.local_variables_initializer())
                # sess.run(tf.global_variables_initializer())
                # sess.run(tf.local_variables_initializer())

                # prediction2 = sess.run(tf.nn.sigmoid(conv10_1), feed_dict={input_data: norm_sample})
                tf_prediction = tf.nn.softmax(conv10_1)
                # prediction = sess.run(tf_prediction, feed_dict={input_data: norm_sample, is_training: False})
                prediction = sess.run(tf_prediction, feed_dict={input_data: norm_sample})#, is_training: False})

                ###############################################
                # input_labels = gph.get_tensor_by_name("input_labels:0")
                # cnppath = os.path.join(in_data_folder[:-4] + 'NP', f)
                # cnp = (ndimage.imread(cnppath).astype(float))
                # cnp = (cnp > (255 // 2)).astype(int)
                # norm_cnp = np.expand_dims(cnp, 0)
                #
                # myweight_map = gph.get_tensor_by_name("weight_map:0")
                # vldpath = os.path.join(in_data_folder[:-4] + 'Valid', f)
                # vld = (ndimage.imread(vldpath).astype(float))
                # vld = (vld > (255 // 2)).astype(int)
                # norm_vld = np.expand_dims(vld, 0)
                #
                # plt.imshow(norm_cnp[0, :, :])
                # plt.show()
                # plt.imshow(norm_vld[0, :, :])
                # plt.show()
                #
                # tf_true_pos_w, tf_true_neg_w, tf_false_pos_w, tf_false_neg_w, tf_precision_w, tf_recall_w, tf_specificity_w, tf_f1_w, tf_accuracy_w, tf_pred_argmax = \
                #     add_metrics(tf.nn.softmax(conv10_1), input_labels, myweight_map, threshold=127)
                # prediction, true_pos, true_neg, false_pos, false_neg, precision, recall, specificity, f1, accuracy, pred_argmax = \
                #     sess.run([tf.nn.softmax(conv10_1), tf_true_pos_w, tf_true_neg_w, tf_false_pos_w, tf_false_neg_w, tf_precision_w, tf_recall_w, tf_specificity_w, tf_f1_w, tf_accuracy_w, tf_pred_argmax],
                #              feed_dict={input_data: norm_sample, input_labels: norm_cnp, myweight_map: norm_vld, is_training: False})
                ###############################################
                # tf_ground_truth = tf.constant()
                #
                # tf_pred_argmax = tf.cast(
                #     tf.argmax(tf_prediction, axis=3, name='pred_argmax'),
                #     dtype=tf.float32)
                #
                # tf_true_pos0 = tf.multiply(tf_pred_argmax, input_labels)
                # tf_true_neg0 = tf.multiply(tf_pred_argmax - 1, input_labels - 1)
                # tf_false_pos0 = tf.multiply(tf_pred_argmax, input_labels - 1)
                # tf_false_neg0 = tf.multiply(tf_pred_argmax - 1, input_labels)
                #
                # tf_true_pos = tf.count_nonzero(tf_true_pos0, dtype=tf.float32, name='true_pos')
                # tf_true_neg = tf.count_nonzero(tf_true_neg0, dtype=tf.float32, name='true_neg')
                # tf_false_pos = tf.count_nonzero(tf_false_pos0, dtype=tf.float32, name='false_pos')
                # tf_false_neg = tf.count_nonzero(tf_false_neg0, dtype=tf.float32, name='false_neg')
                #
                # pred_argmax, prediction, true_pos, true_neg, false_pos, false_neg,\
                #     true_pos0, true_neg0, false_pos0, false_neg0 = sess.run(
                #     [tf_pred_argmax, tf_prediction, tf_true_pos, tf_true_neg, tf_false_pos, tf_false_neg,
                #      tf_true_pos0, tf_true_neg0, tf_false_pos0, tf_false_neg0],
                #     feed_dict={input_data: norm_sample, input_labels: norm_cnp, is_training: False})
                #
                # plt.imshow(pred_argmax[0, :, :])
                # plt.show()
                # plt.imshow(true_pos0[0, :, :] != 0)
                # plt.show()
                # plt.imshow(true_neg0[0, :, :] != 0)
                # plt.show()
                # plt.imshow(false_pos0[0, :, :] != 0)
                # plt.show()
                # plt.imshow(false_neg0[0, :, :] != 0)
                # plt.show()

                ###############################################


                # movmean = gph.get_tensor_by_name("train9/bnorm2/moving_mean:0")
                # print(sess.run(movmean))

                # pred_resized0 = np.round(sktr.resize(prediction[0, :, :, 0], [2048, 2048]))#[maxdim, maxdim]))
                # pred_resized1 = np.round(sktr.resize(prediction[0, :, :, 1], [2048, 2048]))#[maxdim, maxdim]))

                pred0 = prediction[0, :, :, 0]
                pred1 = prediction[0, :, :, 1]

                # plt.imshow(pred0, vmin=0, vmax=1)
                # plt.show()
                # plt.imshow(pred1, vmin=0, vmax=1)
                # plt.show()

                # img0_bin = np.float32(pred_orisize0 > 0)
                # img1_bin = np.float32(pred_orisize1 > 0)
                # img_gtr = np.float32(pred_orisize1 > pred_orisize0)

                pred0_bin = np.uint8(pred0 > 0.5)
                pred1_bin = np.uint8(pred1 > 0.5)

                # plt.imshow(pred0_bin)
                # plt.show()
                # plt.imshow(pred1_bin)
                # plt.show()

                # pred0_orisize = pred0_bin_resized[shift[0]:shift[0]+sample.shape[0], shift[1]:shift[1]+sample.shape[1]]
                # pred1_orisize = pred1_bin_resized[shift[0]:shift[0]+sample.shape[0], shift[1]:shift[1]+sample.shape[1]]

                # cv2.imwrite(os.path.join(out_ch0, f[0:-4] + '.png'), pred0)
                # cv2.imwrite(os.path.join(out_ch1, f[0:-4] + '.png'), pred1)
                np.save(os.path.join(out_ch0, f[0:-4]), pred0)
                np.save(os.path.join(out_ch1, f[0:-4]), pred1)

                cv2.imwrite(os.path.join(out_ch0_bin, f[0:-4] + '.png'), 255*pred0_bin)
                cv2.imwrite(os.path.join(out_ch1_bin, f[0:-4] + '.png'), 255*pred1_bin)

                # self.plotNNFilter(units)
                # print('units 0: ', np.unique(units[0,:,:,0]))
                # print('units 1: ', np.unique(units[0,:,:,1]))
                # print('units 2: ', np.unique(units[0,:,:,2]))
            except IOError as e:
                print('Could not read:\n', filepath, '\nError', e, '- Skipping file.')


def inference_testing(model_folder, in_data_folder, out_folder, cv_folder=None):
    out_ch0_bin = os.path.join(out_folder, 'ch0_bin')
    out_ch1_bin = os.path.join(out_folder, 'ch1_bin')
    out_ch0 = os.path.join(out_folder, 'ch0')
    out_ch1 = os.path.join(out_folder, 'ch1')
    if os.path.isdir(out_folder):
        print('Output folder already exists:', out_folder)
    else:
        print('Creating output folder:', out_folder)
        os.makedirs(out_folder)
        os.makedirs(out_ch0_bin)
        os.makedirs(out_ch1_bin)
        os.makedirs(out_ch0)
        os.makedirs(out_ch1)

    tf.reset_default_graph()
    # imported_meta = tf.train.import_meta_graph(os.path.join(param.output_path, param.model_folder,'Model'))

    # tf.reset_default_graph()
    with tf.Session() as sess:
        model_folder = os.path.normpath(model_folder)
        print("Loading model from:", model_folder)
        # Import graph
        last_checkpoint = tf.train.latest_checkpoint(model_folder)
        # last_checkpoint = os.path.join(model_folder, 'model-97514')#################################################
        print("Loading checkpoint:", last_checkpoint)

        new_saver = tf.train.import_meta_graph(last_checkpoint + '.meta')
        new_saver.restore(sess, last_checkpoint)
        #     print(tf.train.latest_checkpoint(os.path.join(OUTPUT_DIR, MODEL_FOLDER)))

        gph = tf.get_default_graph()

        input_data = gph.get_tensor_by_name("input_data:0")
        is_training = gph.get_tensor_by_name("is_training:0")
        conv10_1 = gph.get_tensor_by_name("train10/Conv2D:0")

        # movmean = gph.get_tensor_by_name("train9/bnorm2/moving_mean:0")
        # print(sess.run(movmean))

        # data_folder = os.path.join(pred_ref_folder, 'fold' + str(nfold))
        # files = os.listdir(os.path.join(data_folder, 'Done'))
        if cv_folder is None:
            files = os.listdir(in_data_folder)
        else:
            files = os.listdir(cv_folder)
        for f in files:
            try:
                filepath = os.path.join(in_data_folder, f)
                # sample = load_sample(pred_data_folder, f)
                # sample = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                print(filepath)
                sample = (ndimage.imread(filepath).astype(float))

                # maxdim = np.amax(sample.shape)
                # sample2 = np.zeros([maxdim, maxdim])
                # shift = np.asarray([np.round((maxdim-sample.shape[0])/2), np.round((maxdim-sample.shape[1])/2)])
                # shift = shift.astype(np.int)
                # sample2[shift[0]:shift[0]+sample.shape[0], shift[1]:shift[1]+sample.shape[1]] = sample
                # resized_sample = np.round(sktr.resize(sample2, [1024, 1024]))#[1792, 1792]))

                # plt.imshow(sample)
                # plt.show()

                norm_sample = normalize_sample(sample)
                norm_sample = np.expand_dims(norm_sample, 0)

                plt.imshow(norm_sample[0,:,:])
                plt.show()

                # init_op = tf.group(
                #     tf.global_variables_initializer(),
                #     tf.local_variables_initializer())
                # sess.run(tf.global_variables_initializer())
                # sess.run(tf.local_variables_initializer())

                # prediction2 = sess.run(tf.nn.sigmoid(conv10_1), feed_dict={input_data: norm_sample})
                tf_prediction = tf.nn.softmax(conv10_1)
                prediction = sess.run(tf_prediction, feed_dict={input_data: norm_sample, is_training: False})

                ###############################################
                input_labels = gph.get_tensor_by_name("input_labels:0")
                cnppath = os.path.join(in_data_folder[:-4] + 'NP', f)
                cnp = (ndimage.imread(cnppath).astype(float))
                cnp = (cnp > (255 // 2)).astype(int)
                norm_cnp = np.expand_dims(cnp, 0)

                myweight_map = gph.get_tensor_by_name("weight_map:0")
                vldpath = os.path.join(in_data_folder[:-4] + 'Valid', f)
                vld = (ndimage.imread(vldpath).astype(float))
                vld = (vld > (255 // 2)).astype(int)
                norm_vld = np.expand_dims(vld, 0)

                plt.imshow(norm_cnp[0, :, :])
                plt.show()
                plt.imshow(norm_vld[0, :, :])
                plt.show()

                tf_true_pos_w, tf_true_neg_w, tf_false_pos_w, tf_false_neg_w, tf_precision_w, tf_recall_w, tf_specificity_w, tf_f1_w, tf_accuracy_w, tf_pred_argmax = \
                    add_metrics(tf.nn.softmax(conv10_1), input_labels, myweight_map, threshold=127)
                prediction, true_pos, true_neg, false_pos, false_neg, precision, recall, specificity, f1, accuracy, pred_argmax = \
                    sess.run([tf.nn.softmax(conv10_1), tf_true_pos_w, tf_true_neg_w, tf_false_pos_w, tf_false_neg_w, tf_precision_w, tf_recall_w, tf_specificity_w, tf_f1_w, tf_accuracy_w, tf_pred_argmax],
                             feed_dict={input_data: norm_sample, input_labels: norm_cnp, myweight_map: norm_vld, is_training: False})
                ###############################################
                # tf_ground_truth = tf.constant()

                tf_pred_argmax = tf.cast(
                    tf.argmax(tf_prediction, axis=3, name='pred_argmax'),
                    dtype=tf.float32)

                tf_true_pos0 = tf.multiply(tf_pred_argmax, input_labels)
                tf_true_neg0 = tf.multiply(tf_pred_argmax - 1, input_labels - 1)
                tf_false_pos0 = tf.multiply(tf_pred_argmax, input_labels - 1)
                tf_false_neg0 = tf.multiply(tf_pred_argmax - 1, input_labels)

                tf_true_pos = tf.count_nonzero(tf_true_pos0, dtype=tf.float32, name='true_pos')
                tf_true_neg = tf.count_nonzero(tf_true_neg0, dtype=tf.float32, name='true_neg')
                tf_false_pos = tf.count_nonzero(tf_false_pos0, dtype=tf.float32, name='false_pos')
                tf_false_neg = tf.count_nonzero(tf_false_neg0, dtype=tf.float32, name='false_neg')

                pred_argmax, prediction, true_pos, true_neg, false_pos, false_neg,\
                    true_pos0, true_neg0, false_pos0, false_neg0 = sess.run(
                    [tf_pred_argmax, tf_prediction, tf_true_pos, tf_true_neg, tf_false_pos, tf_false_neg,
                     tf_true_pos0, tf_true_neg0, tf_false_pos0, tf_false_neg0],
                    feed_dict={input_data: norm_sample, input_labels: norm_cnp, is_training: False})

                plt.imshow(pred_argmax[0, :, :])
                plt.show()
                plt.imshow(true_pos0[0, :, :] != 0)
                plt.show()
                plt.imshow(true_neg0[0, :, :] != 0)
                plt.show()
                plt.imshow(false_pos0[0, :, :] != 0)
                plt.show()
                plt.imshow(false_neg0[0, :, :] != 0)
                plt.show()

                ###############################################


                # movmean = gph.get_tensor_by_name("train9/bnorm2/moving_mean:0")
                # print(sess.run(movmean))

                # pred_resized0 = np.round(sktr.resize(prediction[0, :, :, 0], [2048, 2048]))#[maxdim, maxdim]))
                # pred_resized1 = np.round(sktr.resize(prediction[0, :, :, 1], [2048, 2048]))#[maxdim, maxdim]))

                pred0 = prediction[0, :, :, 0]
                pred1 = prediction[0, :, :, 1]

                plt.imshow(pred0, vmin=0, vmax=1)
                plt.show()
                plt.imshow(pred1, vmin=0, vmax=1)
                plt.show()

                # img0_bin = np.float32(pred_orisize0 > 0)
                # img1_bin = np.float32(pred_orisize1 > 0)
                # img_gtr = np.float32(pred_orisize1 > pred_orisize0)

                pred0_bin = np.uint8(pred0 > 0.5)
                pred1_bin = np.uint8(pred1 > 0.5)

                plt.imshow(pred0_bin)
                plt.show()
                plt.imshow(pred1_bin)
                plt.show()

                # pred0_orisize = pred0_bin_resized[shift[0]:shift[0]+sample.shape[0], shift[1]:shift[1]+sample.shape[1]]
                # pred1_orisize = pred1_bin_resized[shift[0]:shift[0]+sample.shape[0], shift[1]:shift[1]+sample.shape[1]]

                cv2.imwrite(os.path.join(out_ch0, f[0:-4] + '_c1.png'), pred0)
                cv2.imwrite(os.path.join(out_ch1, f[0:-4] + '_c2.png'), pred1)

                cv2.imwrite(os.path.join(out_ch0_bin, f[0:-4] + '.png'), 255*pred0_bin)
                cv2.imwrite(os.path.join(out_ch1_bin, f[0:-4] + '.png'), 255*pred1_bin)

                # self.plotNNFilter(units)
                # print('units 0: ', np.unique(units[0,:,:,0]))
                # print('units 1: ', np.unique(units[0,:,:,1]))
                # print('units 2: ', np.unique(units[0,:,:,2]))
            except IOError as e:
                print('Could not read:\n', filepath, '\nError', e, '- Skipping file.')

#############################################################################################################
    
    
# #     graph_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
# #     print(len(graph_nodes))
# 
# #     for op in tf.get_default_graph().get_operations():
# #         print(str(op.name))
#     
# #     pp = [k.name for k in tf.get_default_graph().get_operations() if 'Unit1' in k.name]
# #     for ii in pp:
# #         print(ii)
#     
#     tf_train_data = gph.get_tensor_by_name("train_data:0")
#     tf_train_labels = gph.get_tensor_by_name("train_labels:0")
# #     unit5_w1 = gph.get_tensor_by_name("Unit5/w1:0")
#     conv10_1 = gph.get_tensor_by_name("train10/Conv2D:0")
# #     r1_2 = gph.get_tensor_by_name("Unit1/r2:0")
#     print(conv10_1.shape)
#     
#     offset = 25#(step * batch_size) % (train_labels.shape[0] - batch_size)
#     batch_data = train_data[offset:(offset + batch_size), :, :]
#     batch_labels = train_labels[offset:(offset + batch_size), :]
#     
#     pred = getActivations(conv10_1, np.expand_dims(batch_data,3))
# #     print('Prediction difference: ', np.sum(np.abs(predictions - pred)))
#     
# #     pred = getActivations(r1_2, np.expand_dims(batch_data,3))
#     
# #     feed_dict = {tf_train_data: np.expand_dims(batch_data,3), 
# #                  tf_train_labels: np.expand_dims(batch_labels,3)}
#     
# #     Unit1_w1 = gph.get_tensor_by_name("Unit1/w1:0")
#     
#     
# #     l = sess.run([loss], feed_dict=feed_dict) 