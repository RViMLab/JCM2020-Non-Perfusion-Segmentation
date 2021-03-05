import os
import numpy as np
import tensorflow as tf

import cv2

from util_ml import normalize_sample

def add_metrics(tf_prediction, tf_ground_truth, weights=None,
                threshold=0.5, network=None):  # NOTICE ORDER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # 2 class metrics

    if network == 'iunet':  # iunet
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
    f1 = tf.realdiv(2 * precision * recall, tf.add(f1_den, 1e-6), name='f1')

    acc_true = tf.add(true_pos, true_neg)
    acc_false = tf.add(false_pos, false_neg)
    accuracy = tf.realdiv(acc_true, tf.add(acc_true, acc_false))

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

    # tf.reset_default_graph()
    with tf.Session() as sess:
        model_folder = os.path.normpath(model_folder)
        print("Loading model from:", model_folder)
        # Import graph
        last_checkpoint = tf.train.latest_checkpoint(model_folder)
        print("Loading checkpoint:", last_checkpoint)

        new_saver = tf.train.import_meta_graph(last_checkpoint + '.meta')
        new_saver.restore(sess, last_checkpoint)

        gph = tf.get_default_graph()

        input_data = gph.get_tensor_by_name("input_data:0")
        conv10_1 = gph.get_tensor_by_name("train10/Conv2D:0")

        if cv_folder is None:
            files = os.listdir(in_data_folder)
        else:
            files = os.listdir(cv_folder)
        for f in files:
            try:
                filepath = os.path.join(in_data_folder, f)
                print(filepath)
                sample = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                norm_sample = normalize_sample(sample)
                norm_sample = np.expand_dims(norm_sample, 0)

                tf_prediction = tf.nn.softmax(conv10_1)
                prediction = sess.run(tf_prediction, feed_dict={input_data: norm_sample})

                pred0 = prediction[0, :, :, 0]
                pred1 = prediction[0, :, :, 1]

                pred0_bin = np.uint8(pred0 > 0.5)
                pred1_bin = np.uint8(pred1 > 0.5)

                np.save(os.path.join(out_ch0, f[0:-4]), pred0)
                np.save(os.path.join(out_ch1, f[0:-4]), pred1)

                cv2.imwrite(os.path.join(out_ch0_bin, f[0:-4] + '.png'), 255*pred0_bin)
                cv2.imwrite(os.path.join(out_ch1_bin, f[0:-4] + '.png'), 255*pred1_bin)
            except IOError as e:
                print('Could not read:\n', filepath, '\nError', e, '- Skipping file.')
