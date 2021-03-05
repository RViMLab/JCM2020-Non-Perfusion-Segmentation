import tensorflow as tf
import numpy as np

import parameters as param

def variable_summaries(name, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
#     with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar(name + '_mean', mean)
#         with tf.name_scope('stddev'):
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(name + '_stddev', stddev)
    tf.summary.scalar(name + '_max', tf.reduce_max(var))
    tf.summary.scalar(name + '_min', tf.reduce_min(var))
    tf.summary.histogram(name + '_histogram', var)
    
def gradient_summaries(loss, num_layers=10):
        gr = tf.get_default_graph()
        for i in range(num_layers):
            nweights = 2 if i+1<6 else (3 if i+1<10 else 1)
            for j in range(nweights):
                weight = gr.get_tensor_by_name('train{}/w{}:0'.format(i+1, j+1))
                grad = tf.gradients(loss, weight)[0]
                mean = tf.reduce_mean(tf.abs(grad))
                tf.summary.scalar('grad_w{}_mean'.format(j+1), mean)#, family='train{}'.format(i+1))
                tf.summary.histogram('grad_w{}'.format(i+1), grad)#, family='train{}'.format(i+1))

# def trainable_model(batch_data, batch_labels):
def trainable_model(loss, algorithm):
    print("Setting up trainable model...")

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        param.start_learning_rate, 
        global_step,
        param.learning_decay_step, 
        param.learning_decay_rate,
        staircase=param.staircase)
#             learning_rate = start_learning_rate
    tf.summary.scalar('learning_rate', learning_rate)

    if algorithm=='sgd':
    #     optimizer = tf.train.MomentumOptimizer(
    #         learning_rate=learning_rate, 
    #         momentum=momentum).minimize(loss, global_step=global_step)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, 
            momentum=param.momentum)
    elif algorithm=='adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
    
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    grad_checks = [tf.check_numerics(grad, 'Gradients exploding') for grad in gradients if grad is not None]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(grad_checks):
        with tf.control_dependencies(update_ops):
            optimize = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
    
#     for index, grad in enumerate(gradients):
#         tf.summary.histogram("{}-grad".format(gradients[index][1].name), gradients[index])
#     for index, grad in enumerate(grad_checks):
#         tf.summary.histogram("{}-grad_checks".format(grad_checks[index][1].name), grad_checks[index])
        
    return optimize, gradients, variables


