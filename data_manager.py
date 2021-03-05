import os
import tensorflow as tf


# Reads data from TFrecord
class DataManager(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''

    # Transforms a scalar string `example_proto` into a pair of a scalar string and
    # a scalar integer, representing an image and its label, respectively.
    def parse_and_decode(self, example_proto, feature_info):
        
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
    
        return features_raw
    
    def preprocess(self, features_raw, feature_info):
        features = dict()
        features = features_raw
        height, width = features['height'], features['width']
        im_shape = tf.cast(tf.stack([height, width]), tf.int64)
        
        features['img'] = tf.reshape(features['img'], im_shape)
        features = {k:( tf.reshape(v, im_shape) 
                           if feature_info[k] == tf.float32 else v ) 
                        for (k, v) in features.items()}
        
        return features
        
    
    def pipeline(self, example_proto):
        feature_info = dict(
            height=tf.int64,
            width=tf.int64,
            img=tf.float32,
            cnp=tf.float32,
            vld=tf.float32,
            gth=tf.float32)
        
        features_raw = self.parse_and_decode(example_proto, feature_info)
        features = self.preprocess(features_raw, feature_info)
    
        return features
        