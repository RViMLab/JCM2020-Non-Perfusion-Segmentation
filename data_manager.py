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
    #     features = {"img": tf.FixedLenFeature((), tf.string, default_value=""),
    #                 "cnp": tf.FixedLenFeature((), tf.string, default_value=""),
    #                 "vld": tf.FixedLenFeature((), tf.string, default_value=""),
    #                 "gth": tf.FixedLenFeature((), tf.string, default_value="")}
        
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
    
    def get_pipeline(self, tfrec_path, batch_size):
        # Build pipeline
#             data_files = [tfrec_path]

            # Define training and validation datasets with the same structure.
            train_dataset = tf.data.TFRecordDataset(os.path.join(data_path_resize, train_files[0]))#r'.\train.tfrecords')

            train_dataset = train_dataset.map(self.pipeline)
            train_dataset = train_dataset.batch(batch_size)

            # A feedable iterator is defined by a handle placeholder and its structure. We
            # could use the `output_types` and `output_shapes` properties of either
            # `training_dataset` or `validation_dataset` here, because they have
            # identical structure.
            iter_handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                iter_handle, train_dataset.output_types, train_dataset.output_shapes)
            next_element = iterator.get_next()

            # You can use feedable iterators with a variety of different kinds of iterator
            # (such as one-shot and initializable iterators).
            train_iterator = train_dataset.make_one_shot_iterator()

            # The `Iterator.string_handle()` method returns a tensor that can be evaluated
            # and used to feed the `handle` placeholder.
            train_handle = session.run(train_iterator.string_handle())
            
            return iterator

        
        