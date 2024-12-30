import tensorflow as tf

class DatasetPipeline(tf.data.Dataset):
    def __new__(cls, labels, conditions, shape=None):
        return tf.data.Dataset.from_tensor_slices((conditions, labels)).prefetch(tf.data.experimental.AUTOTUNE)