import tensorflow as tf
import math
from tensorflow.keras import layers

class PositionEncoder(layers.Layer):
    def __init__(self, seq_len, embedding_len, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embedding_len = embedding_len
        pos_encoding_matrix = self.get_pos_encoding(seq_len, embedding_len)
        self.pos_encoding = self.add_weight(
            name="pos_encoding",
            shape=[seq_len, embedding_len],
            initializer=tf.keras.initializers.Constant(pos_encoding_matrix),
            trainable=False
        )
    
    def get_pos_encoding(self, seq_len, embedding_len):
        positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(embedding_len, dtype=tf.float32)[tf.newaxis, :]
        emb_len_float = tf.cast(embedding_len, dtype=tf.float32)
        angle_rates = 1.0 / tf.pow(10000.0, (2 * (i // 2)) / emb_len_float)
        angle_rads = positions * angle_rates
        is_even = tf.cast(tf.range(embedding_len) % 2 == 0, dtype=tf.float32)[tf.newaxis, :]
        return tf.sin(angle_rads) * is_even + tf.cos(angle_rads) * (1.0 - is_even)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[tf.newaxis, :seq_len, :]
    
    def compute_output_shape(self, input_shape):
        # Keras 3 will read this instead of trying to auto-trace the sparse tensor.
        # It returns the exact shape of your output dense tensor.
        return input_shape 
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "embedding_len": self.embedding_len,
        })
        return config
