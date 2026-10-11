import tensorflow as tf
import math
from tensorflow.keras import layers

class MultiheadAttention(layers.Layer):
    def __init__(self, embedding_len, head_num=1, **kwargs):
        super().__init__(**kwargs)
        self.embedding_len = embedding_len
        self.head_num = head_num
        assert embedding_len % head_num == 0, "embedding_len must be divisible by head_num"
        self.head_dim = embedding_len // head_num
        self.Q = layers.Dense(embedding_len)
        self.K = layers.Dense(embedding_len)
        self.V = layers.Dense(embedding_len)
        self.out_dense = layers.Dense(embedding_len)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.head_num, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        q = self.Q(inputs)
        k = self.K(inputs)
        v = self.V(inputs)
        q = self.split_heads(q, batch_size)  # (batch_size, head_num, seq_len, head_dim)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        qk = tf.matmul(q, k, transpose_b=True) / math.sqrt(self.head_dim)
        # Causal mask
        ones = tf.ones(tf.stack([seq_len, seq_len]), dtype=inputs.dtype)
        lower_triangular = tf.linalg.band_part(ones, num_lower=-1, num_upper=0)
        causal_mask = 1.0 - lower_triangular
        causal_mask = tf.reshape(causal_mask, (1, 1, seq_len, seq_len)) * -1e9
        qk_masked = qk + causal_mask
        qk_softmax = tf.nn.softmax(qk_masked, axis=-1)
        qkv = tf.matmul(qk_softmax, v)
        attention_output = tf.transpose(qkv, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, seq_len, self.embedding_len))
        return self.out_dense(concat_attention)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_len": self.embedding_len,
            "head_num": self.head_num,
        })
        return config
