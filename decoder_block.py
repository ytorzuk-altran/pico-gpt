import tensorflow as tf
from tensorflow.keras import layers
from attention import MultiheadAttention

class DecoderBlock(layers.Layer):
    def __init__(self, embedding_len, head_num, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embedding_len = embedding_len
        self.head_num = head_num
        self.ff_dim = ff_dim
        
        self.mha = MultiheadAttention(embedding_len=embedding_len, head_num=head_num)
        
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embedding_len)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Attention + Residual
        attn_output = self.mha(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # FFN + Residual
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_len": self.embedding_len,
            "head_num": self.head_num,
            "ff_dim": self.ff_dim,
        })
        return config
