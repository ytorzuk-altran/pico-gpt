import tensorflow as tf
from tensorflow.keras import layers
from decoder_block import DecoderBlock
from position_encoder import PositionEncoder

def build_decoder_only_transformer(
    vocab_size, 
    max_len, 
    embedding_len, 
    head_num, 
    ff_dim, 
    num_blocks=4, 
    dropout_rate=0.1
):
    """
    Assembles the final Decoder-Only Transformer (GPT-style) model.
    
    Args:
        vocab_size (int): Size of the text vocabulary.
        max_len (int): Maximum sequence length (context window).
        embedding_len (int): Dimension of token/position embeddings.
        head_num (int): Number of attention heads.
        ff_dim (int): Intermediate dimension of the Feed-Forward Network.
        num_blocks (int): Number of stacked Decoder blocks.
        dropout_rate (float): Dropout probability.
    """
    # 1. Input layer expecting token IDs: Shape [Batch, Sequence_Length]
    inputs = layers.Input(shape=(max_len,), dtype=tf.int32, name="token_inputs")
    
    # 2. Token Embedding
    x = layers.Embedding(
        input_dim=vocab_size, 
        output_dim=embedding_len, 
        name="token_embeddings"
    )(inputs)
    
    # 3. Position Encoding
    x = x * tf.cast(tf.math.sqrt(tf.cast(embedding_len, tf.float32)), x.dtype)
    x = PositionEncoder(
        seq_len=max_len, 
        embedding_len=embedding_len, 
        name="position_embeddings"
    )(x)
    
    # 4. Optional Initial Dropout
    x = layers.Dropout(dropout_rate, name="input_dropout")(x)
    
    # 5. Stacked Decoder Blocks
    for i in range(num_blocks):
        x = DecoderBlock(
            embedding_len=embedding_len, 
            head_num=head_num, 
            ff_dim=ff_dim, 
            rate=dropout_rate,
            name=f"decoder_block_{i}"
        )(x)
        
    # 6. Final Language Modeling Head
    # Outputs raw logits for every token in the vocabulary across the sequence.
    # Output Shape: [Batch, Sequence_Length, Vocab_Size]
    outputs = layers.Dense(vocab_size, activation=None, name="lm_logits")(x)
    
    # Instantiate and return the Keras model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Decoder_Only_Transformer")
    return model

