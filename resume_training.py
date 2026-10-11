import tensorflow as tf
import numpy as np
from model import build_decoder_only_transformer
from datasets import load_dataset
from transformers import AutoTokenizer
from position_encoder import PositionEncoder
from attention import MultiheadAttention
from decoder_block import DecoderBlock
import os

# =====================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# =====================================================================
MAX_LEN = 128              # Maximum context window (sequence length)
BATCH_SIZE = 64            # Batch size for training
BUFFER_SIZE = 500000        # For dataset shuffling
CHUNK_LEN = MAX_LEN + 1

# Model Dimensions
EMBEDDING_LEN = 256
HEAD_NUM = 8
FF_DIM = EMBEDDING_LEN * 4
NUM_BLOCKS = 8

# Cache path for our clean tokens
CACHE_FILE = "wikipedia_processed_tokens.npy"

# Create a directory to save checkpoints
checkpoint_dir = "./training_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_filepath = os.path.join(checkpoint_dir, "mini_gpt_epoch_{epoch:02d}_loss_{loss:.4f}.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,  # Saves the full model architecture + weights
    monitor="loss",           # Tracks training loss since we are streaming raw data
    mode="min",
    save_best_only=True,      # Only saves if the loss drops lower than previous epochs
    verbose=1
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# =====================================================================
# STEP 1: ONE-TIME SECURE PRE-PROCESSING (Safely isolated from TF Graphs)
# =====================================================================
if not os.path.exists(CACHE_FILE):
    print("Tokenizing a batch of Wikipedia in an isolated step to prevent GIL crashes...")
    # Load with streaming=False but pick a fixed number of rows for a massive text pool
    raw_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=False)
    
    token_buffer = []
    max_articles_to_process = 50000  # Generates plenty of training steps for a small model
    
    for i, example in enumerate(raw_dataset):
        if i >= max_articles_to_process:
            break
        # Safe string conversion and encoding
        tokens = tokenizer.encode(str(example["text"]), add_special_tokens=False)
        token_buffer.extend(tokens)
        
    # Trim to fit exactly into CHUNK_LEN increments
    num_chunks = len(token_buffer) // CHUNK_LEN
    token_buffer = token_buffer[:num_chunks * CHUNK_LEN]
    
    # Reshape to a clean 2D NumPy array: [Chunks, 129]
    token_matrix = np.array(token_buffer, dtype=np.int32).reshape(-1, CHUNK_LEN)
    
    print(f"Successfully processed {token_matrix.shape[0]} training fragments!")
    np.save(CACHE_FILE, token_matrix)
    print(f"Saved token matrix to disk: {CACHE_FILE}")

# =====================================================================
# STEP 2: LOADING COMPACT TOKENS DIRECTLY THROUGH PURE TENSORFLOW
# =====================================================================
print("Loading token data safely into pure native TensorFlow...")
token_matrix = np.load(CACHE_FILE)

# Create a dataset directly from the clean numerical array (No raw strings or networking)
dataset = tf.data.Dataset.from_tensor_slices(token_matrix)

def split_input_target(chunk):
    inputs = chunk[:-1]
    targets = chunk[1:]
    return inputs, targets

# Formulate the high-performance training pipeline
train_dataset = (
    dataset
    .map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)
#    .repeat()
    .shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

print("Pipeline constructed successfully with zero native thread locks!")


# =====================================================================
# 4. INSTANTIATING AND COMPILING YOUR MODEL
# =====================================================================
print("\nLoading mini GPT-style Transformer...")
custom_layer_dict = {
    "PositionEncoder": PositionEncoder,
    "MultiheadAttention": MultiheadAttention,
    "DecoderBlock": DecoderBlock
}
mini_gpt = tf.keras.models.load_model(
    "./training_checkpoints/mini_gpt_epoch_08_loss_4.9021.h5",
    custom_objects=custom_layer_dict)

# =====================================================================
# 5. TRAINING RUN
# =====================================================================
# Set steps_per_epoch explicitly because streaming datasets have an infinite/unknown size
steps_per_epoch = 1000
new_total_epochs = 20
last_completed_epoch = 15

print(f"\nStarting pre-training for {new_total_epochs} epochs...")
mini_gpt.fit(
    train_dataset,
    epochs=new_total_epochs, 
    initial_epoch=last_completed_epoch,
    steps_per_epoch=steps_per_epoch,
    callbacks=[checkpoint_callback]
)

