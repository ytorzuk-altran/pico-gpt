import tensorflow as tf
from transformers import AutoTokenizer
from position_encoder import PositionEncoder
from attention import MultiheadAttention
from decoder_block import DecoderBlock
from dataset_pipeline import build_dataset_pipeline
import os

# =====================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# =====================================================================
# Model Dimensions
MAX_LEN = 128              # Maximum context window (sequence length)
EMBEDDING_LEN = 256
HEAD_NUM = 8
FF_DIM = EMBEDDING_LEN * 4
NUM_BLOCKS = 8

# Create a directory to save checkpoints
checkpoint_dir = "./training_checkpoints"
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


train_dataset = build_dataset_pipeline(tokenizer, MAX_LEN + 1)
print("Pipeline constructed successfully!")


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

