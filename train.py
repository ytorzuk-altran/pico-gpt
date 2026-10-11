import tensorflow as tf
import numpy as np
from model import build_decoder_only_transformer
from dataset_pipeline import build_dataset_pipeline
from transformers import AutoTokenizer
import argparse
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

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="./logs", 
    profile_batch="1, 2"  # Profile steps 10 through 20
)

def main():
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    parser = argparse.ArgumentParser(
        description="A script launching miniGPT training."
    )
    parser.add_argument("--checkpoint", type=str, help="Start training from checkpoint")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--initial_epoch", type=int, default=0, help="Initial epoch")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = build_dataset_pipeline(tokenizer, MAX_LEN + 1)
    print("Pipeline constructed successfully!")
    print(f"Vocabulary size is {tokenizer.vocab_size}")
    
    print("Building mini GPT-style Transformer...")
    mini_gpt = build_decoder_only_transformer(
        vocab_size=tokenizer.vocab_size, # Use the tokenizer's exact vocab size
        max_len=MAX_LEN,
        embedding_len=EMBEDDING_LEN,
        head_num=HEAD_NUM,
        ff_dim=FF_DIM,
        num_blocks=NUM_BLOCKS
    )
    # Compile using SparseCategoricalCrossentropy with from_logits=True
    mini_gpt.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    if args.checkpoint:
        print("Loading weights...")
        mini_gpt.load_weights(os.path.join(checkpoint_dir, args.checkpoint))

    steps_per_epoch = 1000

    print(f"\nStarting pre-training for {args.epochs} epochs...")
    mini_gpt.fit(
        train_dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=args.initial_epoch,
        callbacks=[checkpoint_callback]
    )

if __name__ == "__main__":
    main()

