import tensorflow as tf
import numpy as np
import argparse
from tokenizer import build_tokenizer
from model import build_decoder_only_transformer
import os

# =====================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# =====================================================================
# =====================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# =====================================================================
# Model Dimensions
MAX_LEN = 128              # Maximum context window (sequence length)
EMBEDDING_LEN = 256
HEAD_NUM = 8
FF_DIM = EMBEDDING_LEN * 4
NUM_BLOCKS = 8

checkpoint_dir = "./training_checkpoints"

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=40):
    """
    Generates text autoregressively using a trained decoder-only model.
    
    Args:
        model: The trained Keras Transformer model.
        tokenizer: The Hugging Face tokenizer instance used during training.
        prompt (str): The starting text string.
        max_new_tokens (int): How many total tokens to generate.
        temperature (float): Controls creativity (higher = more random, lower = deterministic).
        top_k (int): Limits sampling pool to the top N most probable tokens.
    """
    # 1. Convert text prompt into raw token IDs
    input_ids = tokenizer(prompt).numpy().tolist()
    
    print(f"--- Generating text for prompt: '{prompt}' ---")
    
    for _ in range(max_new_tokens):
        # 1. Grab the latest tokens up to MAX_LEN
        context_window_ids = input_ids[-MAX_LEN:]
        
        # 2. Calculate how much padding is needed
        padding_needed = MAX_LEN - len(context_window_ids)
        
        # 3. Left-pad with 0s (or your tokenizer's pad_token_id)
        padded_ids = [0] * padding_needed + context_window_ids
        
        # 4. Wrap in batch dimension -> Shape becomes exactly [1, 128]
        model_input = tf.convert_to_tensor([padded_ids], dtype=tf.int32)
        
        # 5. Run forward pass
        logits = model(model_input, training=False)
        
        # Grab the logits for the absolute last token
        next_token_logits = logits[0, -1, :]
        
        
        # 4. Apply Temperature Scaling
        # Lower temp squeezes distribution (confident), higher flattens it (random)
        if temperature > 0.0:
            next_token_logits = next_token_logits / temperature
            
            # 5. Apply Top-K Filtering
            # Find the values and indices of the top_k tokens
            values, indices = tf.math.top_k(next_token_logits, k=top_k)
            
            # Create a mask filled with negative infinity for all lower-probability tokens
            mask = tf.ones_like(next_token_logits) * -float('Inf')
            
            # Scatter the true top_k values back into the masked array
            # This completely blocks out the "noise" of bad token predictions
            next_token_logits = tf.tensor_scatter_nd_update(
                mask, 
                tf.expand_dims(indices, axis=-1), 
                values
            )
            
            # 6. Sample using a categorical probability distribution
            next_token_id = tf.random.categorical(
                tf.expand_dims(next_token_logits, axis=0), 
                num_samples=1
            )[0, 0].numpy()
        else:
            # Greedy search: simply pick the absolute highest probability token
            next_token_id = tf.argmax(next_token_logits, axis=-1).numpy()
            
        # 7. Append the generated token back to our ongoing tracking array
        input_ids.append(int(next_token_id))
        
        # Optional early stopping: break if model predicts the End-Of-Sequence token
        if next_token_id == 0:
            break
            
    # 8. Decode the complete array of token IDs back into readable English text
    vocab = tokenizer.get_vocabulary()
    generated_text = " ".join([vocab[idx] for idx in input_ids])
    return generated_text

def main():
    parser = argparse.ArgumentParser(
        description="A sample script launching miniGPT inference."
    )
    parser.add_argument(
        "prompt", 
        type=str, 
        help="Prompt to miniGPT."
    )
    parser.add_argument("--checkpoint", type=str, help="Load model from checkpoint")
    args = parser.parse_args()
    print("\nBuilding your mini GPT-style Transformer...")
    tokenizer = build_tokenizer()
    
    mini_gpt = build_decoder_only_transformer(
        vocab_size=tokenizer.vocabulary_size(),
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
        output = generate_text(
            model=mini_gpt,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=40,
            temperature=1.1,
            top_k=20
        )
        print(output)

if __name__ == "__main__":
    main()

