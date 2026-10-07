import tensorflow as tf
import numpy as np
import argparse
from transformers import AutoTokenizer
from position_encoder import PositionEncoder
from attention import MultiheadAttention
from decoder_block import DecoderBlock

# =====================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# =====================================================================
MAX_LEN = 128              # Maximum context window (sequence length)

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
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
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
        if next_token_id == tokenizer.eos_token_id:
            break
            
    # 8. Decode the complete array of token IDs back into readable English text
    generated_text = tokenizer.decode(input_ids, skip_special_tokens=True)
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
    args = parser.parse_args()
    print("\nBuilding your mini GPT-style Transformer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    custom_layer_dict = {
        "PositionEncoder": PositionEncoder,
        "MultiheadAttention": MultiheadAttention,
        "DecoderBlock": DecoderBlock
    }
    checkpoint_path = "./training_checkpoints/mini_gpt_epoch_15_loss_4.4983.h5"
    print(f"Loading custom Transformer checkpoint from: {checkpoint_path}...")
    loaded_model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects=custom_layer_dict
    )
    output = generate_text(
        model=loaded_model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=40,
        temperature=1.1,
        top_k=20
    )
    print(output)

if __name__ == "__main__":
    main()

