import tensorflow as tf
import numpy as np
import re
import mwparserfromhell
from datasets import load_dataset
import os

def cleanup_wikimedia_text(text):
    wikicode = mwparserfromhell.parse(text)
    text = wikicode.strip_code()
    # Remove Wikipedia-specific artifact artifacts & boilerplates
    # (e.g., "See also", "References", "External links" sections at the end)
    text = re.split(r'\n==\s*(See also|References|External links|Notes|Bibliography)\s*==', text, flags=re.IGNORECASE)[0]
    # Remove markdown/wiki section headers (e.g., == Section ==, === Sub-section ===)
    text = re.sub(r'={2,}\s*(.*?)\s*={2,}', r'\1.', text)
    # Normalize whitespace (replace multiple spaces, tabs, and newlines with a single space/newline)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    # Remove dangling brackets or template leftovers commonly found in raw dumps
    text = re.sub(r'\{\|[^*]*?\|\}', '', text) # Wiki tables
    text = re.sub(r'\[\[File:.*?\]\]', '', text, flags=re.IGNORECASE) # Image tags
    # Remove corrupted dangling brackets
    text = re.sub(r'\{\|.*', '', text) # Wiki tables
    return text

def get_training_text(max_articles_to_process=50000):
    # Load with streaming=False but pick a fixed number of rows for a massive text pool
    raw_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=False)        
    for i, example in enumerate(raw_dataset):
        if i >= max_articles_to_process:
            break
        article = cleanup_wikimedia_text(str(example["text"]))
        yield article

def build_dataset_pipeline(tokenizer, CHUNK_LEN, BATCH_SIZE = 64, BUFFER_SIZE = 500000, CACHE_FILE = "wikipedia_processed_tokens.npy"):
    # =====================================================================
    # STEP 1: ONE-TIME SECURE PRE-PROCESSING (Safely isolated from TF Graphs)
    # =====================================================================
    if not os.path.exists(CACHE_FILE):
        print("Tokenizing a batch of Wikipedia in an isolated step to prevent GIL crashes...")
        token_buffer = []
        for article in get_training_text():
            # Safe string conversion and encoding
            tokens = tokenizer.encode(article, add_special_tokens=False)
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
    return (
        dataset
        .map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)
    #    .repeat()
        .shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
