import tensorflow as tf
import numpy as np
import re
import mwparserfromhell
from datasets import load_dataset
import os

class WikiDataset:
    TRAIN_CACHE_FILE = "wikipedia_training_tokens.npy"
    VALIDATE_CACHE_FILE = "wikipedia_validation_tokens.npy"

    def __init__(self, tokenizer, seq_len):
        self.build_token_matrix(tokenizer, seq_len, WikiDataset.TRAIN_CACHE_FILE, 0, 50000)
        self.build_token_matrix(tokenizer, seq_len, WikiDataset.VALIDATE_CACHE_FILE, 50000, 1000)

    def build_token_matrix(self, tokenizer, seq_len, cache_file, start_from, samples_num):
        chunk_len = seq_len + 1
        if not os.path.exists(cache_file):
            print("Tokenizing Wikipedia...")
            token_buffer = []
            for article in get_training_text(start_from=start_from, max_articles_to_process=samples_num):
                # Safe string conversion and encoding
                tokens = tokenizer(article)
                token_buffer.extend(tokens.numpy())            
            # Trim to fit exactly into chunk_len increments
            num_chunks = len(token_buffer) // chunk_len
            token_buffer = token_buffer[:num_chunks * chunk_len]        
            # Reshape to a clean 2D NumPy array: [Chunks, 129]
            token_matrix = np.array(token_buffer, dtype=np.int32).reshape(-1, chunk_len)        
            print(f"Successfully processed {token_matrix.shape[0]} fragments!")
            np.save(cache_file, token_matrix)
            print(f"Saved token matrix to disk: {cache_file}")

    def split_input_target(self, chunk):
        inputs = chunk[:-1]
        targets = chunk[1:]
        return inputs, targets
    
    def get_training_dataset(self, batch_size = 64, buffer_size = 500000):
        token_matrix = np.load(WikiDataset.TRAIN_CACHE_FILE)
        dataset = tf.data.Dataset.from_tensor_slices(token_matrix)
        return (
            dataset
            .map(self.split_input_target, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

    def get_validation_dataset(self, batch_size = 64, buffer_size = 500000):
        token_matrix = np.load(WikiDataset.VALIDATE_CACHE_FILE)
        dataset = tf.data.Dataset.from_tensor_slices(token_matrix)
        return (
            dataset
            .map(self.split_input_target, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

def get_training_text(start_from=0, max_articles_to_process=50000):
    # Load with streaming=False but pick a fixed number of rows for a massive text pool
    raw_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split='train', streaming=False)
    up_to = start_from + max_articles_to_process
    for i, example in enumerate(raw_dataset):
        if i < start_from:
            continue
        if i >= up_to:
            break
        article = cleanup_wikimedia_text(str(example["text"]))
        print(f'Processing sample #{i-start_from} out of {max_articles_to_process}')
        yield article

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