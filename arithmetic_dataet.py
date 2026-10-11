import random
import os
import numpy as np
import tensorflow as tf

def generate_example(max_num=100):
    a = random.randint(0, max_num)
    b = random.randint(1, max_num)
    op = random.choice(['+', '-', '*', '/'])

    if op == '+':
        result = a + b
    elif op == '-':
        a, b = max(a, b), min(a, b)
        result = a - b
    elif op == '*':
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        result = a * b
    else:
        result = random.randint(0, max_num)
        a = b * result

    return f'Problem: {a} {op} {b}.\nAnswer: {result}'

def build_dataset_pipeline(tokenizer, CHUNK_LEN, BATCH_SIZE = 64, BUFFER_SIZE = 500000, CACHE_FILE = "arithmetic_processed_tokens.npy"):
    if not os.path.exists(CACHE_FILE):
        print("Generating math dataset.")
        random.seed(0)
        samples_num = 100000
        token_buffer = []
        for i in range(samples_num):
            input_ids = tokenizer.encode(generate_example(), add_special_tokens=False)
            context_window_ids = input_ids[-CHUNK_LEN:]
            padding_needed = CHUNK_LEN - len(context_window_ids)
            padded_ids = [0] * padding_needed + context_window_ids
            token_buffer.extend(padded_ids)
        num_chunks = len(token_buffer) // CHUNK_LEN
        token_buffer = token_buffer[:num_chunks * CHUNK_LEN]
        token_matrix = np.array(token_buffer, dtype=np.int32).reshape(-1, CHUNK_LEN)
        print(f"Successfully processed {token_matrix.shape[0]} training fragments!")
        np.save(CACHE_FILE, token_matrix)
        print(f"Saved token matrix to disk: {CACHE_FILE}")
    print("Loading token data safely into pure native TensorFlow...")
    token_matrix = np.load(CACHE_FILE)
    dataset = tf.data.Dataset.from_tensor_slices(token_matrix)    
    def split_input_target(chunk):
        inputs = chunk[:-1]
        targets = chunk[1:]
        return inputs, targets
    return (
        dataset
        .map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )