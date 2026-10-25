import tensorflow as tf
import keras
from wiki_dataset import get_training_text
import json
import os

os.environ["HF_DATASETS_OFFLINE"] = "1"

def build_tokenizer(vocab_file="vocab.json", MAX_LEN = None, MAX_VOCAB_SIZE = 4000):

    try:
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
            tokenizer = keras.layers.TextVectorization(
                max_tokens=len(vocab),
                standardize="lower_and_strip_punctuation",
                split="whitespace",
                output_mode="int",
                output_sequence_length=MAX_LEN,
                vocabulary=vocab,
            )
            return tokenizer
    except FileNotFoundError:
        tokenizer = keras.layers.TextVectorization(
            max_tokens=MAX_VOCAB_SIZE,
            standardize="lower_and_strip_punctuation",
            split="whitespace",
            output_mode="int",
            output_sequence_length=MAX_LEN,
            name="text_vectorization",
        )
        tokenizer.adapt(
            tf.data.Dataset.from_generator(get_training_text,
                                           output_signature=tf.TensorSpec(shape=(), dtype=tf.string)).batch(256)
        )
        vocab = tokenizer.get_vocabulary()
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

        return tokenizer
