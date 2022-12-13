# %%
import os
import re
import random
import string
import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization

# %%


class TokenPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenPositionEmbedding, self).__init__()

        self.token_embed = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim)
        self.position_embed = tf.keras.layers.Embedding(
            input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen)
        positions = self.position_embed(positions)
        x = self.token_embed(x)
        return x+positions

# %%


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dims, num_heads, learning_rate=0.15):
        super(TransformerBlock, self).__init__()

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads, embed_dims)
        self.dropout1 = tf.keras.layers.Dropout(learning_rate)
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.feed_forward_network = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_dims, activation="relu"),
            tf.keras.layers.Dense(embed_dims)
        ])

    def causal_mask(self, batch, size):
        x, y = tf.expand_dims(tf.range(size), 1), tf.range(size)
        mask = x >= y
        mask = tf.reshape(mask, (1, size, size))
        mask = tf.tile(mask, (batch, 1, 1))

        return mask

        self.dropout2 = tf.keras.layers.Dropout(learning_rate)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        mask = self.causal_mask(batch_size, seq_len)

        mha_out = self.attention(inputs, inputs, attention_mask=mask)
        dropout1_out = self.dropout1(mha_out)
        norm1_out = self.norm_1(dropout1_out)
        combined1_out = norm1_out+inputs

        ffn_out = self.feed_forward_network(combined1_out)
        dropout2_out = self.dropout2(ffn_out)
        norm2_out = self.norm_2(dropout2_out)
        combined_2_out = norm2_out+combined1_out

        return combined_2_out

# %%
