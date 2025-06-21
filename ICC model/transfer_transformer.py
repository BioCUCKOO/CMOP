# -*- coding: utf-8 -*-
# @Time    : 2024/11/6  15:59
# @Author  : Gou Yujie
# @File    : Transformer_encode.py
import os
import pandas as pd
import numpy as np
# 先把32种疾病的同一个组学拼接在一起，然后各组学做transformer编码，最后把编码后的特征拼接到一起
import glob
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.metrics import AUC
from keras.optimizers import Adam, RMSprop
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import *
import keras.backend as K
from sklearn.preprocessing import StandardScaler
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from sklearn import metrics
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
import re
from tensorflow import keras


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.projection = layers.Dense(embed_dim)  # 用于将输入矩阵调整到相同的embed_dim维度
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)


    def call(self, x):
        x = self.projection(x)  # 将输入矩阵映射到相同的embed_dim
        return x


class TransformerAutoencoder(tf.keras.Model):
    def __init__(self, maxlen, embed_dim, num_heads, ff_dim, num_blocks):
        super(TransformerAutoencoder, self).__init__()
        self.embedding = TokenAndPositionEmbedding(maxlen, embed_dim)

        # 编码器堆叠
        self.encoder = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_blocks)
        ]

        # 解码器堆叠
        self.decoder = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_blocks)
        ]

        # 全局池化获取编码后的特征
        self.global_pooling = layers.GlobalAveragePooling1D()

        # 重建层，将嵌入维度映射回原始输入维度
        self.reconstruction = layers.Dense(maxlen)

    def call(self, inputs, training=False):
        # 编码部分
        x = self.embedding(inputs)
        for block in self.encoder:
            x = block(x, training=training)
        encoded = self.global_pooling(x)
        decoded = tf.expand_dims(encoded, axis=1)
        for block in self.decoder:
            decoded = block(decoded, training=training)
        reconstructed = self.reconstruction(decoded)
        reconstructed = tf.squeeze(reconstructed, axis=1)

        return reconstructed, encoded  # 返回重建结果和编码特征

    def get_encoder(self):
        # 定义一个单独的编码器模型
        inputs = layers.Input(shape=(self.embedding.maxlen,))
        _, encoded = self.call(inputs)
        encoder_model = keras.Model(inputs=inputs, outputs=encoded, name="encoder")
        return encoder_model


expfo = "F:\cmm/back_match\expression_frame/"
os.chdir(expfo)
embed_dim = 16
num_heads = 8
ff_dim = 512
num_blocks = 2
batch_size_per_replica = 32  # 根据CPU性能调整
over_omic=pd.read_table('total.txt', index_col=0)
over_omic=over_omic.astype(float)
over_omic.replace(np.inf,10, inplace=True)
over_omic.fillna(0, inplace=True)
print(over_omic)
scaler = StandardScaler()
over_omic_data = scaler.fit_transform(over_omic)

dataset = tf.data.Dataset.from_tensor_slices((over_omic_data, over_omic_data))
dataset = dataset.shuffle(buffer_size=1024).batch(256, drop_remainder=False)
maxlen = over_omic_data.shape[1]
input_shape = (maxlen,)
inputs = layers.Input(shape=input_shape)
autoencoder = TransformerAutoencoder(maxlen, embed_dim, num_heads, ff_dim, num_blocks)
reconstructed, encoded = autoencoder(inputs)

model = keras.Model(inputs=inputs, outputs=reconstructed)
print(model.summary())
model.compile(optimizer="adam", loss="mse")
print(inputs.shape, encoded.shape, reconstructed.shape)
checkpoint = ModelCheckpoint("K:\data_analysis\CLIP\DNN\MODEL/ICC_transformer.model",
                             save_weights_only=False,
                             monitor='val_loss',
                             save_best_only=True, mode='min', verbose=1)
model.fit(dataset, validation_data=dataset, epochs=50, callbacks=[checkpoint])
encoder_model = autoencoder.get_encoder()
print(encoder_model.summary())

encoded_features = encoder_model.predict(dataset)
encoder_model.save("K:\data_analysis\CLIP\DNN\MODEL/ICC_encoder.model")
encoded_features = pd.DataFrame(encoded_features, index=list(over_omic.index))
encoded_features = encoded_features.round(4)
print(encoded_features.iloc[:5, :10])
encoded_features.to_csv("K:\data_analysis\CLIP\DNN\MODEL/ICC_omic_feature.txt",sep='\t')
