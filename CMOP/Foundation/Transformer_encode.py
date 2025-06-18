# -*- coding: utf-8 -*-
# @Time    : 2024/11/6  15:59
# @Author  : Gou Yujie
# @File    : Transformer_encode.py
import os
import pandas as pd
import numpy as np
#先把32种疾病的同一个组学拼接在一起，然后各组学做transformer编码，最后把编码后的特征拼接到一起
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
        self.projection = layers.Dense(embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen

    def call(self, x):
        x = self.projection(x)
        return x
class TransformerAutoencoder(tf.keras.Model):
    def __init__(self, maxlen, embed_dim, num_heads, ff_dim, num_blocks):
        super(TransformerAutoencoder, self).__init__()
        self.embedding = TokenAndPositionEmbedding(maxlen, embed_dim)
        self.encoder = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_blocks)
        ]
        self.decoder = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_blocks)
        ]
        self.global_pooling = layers.GlobalAveragePooling1D()
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
        
        return reconstructed, encoded
    def get_encoder(self):
        inputs = layers.Input(shape=(self.embedding.maxlen,))
        _, encoded = self.call(inputs)
        encoder_model = keras.Model(inputs=inputs, outputs=encoded, name="encoder")
        return encoder_model


fopath="/public1/home/scb3720/TCGA/"
# fopath="K:\data_analysis\CLIP\CPTAC&TCGA\TCGA\GDCquery/"
os.chdir(fopath)
# 'snp','Somatic_Mutation'
#
filetypes=['RNA','CNV','methylation','miRNA','snp','Somatic_Mutation']
strategy = tf.distribute.MirroredStrategy()
embed_dim = 16
num_heads = 8
ff_dim = 512
num_blocks = 2
batch_size_per_replica = 32  # 根据CPU性能调整

for name in ['ACC','BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC',
              'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD',  'THCA', 'THYM', 'UCEC','UCS', 'UVM']:
    over = pd.DataFrame()
    if not os.path.exists("TCGA-"+name+'/SM_clinic.txt'):
        somatic=pd.read_table("TCGA-"+name+'/Somatic_Mutation_omic.txt',sep='\t',index_col=0)
        clinic=pd.read_table("TCGA-"+name+'/Somatic_Mutation_clinic.txt',sep='\t',index_col=0)
        clinic.drop_duplicates(subset=['id'],inplace=True)
        for id in clinic['id']:
            one=somatic[[col.split('.')[0] for col in somatic.columns if col.startswith(id)]]
            print(len(one.columns))
            one = one.loc[:, ~one.columns.duplicated()]
            one.dropna(subset=[id+'_Hugo_Symbol'],inplace=True)
            one.index=one[id+'_Hugo_Symbol']
            one.drop([id+'_Hugo_Symbol'],inplace=True,axis=1)
            one.drop([col for col in one.columns if 'non_cancer' in col], inplace=True, axis=1)
            one = one[~one.index.duplicated(keep='first')]
            over=pd.concat([over,one],axis=1,join='outer')
        clinic.to_csv("TCGA-"+name+'/SM_clinic.txt',sep='\t')
        over.to_csv("TCGA-"+name+'/SM_omic.txt',sep='\t')

    SNPframe=pd.DataFrame()
    CNVframe=pd.DataFrame()
    total = pd.read_table("TCGA-" + name + '/CNV_omic.txt', sep='\t', index_col=0)
    total.columns=[i.split('.')[0] for i in total.columns]
    snp=total[[col for col in total.columns if not col.endswith('_Copy_Number') and not col.endswith('_Major_Copy_Number') and not col.endswith('_Minor_Copy_Number')]]
    cnv=total[[col for col in total.columns if col.endswith('_Copy_Number') or col.endswith('_Major_Copy_Number') or col.endswith('_Minor_Copy_Number')]]
    clinic = pd.read_table("TCGA-" + name + '/CNV_clinic.txt', sep='\t', index_col=0)
    snp_clinic=clinic[clinic['id'].isin(snp.columns)]
    snp=snp[[i for i in clinic['id'] if i in snp.columns]]
    cnv_clinic = clinic[clinic['id'].isin([i.split('_')[0] for i in cnv.columns])]
    cnv = cnv.loc[:, ~cnv.columns.duplicated()]
    cnv_clinic = cnv_clinic.drop_duplicates(subset=['id'])
    print(snp.shape,snp_clinic.shape,cnv.shape,cnv_clinic.shape)
    cnv_clinic.to_csv("TCGA-"+name+'/cnv_clinic.txt',sep='\t')
    cnv.to_csv("TCGA-"+name+'/cnv_omic.txt',sep='\t')
    snp_clinic.to_csv("TCGA-"+name+'/snp_clinic.txt',sep='\t')
    snp.to_csv("TCGA-"+name+'/snp_omic.txt',sep='\t')

    methy=pd.read_table("TCGA-"+name+'/methylation_omic.txt',sep='\t',index_col=0)
    clinic=pd.read_table("TCGA-"+name+'/methylation_clinic.txt',sep='\t',index_col=0)
    print(methy.shape,clinic.shape)
    for col in snp.columns:
        if not col in list(snp_clinic['id']):
            print(col)
    print(snp_clinic['id'])

for data_type in filetypes:
    for name in ['BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC',
                  'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD',  'THCA', 'THYM', 'UCEC','UCS', 'UVM']:
        if not os.path.exists("TCGA-"+name+'/%s_omic.txt'%(data_type)):
            continue
        if os.path.exists("/public1/home/scb3720/TCGA/dataprocess/outfeat/%s_%s.txt"%(name,data_type)):
            continue
        omic=pd.read_table("TCGA-"+name+'/%s_omic.txt'%(data_type),sep='\t',index_col=0)
        clinic=pd.read_table("TCGA-"+name+'/%s_clinic.txt'%(data_type),sep='\t',index_col=0)
        clinic.drop_duplicates(subset=['id'],inplace=True)

        if data_type=='Somatic_Mutation':
            colneed=[]
            for colname in ['HGVSc','HGVSp_Short','t_depth','t_ref_count','t_alt_count','n_depth']:
                for col in omic.columns:
                    if colname in col:
                        colneed.append(col.split('.')[0])
            omic=omic[colneed]
            omic.dropna(axis=0,inplace=True,how='all')
            omic.fillna(0,inplace=True)
            omic = omic.loc[:, ~omic.columns.duplicated()]
            for col in omic.columns:
                if col.endswith('HGVSc'):
                    omic[col]=omic[col].astype(str)
                    omic[col]=omic[col].str.extract(r'.*\d+(.*)', expand=False)
                    unique_mutations = omic[col].dropna().unique()
                    mutation_map = {mutation: idx + 1 for idx, mutation in enumerate(unique_mutations)}
                    omic[col] = omic[col].map(mutation_map).fillna(0).astype(int)
                elif col.endswith('HGVSp_Short'):
                    omic[col]=omic[col].astype(str)
                    pattern = r'p\.([A-Z])\d+([A-Z*=])'
                    omic[['Letter_After_Dot', 'Char_After_Last_Digit']] = omic[col].str.extract(pattern, expand=True)
                    omic[col] = omic['Letter_After_Dot'] + '_' + omic['Char_After_Last_Digit']
                    omic.drop(['Letter_After_Dot','Char_After_Last_Digit'],axis=1,inplace=True)
                    unique_combinations = omic[col].dropna().unique()
                    mutation_map = {combination: idx + 1 for idx, combination in enumerate(unique_combinations)}
                    omic[col] = omic[col].map(mutation_map).fillna(0).astype(int)
        if data_type in ['RNA','CNV','miRNA','Somatic_Mutation']:
            over=pd.DataFrame()
            for codeid in clinic['id']:
                one=omic[[col for col in omic.columns if col.startswith(codeid)]]
                one = pd.melt(one)
                one = one[['value']].reset_index(drop=True)
                one.columns=[codeid]
                over=pd.concat([over,one],axis=1)
            omic=over    
        omic.dropna(axis=0,inplace=True,how='all')
        over_omic=omic.T
        over_omic.fillna(0,inplace=True)
        if over_omic.empty:
            continue
        print(over_omic.shape)        

        scaler = StandardScaler()
        over_omic = scaler.fit_transform(over_omic)

        dataset = tf.data.Dataset.from_tensor_slices((over_omic, over_omic))
        dataset = dataset.shuffle(buffer_size=1024).batch(256, drop_remainder=False)

        maxlen=over_omic.shape[1]
        input_shape = (maxlen,)
        inputs = layers.Input(shape=input_shape)
        autoencoder = TransformerAutoencoder(maxlen, embed_dim, num_heads, ff_dim, num_blocks)
        reconstructed,encoded = autoencoder(inputs)
        model = keras.Model(inputs=inputs, outputs=reconstructed)
        print(model.summary())
        model.compile(optimizer="adam", loss="mse")
        print(inputs.shape,encoded.shape,reconstructed.shape)
        checkpoint = ModelCheckpoint(f"/public1/home/scb3720/TCGA/dataprocess/outmodel/%s_%s.model"%(name,data_type), save_weights_only=False,
                                     monitor='val_loss',
                                     save_best_only=True, mode='min', verbose=1)
        model.fit(dataset,validation_data=dataset,epochs=20, callbacks=[checkpoint])
        encoder_model = autoencoder.get_encoder()
        print(encoder_model.summary())

        encoded_features=encoder_model.predict(dataset)
        encoded_features=pd.DataFrame(encoded_features,index=clinic['id'])
        encoded_features=encoded_features.round(4)
        print(encoded_features.iloc[:5,:10])
        encoded_features.to_csv("/public1/home/scb3720/TCGA/dataprocess/outfeat/%s_%s.txt"%(name,data_type),sep='\t')
