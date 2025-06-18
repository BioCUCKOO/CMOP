# -*- coding: utf-8 -*-
# @Time    : 2024/12/8  19:51
# @Author  : Gou Yujie
# @File    : 三个癌迁移CLIP.py
import os
import pandas as pd
import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from sklearn.cluster import KMeans
from keras.models import load_model
from keras.regularizers import l2
import random
from keras import layers
from keras.layers import Dense,Flatten,Dropout,BatchNormalization
import subprocess
import tensorflow as tf
from keras.optimizers import Adam,SGD
from keras.callbacks import Callback,ModelCheckpoint,EarlyStopping
import torch
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from keras.utils import to_categorical
from keras.losses import MeanSquaredError,KLDivergence
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

class CLIPModel(nn.Module):
    def __init__(self, omics_dim, clinical_dim, embedding_dim):#调参
        super(CLIPModel, self).__init__()
        self.omics_encoder = nn.Sequential(
            nn.Linear(omics_dim, 128),
            nn.BatchNorm1d(128),  # 添加批归一化
            nn.ReLU(),
            # nn.Linear(128, 32),
            # nn.Sigmoid(),
            nn.Linear(128, embedding_dim)
        )
        self.apply(self._init_weights)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.BatchNorm1d(128),  # 添加批归一化
            nn.ReLU(),
            # nn.Linear(128, 32),
            # nn.Sigmoid(),
            nn.Linear(128, embedding_dim)
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)  # Xavier 初始化
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, omics, clinical):
        torch.set_printoptions(threshold=float('inf'))
        omics_features = self.omics_encoder(omics)
        clinical_features = self.clinical_encoder(clinical)
        # print('omics',omics_features,'clinical',clinical_features)
        return omics_features, clinical_features

def contrastive_loss(omics_features, clinical_features, temperature=0.5):
    batch_size = omics_features.shape[0]
    labels = torch.arange(batch_size).to(omics_features.device)

    # Normalize features
    omics_features = omics_features / omics_features.norm(dim=1, keepdim=True)
    clinical_features = clinical_features / clinical_features.norm(dim=1, keepdim=True)

    # Compute similarity
    logits = omics_features @ clinical_features.t() / temperature
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def train_clip_model(omics_data, clinical_data,savenamepath,embedding_dim=32, batch_size=1024, num_epochs=500, learning_rate=1e-4):
    if np.isnan(omics_data).any() or np.isnan(clinical_data).any():
        raise ValueError("Input data contains NaNs. Please preprocess your data to remove or impute NaNs.")

    omics_data= scaler.fit_transform(omics_data)
    clinical_data = scaler.fit_transform(clinical_data)
    omics_tensor = torch.FloatTensor(omics_data)
    clinical_tensor = torch.FloatTensor(clinical_data)

    dataset = TensorDataset(omics_tensor, clinical_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # basemodel = CLIPModel(omics_dim=128, clinical_dim=128, embedding_dim=64)
    model = CLIPModel(omics_data.shape[1], clinical_data.shape[1], embedding_dim)
    # basemodel = torch.load("K:\data_analysis\CLIP\CPTAC&TCGA\TCGA\CLIP/TCGA_CLIP.model")
    # # print(model)
    #
    # # # basemodel.load_state_dict()
    # #
    # # 获取模型 A 的参数
    # basemodel_state_dict = basemodel.state_dict()
    #
    # # 获取模型 B 的参数
    # model_state_dict = model.state_dict()
    #
    # # 迁移权重（跳过输入层）
    # for name, param in basemodel_state_dict.items():
    #     # 跳过输入层（omics_encoder.0.weight 和 clinical_encoder.0.weight）
    #     if "omics_encoder.0" in name or "clinical_encoder.0" in name:
    #         continue
    #     if name in model_state_dict:
    #         model_state_dict[name].copy_(param)
    # #
    # # # 将更新后的权重加载到模型 B
    # model.load_state_dict(model_state_dict)
    # # model.load_state_dict(basemodel_state_dict,strict=False)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0
        for omics_batch, clinical_batch in dataloader:
            optimizer.zero_grad()
            omics_features, clinical_features = model(omics_batch, clinical_batch)
            if torch.isnan(omics_features).any() or torch.isnan(clinical_features).any():
                raise ValueError(
                    "Model produced NaNs in the features. Check the model architecture and data preprocessing.")
            loss = contrastive_loss(omics_features, clinical_features)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += abs(loss.item())

        epoch_loss=total_loss / len(dataloader)
        print(epoch_loss)
        if epoch_loss < best_loss:
            torch.save(model, savenamepath)  # 保存完整模型
            best_loss = epoch_loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    model=torch.load(savenamepath)
    return model

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from sklearn.decomposition import PCA
def visualize_embeddings(omics_features, clinical_features,imagename,  labels=None):
    # 将两个嵌入矩阵合并
    omics_features = scaler.fit_transform(omics_features)
    clinical_features = scaler.fit_transform(clinical_features)
    # omics_features = omics_features[:2000, :]
    # clinical_features = clinical_features[:2000, :]
    # combined = torch.cat((omics_features, clinical_features), dim=0).cpu().detach().numpy()
    combined = np.vstack((omics_features, clinical_features))
    # combined = torch.cat((omics_features, clinical_features), dim=0).cpu().detach().numpy()

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=0)
    reduced = tsne.fit_transform(combined)
    # pca = PCA(n_components=2, random_state=42)
    # reduced = pca.fit_transform(combined)
    # 绘制
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced[:len(omics_features), 0], reduced[:len(omics_features), 1], label='Omics', alpha=0.5)
    plt.scatter(reduced[len(omics_features):, 0], reduced[len(omics_features):, 1], label='Clinical', alpha=0.5)

    if labels is not None:
        for i, label in enumerate(labels):
            plt.annotate(label, (reduced[i, 0], reduced[i, 1]))

    plt.legend()
    plt.title(imagename)
    plt.show()
def data_get_feat(omics_data,clinical_data,model):
    omics_data = scaler.fit_transform(omics_data)
    clinical_data = scaler.fit_transform(clinical_data)
    with torch.no_grad():
        omics_feature = model.omics_encoder(torch.FloatTensor(omics_data))
        clinical_feature = model.clinical_encoder(torch.FloatTensor(clinical_data))
    return omics_feature,clinical_feature

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# 1. 使用最近邻算法计算每个样本的最近邻
def compute_nearest_neighbors(data, k=5):
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nbrs.fit(data)
    distances, indices = nbrs.kneighbors(data)
    return indices


# 2. 计算 Jaccard 相似度
def jaccard_index(set1, set2):
    intersection = len(np.intersect1d(set1, set2))
    union = len(np.union1d(set1, set2))
    return intersection / union


# 计算交叉模态匹配矩阵 M
def compute_matching_matrix(K_ij, K_ji, n_samples):
    M = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            # 计算 Jaccard 相似度
            intersection_1 = np.intersect1d(K_ij[i], K_ji[j])
            union_1 = np.union1d(K_ij[i], K_ji[j])
            intersection_2 = np.intersect1d(K_ji[i], K_ij[j])
            union_2 = np.union1d(K_ji[i], K_ij[j])
            M[i, j] = len(np.union1d(intersection_1, intersection_2)) / len(np.union1d(union_1, union_2))

    return M


# 3. 归一化匹配矩阵
def normalize_matching_matrix(M):
    return normalize(M, axis=1, norm='l1')  # 按行归一化


# 4. 计算 Matching Score
def compute_matching_score(M, delta, n_samples):
    normalized_M = normalize_matching_matrix(M)
    MS = np.sum(normalized_M * delta) / n_samples
    return MS


os.chdir("K:\data_analysis\compare\CLIP/")
#
# fo="ICC"
# clinical_data = pd.read_table(fo+'/ICC_CF_processed.txt', index_col=0).astype(float)
# clinical_data.fillna(0, inplace=True)
# omics_data=pd.DataFrame()
# for omicname in ['RNA','protein','phos']:
#     one=pd.read_table(fo+'/ICC_%s.txt'%(omicname),index_col=0)
#     omics_data=pd.concat([omics_data,one],axis=1)
#
# omics_data=omics_data.astype(float)
# omics_data.replace(np.inf,100, inplace=True)
# omics_data.fillna(0, inplace=True)
# omics_data = omics_data.values
# clinical_data = clinical_data.values
# trained_model = train_clip_model(omics_data, clinical_data,'K:\data_analysis\compare\CLIP\ICC/clip_nopretrain.model')
# # trained_model=torch.load("K:\data_analysis\CLIP\CPTAC&TCGA\TCGA/temp\models/%s_CF.model"%(namelist[i]))
# trained_model.eval()
# omics_features, clinical_features = data_get_feat(omics_data, clinical_data, trained_model)
# torch.save(omics_features, fo+'/nopretrain_omic_feature.pt')
# torch.save(clinical_features, fo+'/nopretrain_clinic_feature.pt')
# visualize_embeddings(omics_features, clinical_features,'ICC_nopretrain')

# fo="CELL_SCLC/features/LUNG_"
# clinical_data = pd.read_table(fo+'clinic.txt', index_col=0).astype(float)
# clinical_data.fillna(0, inplace=True)
# omics_data=pd.DataFrame()
# for omicname in ['RNA','protein','phos','CNA']:
#     one=pd.read_table(fo+'%s.txt'%(omicname),index_col=0)
#     omics_data=pd.concat([omics_data,one],axis=1)
#
# omics_data=omics_data.astype(float)
# omics_data.replace(np.inf,100, inplace=True)
# omics_data.fillna(0, inplace=True)
# omics_data = omics_data.values
# clinical_data = clinical_data.values
# trained_model = train_clip_model(omics_data, clinical_data,'K:\data_analysis\compare\CLIP/CELL_SCLC/clipmodel/clip_transfer.model')
# # trained_model=torch.load("K:\data_analysis\CLIP\CPTAC&TCGA\TCGA/temp\models/%s_CF.model"%(namelist[i]))
# trained_model.eval()
# omics_features, clinical_features = data_get_feat(omics_data, clinical_data, trained_model)
# torch.save(omics_features, 'K:\data_analysis\compare\CLIP/CELL_SCLC/clipmodel/nopretrain_omic_feature.pt')
# torch.save(clinical_features, 'K:\data_analysis\compare\CLIP/CELL_SCLC/clipmodel/nopretrain_clinic_feature.pt')
# visualize_embeddings(omics_features, clinical_features,'SCLC_transfer')


# fo="CELL_EndometrialCarcinoma/features/ENDOMET_"
# clinical_data = pd.read_table(fo+'clinic.txt', index_col=0).astype(float)
# clinical_data.fillna(0, inplace=True)
# omics_data=pd.DataFrame()
# for omicname in ['protein','phos','CNV','methylation','miRNA']:
#     one=pd.read_table(fo+'%s.txt'%(omicname),index_col=0)
#     omics_data=pd.concat([omics_data,one],axis=1)
# omics_data=omics_data.astype(float)
# omics_data.replace(np.inf,100, inplace=True)
# omics_data.fillna(0, inplace=True)
# omics_data = omics_data.values
# clinical_data = clinical_data.values
# trained_model = train_clip_model(omics_data, clinical_data,'K:\data_analysis\compare\CLIP/CELL_EndometrialCarcinoma/clipmodel/clip_transfer.model')
# # trained_model=torch.load("K:\data_analysis\CLIP\CPTAC&TCGA\TCGA/temp\models/%s_CF.model"%(namelist[i]))
# trained_model.eval()
# omics_features, clinical_features = data_get_feat(omics_data, clinical_data, trained_model)
# torch.save(omics_features, 'K:\data_analysis\compare\CLIP/CELL_EndometrialCarcinoma/clipmodel/transfer_omic_feature.pt')
# torch.save(clinical_features, 'K:\data_analysis\compare\CLIP/CELL_EndometrialCarcinoma/clipmodel/transfer_clinic_feature.pt')
# visualize_embeddings(omics_features, clinical_features,'EndometrialCarcinoma_transfer')


# 19 0.29123527981113684
# 39 0.2822546964184216
# 59 0.29669112798388386
# 79 0.30174992763315217
# 99 0.29338861781472114
# omics_features=torch.load('K:\data_analysis\compare\CLIP/CELL_EndometrialCarcinoma/clipmodel/transfer_omic_feature.pt')
# clinical_features=torch.load('K:\data_analysis\compare\CLIP/CELL_EndometrialCarcinoma/clipmodel/transfer_clinic_feature.pt')

# 19 0.3082928767490267
# 39 0.32409823884283445
# 59 0.3148908409708708
# 79 0.3061474857741645
# 99 0.30763346849027257
# omics_features=torch.load('K:\data_analysis\compare\CLIP/CELL_EndometrialCarcinoma/clipmodel/nopretrain_omic_feature.pt')
# clinical_features=torch.load('K:\data_analysis\compare\CLIP/CELL_EndometrialCarcinoma/clipmodel/nopretrain_clinic_feature.pt')

# 21 0.30663809574606143
# 43 0.33049781382346216
# 65 0.30920294550830857
# 87 0.3260214719951627
# 109 0.31787256446536827
# omics_features=torch.load('K:\data_analysis\compare\CLIP/CELL_SCLC/clipmodel/nopretrain_omic_feature.pt')
# clinical_features=torch.load('K:\data_analysis\compare\CLIP/CELL_SCLC/clipmodel/nopretrain_clinic_feature.pt')

# omics_features=torch.load('ICC/nopretrain_omic_feature.pt')
# clinical_features=torch.load('ICC/nopretrain_clinic_feature.pt')
# torch.save(clinical_features, fo+'/nopretrain_clinic_feature.pt')
# omics_features=torch.load('K:\data_analysis\compare\CLIP/CELL_EndometrialCarcinoma/clipmodel/transfer_omic_feature.pt')
# clinical_features=torch.load('K:\data_analysis\compare\CLIP/CELL_EndometrialCarcinoma/clipmodel/transfer_clinic_feature.pt')
# visualize_embeddings(omics_features, clinical_features,'SCLC_transfer')
# omics_features=np.array(omics_features)
# clinical_features=np.array(clinical_features)
# omics_features= scaler.fit_transform(omics_features)
# clinical_features= scaler.fit_transform(clinical_features)
# # omics_features = [list(i) for i in list(np.array(omics_features))]
# # clinical_features = [list(i) for i in list(np.array(clinical_features))]
#
# num_samples = len(clinical_features)
# for i in range(5):
#     bas=(num_samples//5)*(i+1)
#     omics_data=omics_features[:bas]
#     clinical_data=clinical_features[:bas]
#     K_ij = compute_nearest_neighbors(omics_data, k=2)  # X 的最近邻
#     K_ji = compute_nearest_neighbors(clinical_data, k=2)  # Y 的最近邻
#     # print(K_ij,K_ji)
#     # 计算匹配矩阵 M
#     M = compute_matching_matrix(K_ij, K_ji,bas)
#     # 归一化匹配矩阵 M
#     normalized_M = normalize_matching_matrix(M)
#     # print(normalized_M.shape)
#     delta = np.zeros((bas, bas), dtype=int)
#     for i in range(bas):
#         delta[i, i] = 1
#     matching_score = compute_matching_score(normalized_M, delta,bas)
#     print(i, matching_score)

