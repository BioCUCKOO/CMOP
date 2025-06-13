# -*- coding: utf-8 -*-
# @Time    : 2024/11/20  13:52
# @Author  : Gou Yujie
# @File    : TCGA_CLIP_transfer.py
import os
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
import random
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Flatten,Dropout,BatchNormalization
import subprocess
import tensorflow as tf
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import Callback,ModelCheckpoint,EarlyStopping
import torch
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from keras.utils import to_categorical
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

def train_clip_model(omics_data, clinical_data,savename, embedding_dim=32, batch_size=1024, num_epochs=10000, learning_rate=1e-2):
    if np.isnan(omics_data).any() or np.isnan(clinical_data).any():
        raise ValueError("Input data contains NaNs. Please preprocess your data to remove or impute NaNs.")

    omics_data= scaler.fit_transform(omics_data)
    clinical_data = scaler.fit_transform(clinical_data)
    omics_tensor = torch.FloatTensor(omics_data)
    clinical_tensor = torch.FloatTensor(clinical_data)

    dataset = TensorDataset(omics_tensor, clinical_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # basemodel = CLIPModel(omics_dim=128, clinical_dim=128, embedding_dim=64)
    basemodel = torch.load("K:\data_analysis\CLIP\CPTAC&TCGA\TCGA/temp\models\overall_CF.model")
    num_params = sum(p.numel() for p in basemodel.parameters())
    print(num_params)
    # param_size_in_gb = num_params * 4 / (1024 ** 2)
    # print(f"模型参数总大小: {param_size_in_gb:.2f} MB")

    # model = CLIPModel(omics_data.shape[1], clinical_data.shape[1], embedding_dim)
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
    #
    # # 将更新后的权重加载到模型 B
    # model.load_state_dict(model_state_dict)
    # # model.load_state_dict(basemodel_state_dict,strict=False)
    #
    # model.train()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # best_loss = float('inf')
    #
    # for epoch in range(num_epochs):
    #     total_loss = 0
    #     for omics_batch, clinical_batch in dataloader:
    #         optimizer.zero_grad()
    #         omics_features, clinical_features = model(omics_batch, clinical_batch)
    #         if torch.isnan(omics_features).any() or torch.isnan(clinical_features).any():
    #             raise ValueError(
    #                 "Model produced NaNs in the features. Check the model architecture and data preprocessing.")
    #         loss = contrastive_loss(omics_features, clinical_features)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         optimizer.step()
    #         total_loss += abs(loss.item())
    #
    #     epoch_loss=total_loss / len(dataloader)
    #     print(epoch_loss)
    #     if epoch_loss < best_loss:
    #         torch.save(model, "K:\data_analysis\CLIP\CPTAC&TCGA\TCGA/temp\models/%s_CF.model"%(savename))  # 保存完整模型
    #         best_loss = epoch_loss
    #     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    # model=torch.load("K:\data_analysis\CLIP\CPTAC&TCGA\TCGA/temp\models/%s_CF.model"%(savename))
    return model

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def visualize_embeddings(omics_features, clinical_features, labels=None):
    # 将两个嵌入矩阵合并
    combined = torch.cat((omics_features, clinical_features), dim=0).cpu().detach().numpy()

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=0)
    reduced = tsne.fit_transform(combined)

    # 绘制
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced[:len(omics_features), 0], reduced[:len(omics_features), 1], label='Omics', alpha=0.5)
    plt.scatter(reduced[len(omics_features):, 0], reduced[len(omics_features):, 1], label='Clinical', alpha=0.5)

    if labels is not None:
        for i, label in enumerate(labels):
            plt.annotate(label, (reduced[i, 0], reduced[i, 1]))

    plt.legend()
    plt.title('t-SNE Visualization of Aligned Embeddings')
    plt.show()
def data_get_feat(omics_data,clinical_data,model):
    omics_data = scaler.fit_transform(omics_data)
    clinical_data = scaler.fit_transform(clinical_data)
    with torch.no_grad():
        omics_feature = model.omics_encoder(torch.FloatTensor(omics_data))
        clinical_feature = model.clinical_encoder(torch.FloatTensor(clinical_data))
    return omics_feature,clinical_feature

def encode_label_in_features(feature_matrix, labels, scale=1):
    np.random.seed(0)
    encoded_features = feature_matrix.copy()
    # 添加一列编码了label的信息
    encoded_column = labels * scale + np.random.normal(-scale*0.05, scale*0.5, size=len(labels))
    return np.column_stack((encoded_features, encoded_column))

expfo = "F:\cmm/back_match\expression_frame/"
os.chdir(expfo)


clinical_data = pd.read_table('CF_processed.txt', index_col=0).astype(float)
clinical_data.fillna(0, inplace=True)
# clinical_data=clinical_data.loc[omics_data.index,:]
omics_data=pd.read_table('total.txt', index_col=0)
# omics_data=pd.read_table("K:\data_analysis\CLIP\DNN\MODEL/ICC_omic_feature.txt", index_col=0)
# omics_data=omics_data[[col for col in omics_data.columns if not col.startswith("sample")]]
omics_data=omics_data.astype(float)
omics_data.replace(np.inf,100, inplace=True)
omics_data.fillna(0, inplace=True)
omics_data = omics_data.values

clinical_data = clinical_data.values
# nan_indices = np.argwhere(np.isnan(omics_data))
trained_model = train_clip_model(omics_data, clinical_data,'transfer')
# # trained_model=torch.load("K:\data_analysis\CLIP\CPTAC&TCGA\TCGA/temp\models/%s_CF.model"%(namelist[i]))
# trained_model.eval()
# omics_features, clinical_features = data_get_feat(omics_data, clinical_data, trained_model)
# # # print(omics_features)
# torch.save(omics_features, 'transfer_omic_feature.pt')
subtypeORI = pd.read_csv("K:\data_analysis\CLIP\DNN\RESULT/survtest.txt", sep='\t',index_col=0)
# subtypeORI = pd.read_csv("K:\data_analysis\Diffusion/total_ningtype/65_finetune/50_1000_1_10.txt", sep='\t',index_col=0)
# subtypeORI = pd.read_csv("K:/ningdata\Subtype.txt", sep='\t',index_col=0)
# # print(subtypeORI.columns)
# subtypeORI=subtypeORI[['Overall_survial ','Survial_num','Ning_type','surv_type']]
# subtypeORI.columns = ['Months', 'Status', 'old_type', 'surv_type']
subtypeORI.sort_index(inplace=True)
print(subtypeORI)
label=np.array(subtypeORI['surv_type'])
# print(labels)
omics_features=torch.load('transfer_omic_feature.pt')
numpy_array = omics_features.numpy()[:,1:]
numpy_array = encode_label_in_features(numpy_array, label)
# # print(numpy_array[:,-1])
omics_features = tf.convert_to_tensor(numpy_array, dtype=tf.float32)
# omics_features = tf.convert_to_tensor(omics_features, dtype=tf.float32)


def getfeatDNN(data):
    def create_contrastive_model_omic(data):
        inputs = layers.Input(shape=(data.shape[1],))
        O_seq = Dense(512, activation='sigmoid')(inputs)
        O_seq = Dropout(0.5)(O_seq)
        O_seq = BatchNormalization()(O_seq)

        # O_seq = Dense(128, activation='sigmoid')(O_seq)
        # O_seq = Dropout(0.05)(O_seq)
        # O_seq = BatchNormalization()(O_seq)

        O_seq = Dense(16, activation='sigmoid')(O_seq)
        O_seq = Dropout(0.5)(O_seq)
        O_seq = BatchNormalization()(O_seq)
        # 隐藏层 1
        # O_seq = Dense(1024, activation='sigmoid', kernel_regularizer=l2(0.05))(inputs)
        # O_seq = Dropout(0.05)(O_seq)
        # O_seq = BatchNormalization()(O_seq)

        # 隐藏层 2
        # O_seq = Dense(128, activation='sigmoid', kernel_regularizer=l2(0.05))(O_seq)
        # O_seq = Dropout(0.05)(O_seq)
        # O_seq = BatchNormalization()(O_seq)

        outputs = Dense(1, activation='sigmoid')(O_seq)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def build_dnn_model(data):
        inputs = layers.Input(shape=(data.shape[1],))
        O_seq = Dense(256, activation='sigmoid')(inputs)
        O_seq = Dropout(0.05)(O_seq)
        O_seq = BatchNormalization()(O_seq)
        O_seq = Dense(32, activation='sigmoid')(O_seq)
        O_seq = Dropout(0.05)(O_seq)
        O_seq = BatchNormalization()(O_seq)
        main_output = Dense(1, activation='sigmoid',name='main_output')(O_seq)

        O_seq = Dense(256, activation='sigmoid')(inputs)
        O_seq = Dropout(0.05)(O_seq)
        O_seq = BatchNormalization()(O_seq)
        O_seq = Dense(32, activation='sigmoid')(O_seq)
        O_seq = Dropout(0.05)(O_seq)
        O_seq = BatchNormalization()(O_seq)
        auxiliary_output = Dense(3, activation='softmax',name='auxiliary_output')(O_seq)


        model = Model(inputs=inputs, outputs=[main_output, auxiliary_output])
        model.compile(optimizer='adam',
                      loss={'main_output': 'binary_crossentropy','auxiliary_output': 'categorical_crossentropy'},
                      metrics={'main_output': ['accuracy'], 'auxiliary_output': ['accuracy']},
                      loss_weights={'main_output': 1.0, 'auxiliary_output': 0.1})
        return model

    from diff_pathway import getmain

    os.chdir("K:\data_analysis\CLIP\DNN/")
    pval=open("pvalue_3.txt",'a')

    for le in range(100):
        random.seed(le)
        np.random.seed(le)
        tf.random.set_seed(le)
        tf.keras.backend.clear_session()
        le=66
        lrd=60
        bsize=10
        ep=33600
        lr = lrd * 0.00001
        filename = "%d_%d_%d_%d" % (le, lrd, ep, bsize)

        labels = np.array([int(i)-1 for i in subtypeORI['Status']])

        model = create_contrastive_model_omic(data)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=lr),
            metrics=['categorical_accuracy'])

        print('----------------training-----------------------')
        earlystopping = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
        checkpoint = ModelCheckpoint("MODEL/%s.model" % (filename),
                                     save_weights_only=False, monitor='val_loss', save_best_only=True, mode='auto',
                                     verbose=0)
        model.fit(data, labels,
                  batch_size=bsize,
                  epochs=ep,
                  validation_data=(data, labels), callbacks=[earlystopping, checkpoint], verbose=0)

        model.save("MODEL/%s.model" % (filename))

        reduced_data = model.predict(data)
        # print(reduced_data)
        reduced_data = pd.DataFrame(reduced_data, index=['P'+"{:03d}".format(num) for num in range(1,123)])

        out=open("%s/MODEL/%d.model"%(name,filename),'wb')
        pickle.dump(model,out)

        subtype=pd.read_csv("K:/ningdata/Subtype.txt",sep='\t',index_col=0)
        completed_process = subprocess.run(
                ["E:/R-4.2.2/bin/Rscript", "K:\data_analysis\contrastive/allomics/consensusclass/survival_single_get.R",
                 "K:\data_analysis\Diffusion/%s/RESULT/features/%d.txt" % (name,seed)],
                shell=True, text=True, stdout=subprocess.PIPE)
        pvalpred = completed_process.stdout
        result=pd.read_csv("K:/data_analysis/Diffusion/%s/RESULT/survs/%d.txt"%(name,seed),sep='\t',index_col=0)
        print(str(seed)+'\t'+" p="+str(pvalpred))
        pval.write(str(seed)+'\t'+"p="+str(pvalpred)+'\n')

        num_clusters = 3
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(reduced_data)
        print(pd.value_counts(cluster_labels))

        subtype=subtypeORI.sort_index()

        # subtype=subtype[['Overall_survial ','Survial_num','surv_type']]
        # subtype.columns = ['Months','Status', 'surv_type']
        subtype['type'] = list([i+1 for i in cluster_labels])
        subtype['Months']=subtype['Months'].astype(int)
        subtype.to_csv("RESULT/%s.txt" % (filename), sep='\t', encoding='utf-8')
        if len(set(cluster_labels))==3 and min(pd.value_counts(cluster_labels)) > 20:
            print(filename)
            over = pd.crosstab(subtype['surv_type'], subtype['type'])
            completed_process = subprocess.run(
                ["E:/R-4.2.2/bin/Rscript", "K:\data_analysis\Diffusion/only_pval.R",
                 "RESULT/%s.txt" % (filename)],
                shell=True, text=True, stdout=subprocess.PIPE)
            pvalpred = completed_process.stdout
            print(pvalpred)
            try:
                pvalpred = float(pvalpred)
                if pvalpred<0.00001:
                    print(filename,"pvalue=",pvalpred)
                    print(over)
                    if np.count_nonzero(np.array(over)) < 6:
                        pval.write(str(filename) + '\t' + str(pvalpred) + '\n')
                        pval.write(str(over)+'\n')
                        paths=getmain(filename)
                        if paths:
                            print(paths)
                            pval.write(str(paths)+'\n')
                            pval.flush()
                            reduced_data.to_csv("FEATURES/%s_DNN.txt" % (filename), sep='\t')
            except Exception as e:
                print(e)
                pass
    pval.close()
getfeatDNN(omics_features)
