#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from collections import defaultdict


# In[5]:


def random_keypoints(image, num_keypoints):
    keypoints = []
    height, width = image.shape[:2]
    
    # 计算每个网格的大小
    grid_size_x = width // int(np.sqrt(num_keypoints))
    grid_size_y = height // int(np.sqrt(num_keypoints))
    
    # 在每个网格中选择一个点作为关键点
    for i in range(0, width, grid_size_x):
        for j in range(0, height, grid_size_y):
            x = i + grid_size_x // 2
            y = j + grid_size_y // 2
            keypoints.append(cv2.KeyPoint(x, y, 10))  # 10是关键点的尺度
            
            # 如果已经选择了足够多的关键点，则停止
            if len(keypoints) == num_keypoints:
                return keypoints
    
    return keypoints


# In[6]:


def extract_features(image_path, num_keypoints):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints = random_keypoints(img, num_keypoints)
    keypoints, descriptors = sift.compute(img, keypoints)
    return descriptors


# In[7]:


def build_vocabulary(train_data_dir, num_clusters, num_keypoints):
    # 定义提取特征的函数
    def extract_features_from_folder(folder_path, num_keypoints=num_keypoints):
        features = []
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img_features = extract_features(img_path,num_keypoints=num_keypoints)
            features.extend(img_features)
        return features
    
    # 遍历每个类别文件夹并提取特征
    all_features = []
    for class_name in os.listdir(train_data_dir):
        class_dir = os.path.join(train_data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_features = extract_features_from_folder(class_dir, num_keypoints=num_keypoints)
        all_features.extend(class_features)
    
    # 将提取的特征转换成特征矩阵
    features_matrix = np.array(all_features)
    
    # 应用K均值算法进行聚类
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    kmeans.fit(features_matrix)
    
    # 得到词典，即聚类中心
    vocabulary = kmeans.cluster_centers_
    
    return vocabulary


# In[ ]:




