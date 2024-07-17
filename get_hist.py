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


# In[ ]:


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


# In[ ]:


def extract_features(image_path, num_keypoints):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints = random_keypoints(img, num_keypoints)
    keypoints, descriptors = sift.compute(img, keypoints)
    return descriptors


# In[1]:


def get_image_files_and_integer_labels(base_dir, image_extensions=['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
    # Sort folder names case-insensitively and create the mapping
    folders = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))], key=lambda x: x.lower())
    folder_to_int_label = {folder.lower(): i+1 for i, folder in enumerate(folders)}  # Mapping is lowercase

    image_files = []
    int_labels = []

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        for extension in image_extensions:
            for file in glob.glob(os.path.join(folder_path, '*' + extension)):
                image_files.append(file)
                # Assign labels using lowercase folder name for consistency
                int_labels.append(folder_to_int_label[folder.lower()])
    
    return image_files, int_labels


# In[2]:


def get_hist(image_files, int_labels, vocabulary,num_keypoints):
    histograms = []
    for image_file, label in zip(image_files, int_labels):
        img_features = extract_features(image_file,num_keypoints=num_keypoints)  # 提取图像特征
        hist = np.zeros(len(vocabulary))
        for feature in img_features:
            # 计算图像的直方图
            distances = np.linalg.norm(vocabulary - feature, axis=1)
            nearest_word_idx = np.argmin(distances)
            hist[nearest_word_idx] += 1
        histograms.append((hist, label))  # 存储直方图和对应的标签
    return histograms


# In[ ]:




