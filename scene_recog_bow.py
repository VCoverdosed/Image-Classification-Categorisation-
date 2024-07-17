#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
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


# In[2]:


from build_vocabulary import random_keypoints,extract_features,build_vocabulary
from get_hist import get_image_files_and_integer_labels,get_hist
from svm_classify import svm_classify


# In[4]:


def train(train_data_dir, num_clusters, num_keypoints):
    # 步骤1：构建词汇表
    vocabulary = build_vocabulary(train_data_dir, num_clusters=num_clusters,num_keypoints=num_keypoints)
    # 步骤2:获取图片文件路径并按字母顺序标签
    image_files, int_labels = get_image_files_and_integer_labels(train_data_dir)
    # 步骤3：获取视觉单词的直方图
    histograms = get_hist(image_files, int_labels, vocabulary,num_keypoints)
    # 步骤4:用一对多svm分类器训练
    ovr_classifier = svm_classify(histograms)
    with open('trained_svm.pkl', 'wb') as f:
        pickle.dump(ovr_classifier, f)
    return ovr_classifier,vocabulary


# In[5]:


train_data_dir = "/users/huangsicheng/Desktop/DSA5203/Assignment3/train"#write in train_data_dir


# In[6]:


trained_svm,vocabulary = train(train_data_dir, num_clusters=200, num_keypoints=256)


# In[7]:


with open('trained_svm.pkl', 'rb') as f:
    trained_svm_dir = pickle.load(f)


# In[8]:


test_data_dir= "/users/huangsicheng/Desktop/DSA5203/Assignment3/train"#write in test_data_dir


# In[9]:


def test(test_data_dir, trained_svm_dir, vocabulary,num_keypoints):
    # 获取图片文件路径并按字母顺序标签
    image_files, int_labels = get_image_files_and_integer_labels(test_data_dir)
    histograms = get_hist(image_files, int_labels, vocabulary,num_keypoints)
    # 将test_data得到的直方图与标签信息录入X_test与y_test
    X_test, y_test = zip(*histograms)
    # 使用训练好的模型进行预测
    y_predict = trained_svm_dir.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_predict)
    return accuracy


# In[10]:


accuracy = test(test_data_dir, trained_svm_dir, vocabulary,num_keypoints=256)


# In[11]:


print(accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




