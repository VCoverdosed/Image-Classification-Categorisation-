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


# In[1]:


def svm_classify(histograms):
    X_train, y_train = zip(*histograms)
    
    # 初始化高斯核支持向量机分类器
    svm_classifier = SVC(kernel='rbf')

    # 使用一对所有方法包装分类器
    ovr_classifier = OneVsRestClassifier(svm_classifier)

    # 在训练集上训练分类器
    ovr_classifier.fit(X_train, y_train)
    return ovr_classifier

