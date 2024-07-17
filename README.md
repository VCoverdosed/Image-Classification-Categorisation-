# Image-Classification-Categorisation-
The   system   based   on   a   Bag-of-Words   feature   model   and   a   non-linear   SVM classifier

# Introduction
In the evolving field of computer vision, automating image classification remains pivotal across
diverse industries. This report explores an advanced approach by integrating the Bag-of-Words (BoW)
model with Support Vector Machines (SVM) for effective image classification.

Originally from natural language processing, the BoW model adapts to visual content by quantising
image features into a structured vocabulary. This transformation facilitates the use of SVM, known
for its proficiency with high-dimensional data and complex nonlinear decision boundaries.

Our methodology consists of key steps: feature extraction, clustering to build the vocabulary,
creating histogram representations of images, and classification via SVM. The report showcases the
training process and the approximate accuracy when tested on the given dataset when split by 9:1
ratio. Evaluations on some parameter adjustments will be covered 

# Methodology

## Sampling Keypoints

The **uniform\_keypoints** function systematically selects keypoints by dividing the image into samll grids, where each grid cell's size is determined by the desired number of keypoints(an input parameter). Within each grid cell, it selects the centroid as the keypoint, ensuring a uniform spatial distribution across the image. 

## Feature Extraction with SIFT
The **extract\_features** function encapsulates a complete workflow for feature extraction using the Scale-Invariant Feature Transform (SIFT) algorithm. The SIFT algorithm is designed to detect and describe local features in images that are invariant to scale, rotation, and partially invariant to changes in illumination and viewpoint.

The algorithm computes sampled keypoints' gradients in their neighbourhoods, then represents them with descriptors that capture the dominant orientations and gradient magnitudes. 

## Building Vocabulary
By aggregating SIFT descriptors obtained above, the **build\_vocabulary** function then employs K-means clustering to determine prominent visual patterns or "words." Here we have the number of clusters as an input parameter. The resulting vocabulary, composed of cluster centers, serves as a crucial tool for encoding images into a histogram of visual words.

## Histogram Construction
The **get\_hist** function maps the extracted SIFT descriptors of each image to the closest visual words in our pre-established vocabulary, this function constructs histograms that display the frequency of each visual word's occurrence in an image. These histograms, paired with their corresponding labels, provide a standardised input format that captures the essential visual characteristics of the images, which then get fed into our model training process. 

## One-vs-All SVM
Support Vector Machine (SVM) finds the optimal hyperplane that maximises the margin between different classes in the feature space. In this case, SVM is employed with a Radial Basis Function (RBF) kernel to handle the non-linear distribution of data, using a one-vs-all strategy to adapt its binary classification strength to a multi-class setting, thereby enabling it to classify images based on the histograms of visual words derived from the SIFT descriptors.

## Training Model
We train the model by feeding it the given 1500 images in 15 categories, dividing it into training and validation sets in a ratio of . Our supporting function **get\_image\_files\_and\_integer\_labels** identifies and sorts directory names within the base directory(folder named "train"), as each directory is assigned a unique integer label in the ascending case-insensitive alphabetical order, it iterates through each sub-folder, collecting image files that match the extensions we named, then appends each image's path and its corresponding integer label to separate lists. Now we obtain a structured dataset ready for further image processing tasks such as feature extraction and model training.  


## Testing Model
After training the model, we save the model and then test it using another unseen dataset, with the accuracy printing function at the end.

# Parameter Adjustment
In this section, we will plot the cross validation comparison graphs for our model with different sampling methods, different vocabulary sizes, and different kernels for SVM.

![fig1](https://github.com/user-attachments/assets/cf27ab7e-2f03-4656-a021-1e58b87ac25a)

**Fig 1.** here shows that using random sampling is not as good as uniform sampling, and is actually less accurate by a large fraction. This is a bit surprising, given that intuitively, randomisation can help with gaining a more stochastic, and potentially more generalised representation of an image, which might be beneficial in the image classification task where overfitting to particular patterns of features is a concern. 


<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/user-attachments/assets/27a089f8-69c3-4aaf-8b2a-ab9db4e532f0" alt="Fig.2" style="width: 45%;"/>
  <img src="https://github.com/user-attachments/assets/beb76142-57bd-4261-816b-688f2883841d" style="width: 45%;"/>
</div>

**Fig 2.** we compare different vocabulary sizes when setting input parameters for the number of clusters and the number of keypoints. As we can see, if we increase the number of clusters, the accuracy improves, which appears to be intuitive. If we increase the number of keypoints, it seems to bump up the accuracy when it is below 30, then fluctuates quite a lot once set above 30, between 0.54 and 0.63. 



![fig4](https://github.com/user-attachments/assets/f14cb1b0-12cc-46ff-8228-a394afabc80e)

In **Fig 3.** we can see that the SVM model with linear kernel omits lower accuracy than the SVM model with Gaussian kernel(RBF). This may be due to SIFT features typically displaying complex patterns that are not linearly separable. An RBF kernel effectively maps these features into a higher dimensional space, allowing for a linear decision boundary that can separate classes which are non-linearly separable in their original form, reflecting the inherent complexities of visual data. However, a more careful evaluation should be carried out to manage the balance between model complexity and overfitting. 

# Accuracy


