# Comparative Analysis of Dimensionality Reduction Techniques with Clustering Algorithms on Fashion MNIST

This repository contains the implementation and results of a comparative study evaluating different dimensionality reduction techniques combined with clustering algorithms on the Fashion MNIST dataset.

[See paper](https://drive.google.com/file/d/1JosZcEmOQKx1z8h2V174_Y1foELeNcLG/view?usp=sharing)

<img src="/img/coverpage.jpg" alt= “” width="30%" style="display: block; margin: 0 auto">

## Table of Contents
- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
  - [Data Preprocessing](#data-preprocessing)
  - [Dimensionality Reduction Techniques](#dimensionality-reduction-techniques)
    - [PCA](#pca)
    - [SAE](#sae)
    - [UMAP](#umap)
  - [Clustering Algorithms](#clustering-algorithms)
- [Experimental Results](#experimental-results)
- [Conclusions](#conclusions)
  - [Main Findings](#main-findings)
  - [Recommendations](#recommendations)
- [References](#references)

## Introduction

### Problem Description
Clustering is an unsupervised, semi-supervised, and supervised method of classifying similar objects into clusters and different objects into other clusters. Cluster analysis includes various methods and algorithms for classifying similar objects into corresponding categories by recognizing similar patterns between them. The high dimensionality of data makes the process more demanding to obtain good results. However, researchers propose and develop dimensionality reduction techniques. These methods are used to improve the accuracy of feature learning, reduce training time, and decrease dimensions as a preprocessing step. This eliminates redundant elements or noise while maintaining the data's "state." This paper aims to comparatively analyze dimensionality reduction methods alongside clustering algorithms. The comparison is made using three clustering algorithms to consider their performance metrics. Fashion MNIST was chosen as the training database.

### Structure
First, the database, dimensionality reduction methods, and clustering algorithms are presented along with their respective parameter values. Then, the experimental results for each technique are provided, and the chapter concludes with a comparative analysis. Finally, the results are analyzed to determine if there is a better technique or combination of technique with clustering model based on specific performance metrics.

## Theoretical Background

### Data Preprocessing
The Fashion MNIST from Keras was used as the database. As shown in Figure 1, the database includes 60,000 images of dimensions 28x28 representing various types of clothing and apparel in black and white shades. This database contains 10 different types of clothing, which the clustering algorithms are called to cluster appropriately.

The database was divided into three subsets: the training subset, the validation subset, and finally, the test subset. The validation set is essential during the training process of neural network models. The split into subsets was done according to the division presented below in Table 1.

| Subset | Percentage |
|--------|------------|
| Train set | 77.14% |
| Validation set | 8.57% |
| Test set | 14.28% |

### Dimensionality Reduction Techniques

#### PCA
Starting with the PCA technique, this is a preprocessing task performed before applying any machine learning algorithm. The PCA technique transforms data into a lower-dimensional space using Singular Value Decomposition analysis. The first feature that describes the greatest variance is the first principal component and is placed in the first coordinate. Similarly, the second feature that describes the second greatest variance is the second principal component. In this way, 90% of the database can be described with only 2-3 principal components (Maćkiewicz and Ratajczak, 1993, p. 304).

For the PCA technique, the parameters n_components and whiten were determined, which affect the performance and how dimensionality reduction is implemented. Initially, the n_components parameter determines the percentage of the total variance of the data that we want to maintain after dimensionality reduction. In this parameter, a value of 0.99 was set, meaning that as many components are selected to maintain 99% of the variance. The whiten parameter multiplies the components by the square root of n_samples and then divides them by their eigenvalues. This makes the outputs uncorrelated and ensures that the dimensions have a comparable scale. Although the whiten parameter removes some information from the transformation, it can improve the predictive accuracy of models ("PCA," n.d.).

Before applying PCA, the data was normalized using the Standard Scale method ("Detecting Anomalies in Financial Data Using Machine Learning Algorithms," 2024, p. 12).

#### SAE
An autoencoder is a special type of neural network used to compress input data and reconstruct it from the compressed form. Autoencoders include two main parts: the Encoder and the Decoder. The Encoder is trained to reduce the dimensions of the input data and compress the information into a latent space. The Decoder takes over decoding, reconstructing the original data from the latent space. The latent space layer contains the most important features of the data.

For the SAE architecture, the following parameters were determined, which affect performance: the latent_dim with a value of 64, which represents the number of dimensions of the latent space. A small value of this parameter can lead to loss of important information, while a large value can retain redundant information, increasing the chances of successful reconstruction.

Regarding the layers and the number of neurons, the initial layer was set as Dense(256), followed by Dense(128). Many neurons increase computational complexity with the risk of overfitting, while few neurons reduce the model's learning capacity.

For the activation function, ReLU was selected in the intermediate layers, as it enhances non-linearity and facilitates learning complex relationships. In the final layer, the Sigmoid function was used, which limits the output values to the interval [0,1] and is suitable for normalized image data.

Finally, a Reshape function was added, which ensures that the final dimensions of the outputs match the original dimensions of the images.

During training, early stopping was applied, which terminates the training process before completing all epochs when performance on the validation set stops improving.

#### UMAP
The last dimensionality reduction technique applied is UMAP. The basic parameters selected are: n_neighbors, with a value of 15, which determines the number of neighbors taken into account for calculating the local structure of the data. Smaller values emphasize local relationships between data, while larger values capture the overall, general structure. The min_dist parameter, with a value of 0.1, controls the minimum distance between points in the low-dimensional space. Small values create more compact local clusters, while larger values lead to greater dispersion of the data. Finally, n_components, with a value of 2, determines the dimension of the final space, which is 2D (two-dimensional) in this case.

### Clustering Algorithms
After applying dimensionality reduction techniques, clustering algorithms were applied to evaluate the efficiency of each method. The clustering algorithms used are the following three:

For the MiniBatchKMeans clustering algorithm, the parameter n_clusters=10 was set, which defines the number of clusters. DBSCAN has two main parameters that largely determine the effectiveness of clustering. The parameter ε (eps) is the maximum distance between two points to be considered neighbors. The second parameter is min_samples and defines the minimum number of points required to form a dense area. To find the parameters, experimentation was done with different values, and it was found that, for normalized data, the value of eps=5 and min_samples equal to 6. For data from the PCA technique, it was found that eps=15 and min_samples=5 gives appropriate results. With the SAE technique, values with eps=7.65 and min_samples=7 showed 16% noise. Finally, for UMAP, eps=0.25 and min_samples=15 were found to be appropriate.

The last clustering algorithm is Agglomerative Clustering. Initially, the parameter n_clusters=10 was set to match the number of categories in the data. Then, for the linkage parameter, the four available ways of connecting clusters were examined: single, complete, average, and ward. After testing, it was found that ward provided the best results for all dimensionality reduction techniques.

## Experimental Results

The performance of the models was evaluated using several metrics:

- **Calinski-Harabasz Score**: Higher values indicate better defined clusters
- **Davies-Bouldin Score**: Lower values indicate better separation between clusters
- **Silhouette Score**: Higher values indicate better clustered data
- **V-Measure Score**: Higher values indicate better clustering performance compared to ground truth

## Conclusions

### Main Findings
In summary, the purpose of this document was to analyze and identify the best clustering models, in combination with different dimensionality reduction techniques, and to compare them with the case of "raw" data, without preprocessing. The fashion dataset and its division into subsets, which were used for the analysis, were mentioned. Additionally, a comparative commentary of the results for each technique and model was made, highlighting the best and worst performances.

From the results of the previous chapters, it was found that the PCA technique generally showed the worst performance. However, in Figure 10, it was observed that PCA recorded better scores than the other techniques on average, although this does not reflect the overall picture for all metrics. Comparing the SAE technique with the Raw data, it was found that SAE had similar performance to directly using the data, with overlapping results. Nevertheless, SAE performs better in terms of the Davies-Bouldin metric, achieving less dispersion and more compact results compared to Raw data.

The UMAP technique shows worse results only in the Davies-Bouldin metric, where it records the highest values compared to SAE and PCA. However, UMAP excels in the Silhouette, Calinski-Harabasz, and V-measure metrics, providing more compact and stable results. Moreover, it presents overall better results compared to SAE and Raw data, and is the best statistical technique according to the V-Measure metric. Therefore, UMAP appears to be the most effective technique, as it performs better or equally in most metrics and shows stability in the results.

Regarding the clustering models, it was calculated how many times each model showed the best performance in each metric. Agglomerative Clustering with SAE recorded the best results in the Calinski-Harabasz metric (Figure 4). DBSCAN, on the other hand, presented the best score in the Davies-Bouldin metric using raw data (Figure 5). Regarding the Silhouette Score metric, mini Batch K-Means achieved the best results using the UMAP technique (Figure 6), while, finally, mini Batch K-Means maintained the lead in the V-Measure metric (Figure 7). This demonstrates that the mini Batch K-Means model performs better than the rest, having high performance in two of the most important metrics.

In conclusion, UMAP combined with Agglomerative Clustering and mini Batch K-Means emerge as the most powerful combinations for clustering data, with UMAP generally offering the most consistent and robust performance.

### Recommendations
Further improvement can be achieved in the DBSCAN algorithm, as significantly low scores were observed compared to other models. These low performances are mainly attributed to the values of DBSCAN parameters, such as 'ε' (the maximum distance between two data points to be considered neighbors) and the minimum number of samples that determines the number of points to form a cluster. The correct selection of these parameters is often difficult and can have a significant impact on the algorithm's performance, as mentioned by Karami and Johansson (2014).

To improve DBSCAN's performance, the literature suggests various techniques such as using MVO (Galaxy-Based Search Algorithm). Specifically, this method utilizes galaxies with high expansion rates to direct the algorithm's parameters to areas that lead to better clustering (Lai et al., 2019) through white and black holes. This method could offer optimization in DBSCAN parameters and improve the algorithm's performance.

## References

- Maćkiewicz, A., & Ratajczak, W. (1993). Principal components analysis (PCA). Computers & Geosciences, 19(3), 303-342.
- Detecting Anomalies in Financial Data Using Machine Learning Algorithms. (2024), p. 12.
- Karami, A., & Johansson, R. (2014). Choosing DBSCAN Parameters Automatically using Differential Evolution. International Journal of Computer Applications, 91(7).
- Lai, J., Zhu, D., Ting, Y., & Cao, J. (2019). SGDBSCAN: A Galaxy-based DBSCAN Algorithm with Self-adapted Parameters. 2019 IEEE 4th International Conference on Cloud Computing and Big Data Analysis (ICCCBDA).
- Lee, C. (2024). Choosing the right number of principal components.
- Salem, A. M., & Hussein, A. A. A. (2019). Feature Selection for High Dimensional Data Classification. Menoufia Journal of Electronic Engineering Research, 28(1).
