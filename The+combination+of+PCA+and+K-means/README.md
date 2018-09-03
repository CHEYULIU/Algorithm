
# The combination of PCA and K-means algorithm
---
What is PCA and K-means?  What kind of question that you need to use this two methods together?

### PCA and K-means
---
* PCA (Principal Components Analysis)  
  
In short, this is the method to lower the dimension of a data set. As we are dealing with a certain data set, sometimes we'll face a problem that we cannot explain the data due to the high dimension issue. For example, if you have a data set X in the space of 200,000 times 200. Each row represents a sample of data set, and each column stands for the feature of the data set. No matter whether you have a correspodent label Y or let's say dependent variable, you cannot easily explain the relationship between the Y and X dataset. Therefore, the reason we use the PCA is to lower the dimension of X dataset until we can explain how X affects Y.  
  

* K-means clustering
  
Clustering is the method to calculate the distance between each points. And each clustering methods contain different ways to calculate. In the  K-means method, it is the method to calculate the distance between each points in the dataset and its mean in each clusters. Before doing the calculation, it is necessary to classify the number of clusters in the beginning. After that, it can be calculated.  


### Problem  
---
* Q1: Accourding to the characteristic of Y data set to seperate the X data set (You can see Y as X's LABEL)  
* Q2: Doing the clustering algorithm in the seperated X data set and visualizing the result  
* Q3: According to the result of clustering, making a meaningful conclusion of it  

### Solution    
This question is design by someone else, but the solution of combination of PCA and K-means is come up with by myself. If you have another brilliant idea to solve the problem, please leave your comment.

---
* A1: Seperate the X data set into two classification based on the frequency of Y data set  

In order to find out the characteristic of Y data set, I drew the figure of Y data set to observe its frequency. After I did this, I found out that the Y data set is highly concentrated into its mean, which means the kurtosis of the data set is pretty high. Besides, the shape of its frequency looks like a bell. Given my observation, I think the Y data set and its correspondent X data set could be devided into two groups. One of them are highly concentrated Y data set with its X data set. The other one is the remaining.  

Specifically, I classified the Y data set within the range of its mean plus and minus one sigma as the first classification. The remaining Y data set and its X data set belongs to the second classification.

* A2: Use PCA to lower the dimension in order to visualize the result and choose the K-means as the clustering method  

Undoubtedly, there are several ways to do the clustering, but after I considered the essence of the question and the dimension of the X data set, I thought using the K-means as the method would be appropriate. But before we do that, it is necessary to lower the dimension of X data set so that it could be visualized. When it comes to lowering dimension, the first method popped up in my mind is Principal Component Analysis. By lowering the dimension to 2-D (or 3-D), we can further do the K-means clustering to observe each clusters' relationship.
  
* A3: Make a brief conclusion  

As we can see in the following, the result of each classification (the first classification with 4 clusters and the second classification with 2 clusters) demonstrates a similiar characteristic, which is that they are devided vertically along x-axis. Because the x-axis represents the first component and y-axis represents the second component, the cluster that is far away y-axis means that it it hard to be explained by the second component.  


![png](output_7_0.png)  

