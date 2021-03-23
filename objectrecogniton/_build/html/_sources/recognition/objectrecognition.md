# Object Recognition

In this subsection conventional methods for object recognition are described. What is *conventional*? And which type of *object recognition*? In the context of this section *conventional* means *no deeplearning* and *object recognition* refers to the task, where given an image, the category of the object, which is predominant in this image must be determined. 
Even though *deep learning* approaches are excluded in this section, the evolution path of conventional techniques, as described here, actually lead to the development of deep neural networks for object recognition. Hence, understanding the evolution of conventional techniques also helps to understand the ideas and concepts of deep learning. Deep neural networks for object recognition are subject of all of the following sections.
 

## Bags of Visual Words

The goal of **feature engineering** is to find for a given input an informative feature representation, which can be passed as a numeric vector to a machine learning algorithm, such that this algorithm is able to learn a good model, e.g. an accurate image-classifier. One important and popular approach for this essential question is the so called **Bag of Visual Words** feature representation. This approach is actually borrowed from the field of *Natural Language Processing*, where *Bag of Word*-representations are a standard representations for texts. In this subsection we first describe the *Bag of Word*-model as it is applied in NLP. Then this model is transformed to the concept *Bag of Visual Words* for Object Recognition.

### The origin: Bag of Word Document Model in NLP

In NLP documents are typically described as vectors, which count the occurence of each word in the document. These vectors are called **Bag of Words (BoW)**. The BoW-representations of all text in a given corpus constitute the Document-Word matrix. Each document belongs to a single row in the matrix and each word, which is present at least once in the entire corpus, belongs to a single column in the matrix. The set of all word in the corpus is called the *vocabulary*.

For example, assume that there are only 2 documents in the corpus:

* **Document 1:** *cars, trucks and motorcyles pollute air*
* **Document 2:** *clean air for free*

The corresponding document-word-matrix of this corpus is:


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/BoW.png
---
width: 650px
align: center
name: BoW
---
Bag-of-Word representation of 2 simple documents
```

### From BoW to Bag of Visual Words

In contrast to NLP in Object Recognition the input are not texts but images. Moreover, the **input objects are not described by words but by local features**, such as e.g. SIFT features. Hence the *Bag of Visual Words*-model of an image, contains for each local feature the frequency of the feature in the image. However, this idea must be adapted slightly because there is an important distinction between words in a textcorpus and visual words in a collection of images: **In the text corpus there exists a finite number of different words, but the space of visual words in an image-collection is continuous with an infinite number of possible descriptors** (note that the local descriptors are float-vectors). Hence, the problem is *how to quantize the continuous space of local descriptors into a finite discrete set of clusters?* The answer is: *By applying a clustering-algorithms!* Such an algorithm, e.g. k-means clustering, calculates for a given set of vectors and for a user-defined number of clusters $k$, a partition into $k$ cells (=clusters), such that similar vectors are assigned to the same cluster. 


The centroids of the calculated $k$ clusters constitute the set of visual words, also called the **Visual Vocabulary:** 

\begin{equation}
V=\lbrace v_t \rbrace_{t=1}^K
\end{equation}

In the Bag-Of-Visual Word matrix, each of the $k$ columns corresponds to a visual word $v_t \in V$ and each image in the set ov $N$ images

\begin{equation}
I=\lbrace I_i \rbrace_{i=1}^N
\end{equation} 

corresponds to a row. The entry $N(t,i)$ in row $i$, column $t$ of the matrix counts the number of times visual word $v_t$ occurs in image $I_i$. Or more accurate: $N(t,i)$ is the number of local descriptors in image $i$, which are assigned to the $t.th$ cluster.  


$$
\begin{array}{c|ccccc}
 &  v_1  &  v_2  &  v_3  &  \cdots  &  v_K  \\ 
\hline 
 I_1  & 2 & 0 & 1 &   \cdots   & 0 \\ 
 I_2  & 0 & 0 & 3 &  \cdots  & 1 \\ 
 \vdots  &  \vdots  &  \vdots  &  \vdots  &  \vdots  &  \vdots  \\  
 I_N  & 2 & 2 & 0 &  \cdots  & 0 \\ 
\end{array} 
$$ (bovwmatrix)

Once images are described in this numeric matrix-form, any supervised or unsupervised machine learning algorithm can be applied, e.g. for content based image retrieval (CBIR) or object recognition. The main drawback of this representation is, that spatial information in the image is totally ignored. Bag-of-Visual-Word representation encode information about what type of structures (visual words) are contained in the image, but not where these structures appear.




```{figure} https://maucher.home.hdm-stuttgart.de/Pics/visualwordsSchema1.PNG
---
width: 650px
align: center
name: visualWords1
---
Calculation of visual words: First, from all images of the given collection, the local descriptors (e.g. SIFT) are extracted. These descriptors are points in the d-dimensional space (d=128 in the case of SIFT). The set of all descriptors is passed to a clustering algorithm, which partitions the d-dimensional space into a discrete set of k clusters. The center of a cluster (green points) is just the centroid of all descriptors, which belong to this cluster. Each cluster center constitutes a visual word. The set of all visual words is called visual vocabulary.
```



```{figure} https://maucher.home.hdm-stuttgart.de/Pics/visualwordsSchema2.PNG
---
width: 650px
align: center
name: visualWords2
---
Mapping of local descriptors to visual words: Once the visual words (cluster-centroids) are known, all local descriptors, can be mapped to their closest visual word. After this mapping for each visual word it's frequency in the given image can be determined and the vector of all these frequencies is the feature vector of the image.
```


The image below is from {cite}`Sivic03`. 


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/visualwordsPatches.PNG
---
width: 650px
align: center
name: visualWordPatches
---
For 4 different visual words, local image-regions are displayed, which are assigned to the same visual word. As can be seen local descriptors, which belong to the same visual word, actually describe similar local regions.
```


#### k-means clustering for determining the visual words

K-means clustering is maybe the most popular algorithm of **unsupervised machine-learning**. Due to it's simplicity and low complexity it can be found in many applications. One of it's main application categories is **quantization**. E.g. statistical quantizations of analog signals such as audio. In the context of this section k-means is applied for quantizing the continuous space of d-dimensional local descriptors.

The algorithm's flow chart is depicted below:


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/kmeansClustEnglish.png
---
width: 350px
align: center
name: kmeans
---
Flow chart of k-means clustering
```


**k-means clustering:**

1. First, the number of clusters $k$ must be defined by the user
2. Then $k$ initial cluster centers $\mathbf{v}_i$ are randomly placed into the d-dimensional space  
3. Next, each of the given vectors $\mathbf{x}_p \in T$ is assigned to it's closest center-vector $\mathbf{v}_i$:

	\begin{equation}
	\left\|\mathbf{x}_p-\mathbf{v}_i \right\| = \min\limits_{j \in 1 \ldots k } \left\|\mathbf{x}_p-\mathbf{v}_j \right\|,
	\end{equation}
	
	where $\left\| \mathbf{x}-\mathbf{v} \right\|$ is the distance between $\mathbf{x}$ and $\mathbf{v}$.
4. After this assignment for each cluster $C_i$ the new cluster centroid $\mathbf{v}_i$ is calculated as follows:

	\begin{equation}
	\mathbf{v}_i = \frac{1}{n_i} \sum\limits_{\forall \mathbf{x}_p \in C_i} \mathbf{x}_p,
	\end{equation}
	
	where $n_i$ is the number of vectors that are assigned to $C_i$. 
5. Continue with step 3 until the cluster-centers remain at a stable position.


This algorithm guarantees, that it's result is a local minimum of the **Reconstruction Error**, which is defined by
\begin{equation}
E=\sum\limits_{i=1}^K \sum\limits_{\forall \mathbf{x}_p \in C_i} \left\|\mathbf{x}_p-\mathbf{v}_i \right\|
\end{equation}

A nice demonstration of the k-means clustering algorithm is given here [k-means clustering demo](https://stanford.edu/class/engr108/visualizations/kmeans/kmeans.html).

Drawbacks of the algorithms are: 

* the number of clusters $k$ must be definied in advance by the user
* the algorithm terminates in a local minimum of the reconstruction error and different runs of the algorithms usually terminate in different local minimas, since the random placement of the initial cluster-centers differ.
* as usual in unsupervised learning the quality of the result can hardly be assesed. Quality can be assesed implicetly, if the cluster-result is applied in the context of a downstream supervised learning task, e.g. for image classification. 

Concerning these problems, the authors of *Visual categorization with bags of keypoints* ({cite}`Csurka04`) propose:

1. Iterate over a wide range of values for $k$
2. For each $k$ restart k-means 10 times to obtain 10 different vocabularies 
3. For each $k$ and each vocabulary train and test a multiclass classifier (e.g. Naive Bayes as will be described below)
4. Select $k$ and vocabulary, which yields lowest classification error rate.  



### Bag-Of-Visual-Words based image classification

As already mentioned, the Bag of Visual Words matrix, as given in equation {eq}`bovwmatrix`, can be passed as input to any machine learning algorithm. In the original work on visual words ({cite}`Csurka04`), the authors applied a Naive Bayes Classifier. This approach and the results obtained in {cite}`Csurka04` are described in this subsection. Note, that the Naive Bayes classifier has already been described in [Histogram-based Naive Bayes Object Recognition](../features/naiveBayesHistogram).

**Inference:**

Assume that a query image $I_q$ shall be assigned to one of $L$ object categories $C_1,C_2,\ldots,C_L$. For this a trained Naive Bayes Classifier calculates the **a-posteriori probability**

$$
P(C_j|I_q) = \frac{\prod_{t=1}^K P(v_t|C_j)^{N(t,q)} P(C_j)}{P(I_q)}
$$ (eq:NBfull)

for all classes $C_j$. The class which maximizes this expression is the most probable category.

The class which maximizes {eq}`eq:NBfull` is the same as the class which maximizes

$$
P^*(C_j|I_q) = \prod_{t=1}^K P(v_t|C_j)^{N(t,q)} P(C_j)
$$ (eq:NBred)

because the denominator in {eq}`eq:NBfull` is independent of the class. 

**Training:**

In order to calculate {eq}`eq:NBred` for all classes, the terms on the right hand side of this equation must be estimated for all classes and all visual words from a given set of $N$ training samples (pairs of images and corresponding class-label). 

The *a-priori probabilities* of the classes are estimated by

\begin{equation}
P(C_j)=\frac{|C_j|}{N},
\end{equation}  

for all classes $C_1,C_2,\ldots,C_L$.

For all visual words $v_t$ and all classes $C_j$ the **likelihood** is estimated by

\begin{equation}
P(v_t|C_j)=\frac{\sum\limits_{I_i \in C_j}N(t,i)}{\sum\limits_{s=1}^K \sum\limits_{I_i \in C_j}N(s,i)}
\end{equation}

In order to avoid likelihoods of value $0$ **Laplace Smoothing** is applied instead of the equation above:

\begin{equation}
P(v_t|C_j)=\frac{1+\sum\limits_{I_i \in C_j}N(t,i)}{K+\sum\limits_{s=1}^K \sum\limits_{I_i \in C_j}N(s,i)}
\end{equation} 

In {cite}`Csurka04` the authors applied a dataset of 1776 images of 7 different classes. 700 images thereof have been applied for test, the remaining for training. Below for each of the 7 classes a few examples are shown:



```{figure} https://maucher.home.hdm-stuttgart.de/Pics/csurkaDataset.PNG
---
width: 650px
align: center
name: datasetczurka
---
Some representatives for each of the 7 classes. As can be seen the intraclass-variance due to pose, view-point, scale, color, etc. is quite high. Moreover, there is a significant amount of background clutter
```

The resolution of the images varies from 0.3 to 2.1 megapixels. Only the luminance channel (greyscale-image) of the color-images has been applied for the classification task.

The confusion matrix and the the mean-rank[^footnote1] on the test-data is depicted in the image below:


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/csurkaNaiveBayesResult.PNG
---
width: 350px
align: center
name: confmat
---
Confusion matrix and mean rank on test data for Naive Bayes classification. k=1000 visual words have been applied
```


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/csurkaSVMResult.PNG
---
width: 350px
align: center
name: confmatsvm
---
Confusion matrix and mean rank on test data for SVM classification. k=1000 visual words have been applied
```

By applying a SVM the error rate dropped from $0.28$ in the case of Naive Bayes to $0.15$.

Concerning the question on *Which size of Visual Vocabulary $k$ is the best?* the following chart shows the error-rate decrease with increasing $k$ in the Naive Bayes classifier option (from {cite}`Csurka04`).


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/bowBestK.png
---
width: 350px
align: center
name: errorrateVsK
---
Reduction of error rate with increasing k in the Naive Bayes classification case
```

### Bag-Of-Visual-Words Design Choices

The work of Czurka et al ({cite}`Csurka04`) introduced the idea of visual words and demonstrated it's potential. Moreover, it motivated a wide range of research questions in it's context. These questions are sketched in the following subsections:

#### How to sample local descriptors from the image?

In [Local Fetures](../features/localFeatures) SIFT has been introduced. There, only the option that these descriptors are sampled at the image's keypoint (sparse sampling) has been mentioned. However, keypoint detection on on hand and describing a local region on the other hand are decoupled. In fact, as the image below shows, local descriptors can also be extracted at all points of a regular grid (dense sampling) or even at randomly determined points (random sampling). 



```{figure} https://maucher.home.hdm-stuttgart.de/Pics/localFeatureSampling.PNG
---
width: 550px
align: center
name: samplingOptions
---
Sparse sampling (left), dense sampling (center) and random sampling (right)
```

Random sampling is frequently the option of choice for *scene-detection*. The advantage of dense, and random sampling is that no effort must be spent for finding the keypoints.

#### Data for Training the Visual Vocabulary?



Visual words must be learned, e.g. by applying the k-means algorithm. Which image-datasets shall be applied for training? The best results are obtained, if the same training data is applied as for the downstream task (e.g. image classification). However, it is also possible to learn a set of generally applicable visual words, which can then be applied for different downstream tasks with different training data. This approach may be helpful, if only few labeled data is available for the downstream task - too few to learn a good visual vocabulary.


#### Number of Visual Words and how to count words in the matrix?

Up to now, we pretended that the entries in the Bag of Visual Words matrix are the numbers of local descriptors in the image of the current row, which are assigned to the visual word of the current columnt. However, using the visual-word frequencies is only one option. In BoWs in NLP an alternative to word-frequencies is the so called **tf-idf** (term frequency -inverse document frequency), where

* term frequency $tf_{i,j}$] counts how often word $i$ appears in document $j$, and

* inverse document frequency $idf_i$ is the logarithm of inverse ratio of documents (images) that contain (visual) word $i$:

	$$
	idf_i=\log\left(\frac{N}{n_i}\right).
	$$

	Here, $N$ is the number of documents, and $n_i$ is the number of documents, that contain word $i$.

* term frequency -inverse document frequency is then defined as follows:

	$$
	tf\mbox{-}idf_{i,j}=tf_{i,j} \cdot idf_i
	$$ 
	
The idea of tf-idf is that words, which occur in many documents, are less informative and are therefore weighted with a small value, whereas for rarely appearing words the weight $idf_i$ is higher. For example in the *Video-Google*-work {cite}`Sivic03`, the three matrix-entry options 

* binary count (1 if visual word appears at least once in the image, 0 otherwise)
* term-frequency count
* term frequency -inverse document frequency

have been compared. The best results have been obtained by tf-idf, the worst by the binary count.

The problem on the size of the visual vocabulary, i.e. the number of different clusters $k$, has already been mentioned above. As {numref}`errorrateVsK` shows, in general the error-rate of the downstream task (classification) decreases, with an increasing number of visual words. However, the error-rate-decrease gets smaller in the range of large $k$ - values. It also important to be aware, that this plot has been drawn for a specific task and a specific dataset. The question on the number of viusal words must be answered individually for each data set and each task.


#### Algorithms for Training and Encoding?  

In the context of visual vocabularies the two stages *training* and *encoding* must be distinguished. **Training** of a visual vocabulary means to apply a clustering algorithm to the given training-data in order to determine the set of visual words. **Encoding** means to apply the knwon visual vocabulary in the sense, that for new local-descriptors, the corresponding visual word is determined. Actually, training and encoding are decoupled, i.e. the method, used to encode is independent of the method applied for training. One interesting alternative for training visual vocabularies has been introduced in {cite}`Nister06`. In contrast to the standard k-means algorithm, this approach applies a hierarchical k-means clustering, which generates not a flat set of visual words, but a **vocabulary-tree.** The advantage of such an hierarchically ordered vocabulary is, that encoding, i.e. the assignment of new vectors to visual words, is much faster in this structure. As sketched in {numref}`hierClusttraining` the algorithm first partitions the entire d-dimensional space into a small number (= branching factor $k$) different subregions. In the next iteration each subregion is again partitioned into $k$ subregions and so on. This process is repeated until the maximum depth $L$ of the tree is reached. 

**Training: Hierarchical k-means clustering:**

1. Initialisation: Define branching-factor $k$ and maximum depth $L$. Set current depth $l=1$. 
2. Apply k-means algorithm to training-set in order to determine k different clusters in depth $l=1$.
3. Determine for each subregion in depth $l$ the subsetset of training-data, which is assigned to this subregion.
3. Apply k-means algorithm to each of the subregions in depth $l$, by applying the subregion-specific traning-subset. In this way new subregions for depth $l+1$ are generated. The umber of subregions in depth $l+1$ is $k$ times the number of subregions in depth $l$.
4. Set $l := l+1$ and continue with step 3 until $l=L$



```{figure} https://maucher.home.hdm-stuttgart.de/Pics/vocabTree1.png
---
width: 350px
align: center
name: hierClusttraining
---
Training: Hierarchical application of $k$-means clustering, for branching factor $k=3$.
```


**Encoding:**
In the encoding phase a new vector $p$ (point) must be assigned to it's closest cluster-center (visual word). In the hierearchical tree of visual words, in the first iteration the distance between $p$ and $k$ cluster-enters must be determined and the closest is selected. In the next iteration again only $k$ distances must be determined and compared- the distances between $p$ and the $k$ cluster centers of the subregion, which has been found in the previous iteration. In total, far fewer distances must calculated and compared, than in the encoding phase w.r.t. a flat set of visual words.


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/vocabTreeRecognition.PNG
---
width: 450px
align: center
name: hierClustencoding
---
Encoding: In the visual-word-tree new local descriptors must be evaluated in each iteration only w.r.t. $k$ different cluster-centers.
```

The plot below (source {cite}`Nister06`), depicts the performance increase with increasing number of visual words and increasing branching factor $k$.


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/vocabTreePerformance.PNG
---
width: 550px
align: center
name: hierClustPerformance
---
Accuracy vs. number of Leaf Nodes and vs. branching factor
```

In {cite}`Nister06` visual-vocabulary-trees have been applied for real-time recognition of CD-covers:

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/vocabTreeTest.PNG
---
width: 350px
align: center
name: hierClustTest
---
Real-time recognition of CD-covers.
```

There are much more options for implementing the training and encoding of visual words. For example instead of k-means or hierarchical k-means other unsupervised learning algorithms, such as e.g. *Restricted Boltzman Machines (RBM)* are applied. A comparison can be found e.g. in {cite}`Coates11`. Further down in this section the concept of *sparse coding* as an alternative to the *nearest-cluster-center-encoding* will be introduced.

### Bag of Visual Words: Summary of Pros and Cons

The Bag-of-Visual-Words (BoVW) concept has proven to be a good feature representation for robust object recognition (w.r.t. scale, background clutter, partial occlusion, translation and rotation). However, it suffers from a major drawback: The lack of spatial information. BoVW describe **what** is in the image but not **where**. Depending on the application this may be a bad waste of important information. In the next subsection *Spatial Pyramid Matching*, the most important approach to integrate spatial information and the idea of BoVW, is described. For better understanding the adaptations and extensions, {numref}`layersBoW` sketches the overall architecture of object recognition classifiers, that apply BoVW.


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/layerSchemaBoW.png
---
width: 550px
align: center
name: layersBoW
---
Given an input image, represented by its pixels, first a set of low level local features are extracted, e.g. SIFT features. Low-level features are vectors in a d-dimensional space. This space is quantized by a clustering algorithm, e.g. k-means. More concrete: Given a training-set of d-dimensional low-level features, the clustering algorithm calculates a set of $k$ cluster centers, which are the visual words. Once the visual words are known, any low-level feature vector can be assigned to a visual word and the visual words of all low-level features in the image constitute the BoVW-representation of the image. This BovW is also called the mid-level representation of the image. The mid-level representation of the image can be passed to any supervised learning algorithm for classification. This classifier is trained by many pairs of labeled input data.  
```



## Spatial Pyramid Matching
**Spatial Pyramid Matching (SPM)** has been introduced in {cite}`Lazebnik06`. It can be considered as combination of **Pyramid Match Kernels** and **BoVW**. It's main advantage is that it integrates spatial information into BoVW. The underlying idea is **Subdivide and Disorder**. 

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/localHist.png
---
width: 350px
align: center
name: subdivide
---
xxx
```


In this subsection, first the concept of *Pyramid Match Kernels*, as introduced in {cite}`Grauman07` will be described.





### Pyramid Match Kernel



```{figure} https://maucher.home.hdm-stuttgart.de/Pics/pyramidMatchConcept.png
---
width: 350px
align: center
name: subdivide
---
xxx
```



```{figure} https://maucher.home.hdm-stuttgart.de/Pics/pyramidMatchExample.png
---
width: 350px
align: center
name: xxx2
---
xxx
```



```{figure} https://maucher.home.hdm-stuttgart.de/Pics/spatialPyramidsToyExample.PNG
---
width: 350px
align: center
name: xxx2
---
xxx
```


### !This section and all the following sections are under construction!

[^footnote1]: **Mean Rank** is the mean position of the correct label, when labels are sorted in decreasing order w.r.t. the classifier score, as calculated in equation {eq}`eq:NBred`
