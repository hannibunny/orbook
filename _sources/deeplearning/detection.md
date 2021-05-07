# Object Detection

## Introduction

### Goal

The goal of object detection is to dermine for a given image:	 

* Which objects are in the image
* Where are these objects
* Confidence score of detection

<figure align="center">
  <img src="https://maucher.home.hdm-stuttgart.de/Pics/ILSVRC2014detection.PNG" style="width:600px" align="center">
  <figurecaption>Image Source: <a href="http://image-net.org/challenges/LSVRC/2014/index">http://image-net.org/challenges/LSVRC/2014/index</a></figurecaption>
</figure>

### Approaches

An old approach for obect detection is sliding window detection. Here a window is slided over the entire image. For each window a classifier determines, which object is contained in the window. In the most simple version only one object-type (e.g. car) is of interest and the binary classifier has to distinguish if this object is contained in the window or not. The corresponding position information provided implicitely by the position of the current window within the image. This approach is depicted below:


<figure align="center">
  <img src="https://maucher.home.hdm-stuttgart.de/Pics/slidingWindow.jpg" style="width:700px" align="center">  
<figcaption>
Sliding window approach for object detection 
</figcaption>
</figure> 

For this approach any supervised Machine-Learning algorithm can be applied, e.g. Support Vector Machines (SVM) or Mult-Layer-Perceptron. The content of a single window is either passed directly, in terms of pixel intensities, to the classifier or represented by other types of manually or automatically extracted features (see [section features](../features/globalDescriptors)). The sliding window approach is computationally expensive, since windows of different sizes and aspect ratios have to be shifted in a fine-granular manner over the entire image-space

Subject of this section is another approach for object detection: Deep Neural Networks, such as R-CNN and Yolo.

### Performance Measure

**Intersection over Union (IoU) and mean Average Precision (mAP)**

If $A$ is the set of detected pixels and $B$ is the set of Groundtruth-pixel, their IoU is defined to be
	
$$
IoU(A, B) = \frac{|A \cap B|}{|A \cup B|},
$$

where $|X|$ is the number of pixels in set $X$. Usually two bounding boxes (pixel sets) are said to **match**, if their IoU is $>0.5$. The output of the detector is correct, if

* detected object = groundtruth object, and
* detected bounding box and groundtruth bounding box match ( IoU is $>0.5$) 

Based on the correct and erroneous outputs the number of true positives (TP), true negatives (TN), false positives (FP), false negatives (FN) and the **recall** and **precision** can be determined. For a given class the **Average Precision (AP)** is related to the area under the precision-recall curve for a class:

$$
AP=\frac{1}{11} \sum\limits_{r \in \lbrace 0, 0.1, \ldots, 1 \rbrace} p_{interp}(r),
$$

where $p_{interp}(r)$ is the interpolated precision at recall $r$. The mean of these average individual-class-precisions is the **mean Average Precision (mAP)**. More info can be obtained from [PASCAL VOC Challenge](http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf)

## Region Proposals

Many Deep Neural Networks for Object Detection, e.g. R-CNN, require region proposals. The task of region proposal methods is to propose a list of bounding boxes within the image, which likely contain an object of interest. In order to determine these regions image-segmentation algorithms are applied, such as hierarchical clustering, mean-shift clustering or graph-based segmentation. The outputs of the applied segmentation process is often refined by methods such as selective search (see e.g. [here](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)).

## R-CNN 

R-CNN (Regions with CNN features) has been published 2014 in {cite}`GirshickDonahueDarrellEtAl`. It combines **Region Proposals** and **CNNs**. As mentioned above *Region Proposals* are candidate boxes, which likely contain an object. R-CNN does not require a specific region proposal algorithm. However, in the experiments of {cite}`GirshickDonahueDarrellEtAl` the authors applied [Selective Search](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)). 
	 
### Inference Process

The overall R-CNN process, as depicted in the image below, consists of the following steps:  
 
1. Apply **Selective Search** for calculating about 2000 region proposals for the given image.
2. Warp each region proposal to a fixed size, since the **CNN requires a fixed-size input**.
3. Pass each warped region proposal through the feature extractor-part of a CNN (modified AlexNet). The CNN-extractor provides for each region proposal a feature-vector of length 4096 to the following classifier. 
4. **Linear SVM-Classifier** calculates a probability for each class. and each region proposal
5. Given all scored regions in an image, a greedy non-maximum suppression is applied (for each class independently) that rejects a region if it has an IoU overlap with a higher scoring selected region larger than a learned threshold.
5. Since the region proposals are not accurate, **Bounding-Box Regression** is applied to compute accurate bounding boxes. Input are coordinates of the region proposal. Output are the coordinates of the groundtruth bounding-box.
 
<figure align="center"><img src="https://maucher.home.hdm-stuttgart.de/Pics/R-CNN.PNG" style="width:600px" align="center">
<figcaption>
	Image Source: Source: <a href="https://arxiv.org/pdf/1311.2524.pdf">https://arxiv.org/pdf/1311.2524.pdf</a>
</figcaption>
</figure> 


### Training

R-CNN consists of 3 modells, which are trained as described below.

**Training the CNN-Feature Extractor:**

* The CNN feature extractor is pretrained on the ILSVRC2012 imagenet data.
* The CNN feature extractor is fine-tuned with the data, given in the current task in order to adapt it to the relevant domain. For this the AlexNet classifier, which is designed to distinguish 1000 classes, is replaced by a classifier, which can either distinguish *21* classes in the case of PASCAL VOC 2010 data (20 classes + background), or *201* classes in the case of ILSVRC2013. 
* Each batch consists of 32 positives and 96 negatives. A region proposal is labeled *positive* if it's overlap with a ground-truth box is $IoU \geq 0.5$, otherwise background. 
 
**Training the SVM:**

* After training and fine-tuning the CNN-feature extractor, it can be applied to calculate the 4096-dimensional feature vector for each region proposal.
* The feature vectors are applied as the input of the SVM. For each class a binary linear SVM is learned.
* For training SVMs only the ground-truth boxes are taken as *positive* examples for their respective classes and proposals with less than 0.3 IoU overlap with all instances of a class are labeled as *negative* for that class.  


**Training the Bounding-Box-Regressor:**

After scoring each region proposal with a class-specific detection SVM, a new bounding box using a class-specific bounding-box regressor is predicted.

* Set of $N$ training pairs $\{(P^i,G^i)\}$, where

$$
P^i=(P^i_x,P^i_y,P^i_w,P^i_h) \quad \mbox{ and } \quad G^i=(G^i_x,G^i_y,G^i_w,G^i_h)
$$ 

specify the pixel coordinates of the center, width and height of the region proposal (P) and ground-truth (G) bounding-box, respectively. 

* The goal is to learn a transformation from $P$ to $G$ (index $i$ is omitted ). This transformation is parametrized as follows:

\begin{eqnarray}
\hat{G}_x & = & P_w d_x(P) + P_x \\
\hat{G}_y & = & P_h d_y(P) + P_y \\
\hat{G}_w & = & P_w \exp(d_w(P))\\
\hat{G}_h & = & P_h \exp(d_h(P))\\
\end{eqnarray}

where $\hat{G}$ is the regressors prediction for $G$. Each of the 4 function $d_*(P)$ is modeled as a linear function 

$$
d_*(P)= w_*^T \phi(P)
$$

of the output $\phi(P)$ of the last pooling layer of the CNN-extractor. The weight vectors $w_*^T$ are learned by Ridge Regression:

$$
w_* = \operatorname*{argmin}_{\hat{w}_*} \sum\limits_{i=1}^N (t^i - \hat{w}_* \phi(P^i))^2 + \lambda \mid\mid \hat{w}_* \mid\mid^2,
$$

where the components of the regression targets $t$ are 

\begin{eqnarray}
t_x & = & (G_x - P_x) / P_w \\
t_y & = & (G_y - P_y) / P_h \\
t_w & = & \log(G_w / P_w) \\
t_h & = & \log(G_h / P_h). \\
\end{eqnarray}


	
### R-CNN Performance and Drawbacks

In {cite}`GirshickDonahueDarrellEtAl` the following R-CNN performance figures were obtained:

* mAP (mean Average Precision) of $53.7\%$ on PASCAL VOC 2010- compared to $35.1\%$ of an approach, which uses the same region proposals, but spatial pyramid matching + BoW

* mAP of $33.4\%$ on ILSVRC 2013 Detection benchmark- compared to $24.3\%$ of the previous best result OverFeat {cite}`SermanetEigenZhangEtAl` 
  

<figure align="center"><img src="https://maucher.home.hdm-stuttgart.de/Pics/rcnnPerformanceVOC.png" style="width:600px" align="center">
	<figcaption>
		Image Source: Source: <a href="https://arxiv.org/pdf/1311.2524.pdf">https://arxiv.org/pdf/1311.2524.pdf</a>
	</figcaption>
</figure> 

The **drawbacks** of R-CNN are:

* 3 models must be trained
* Long training time (see picture below)
* About 2000 region proposals must by classified per image
* Selective Search for generating the region proposals is decoupled from R-CNN. No joint-fine tuning possible
* Inference time has been 47 seconds per image. Not suitable for real-time applications
* In particular for small regions the warping to a fixed size CNN-input is a \alert{waste} of computational resources and yields slow detection

<figure align="center"><img src="https://maucher.home.hdm-stuttgart.de/Pics/rcnnTime.png" style="width:500px" align="center">
	<figcaption>
		Image Source: Source: <a href="https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e">https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e</a>
	</figcaption>
</figure> 

## Spatial Pyramid Pooling (SPPnet)

Spatial Pyramid Pooling has been published 2015 in {cite}`HeZhangRenEtAl2014`. It integrates **Spatial Pyramid Pooling** ({cite}`Lazebnik06`) into CNNs. SPPnet is applicable for classification and detection: In the ILSVRC 2014 challenge it achieved rank 3 and rank 2 in in the classification- and detection task respectively. Same as R-CNN, it is also based on region proposals**. However,
 
* no warping to fixed CNN-input size is required. 
* it **passes each image only once through conv- and pool-layers of CNN**. 

SPPnet is $24-102$ times faster than R-CNN in the inference phase.

### Problem of fixed size input to CNN

<figure align="center"><img src="https://maucher.home.hdm-stuttgart.de/Pics/CNNpartitioned.png" style="width:500px" align="center">
</figure> 
 
 
In a CNN, as depicted above, convolutional- and pooling layer can easily cope with varying input-size. However, **fully connected layers require fixed-size input**, because for this type varying size means varying number of weights.  
 

### SPPnet allows variable size input

Maybe, the most important contribution of SPPnet is it's integration of a method, which is able to map an arbitray size output of the feature-extractor part to a fixed size input to the classifier part of a CNN.  

<figure align="center"><img src="https://maucher.home.hdm-stuttgart.de/Pics/sppNetFixedSize.PNG" style="width:450px" align="center">
	<figcaption>
		Image Source: Source: <a href="https://arxiv.org/pdf/1406.4729v4.pdf">https://arxiv.org/pdf/1406.4729v4.pdf</a>
	</figcaption>
</figure> 

Actually this mapping method to a fixed size classifier-input is the spatial pyramid pooling method as introduced long before in {cite}`Lazebnik06`) - see also [section object recognition](../recognition/objectrecognition). As sketched below, the arbitrary-sized feature maps at the output of the last convolutional layer are paritioned with an increasing granularity:

* on level 0 each feature map constitutes a single region
* on level 1 each feature map is partitioned into $2 \times 2$ subregions.
* on level 2 each feature map is partitioned into $3 \times 3$ subregions.
* on level 3 each feature map is partitioned into $6 \times 6$ subregions.

Within each region max-pooling is applied to compute one value per region. The number of all regions in the entire pyramid is therefore $1+4+9+36=50$. 
Since the number of all regions in the entire pyramid is independent of the size of the feature maps, the successive fully-connected layers always receive the same number of inputs. SPPnet, as defined in {cite}`HeZhangRenEtAl2014` provides 256 feature maps at the last convolution layer. Therefore the fixed-size input to the classifier has length $50*256 = 12800$. As in R-CNN

* a binary linear SVM is trained for each class (now the size of the SVM input-vectors is 12800)
* a bounding box regression model is learned.   


<figure align="center"><img src="https://maucher.home.hdm-stuttgart.de/Pics/sppNet.png" style="width:600px" align="center">
	<figcaption>
		Image Source: Source: <a href="https://arxiv.org/pdf/1406.4729v4.pdf">https://arxiv.org/pdf/1406.4729v4.pdf</a>
	</figcaption>
</figure> 

Note that with this approach, it is possible to pass the entire image only once through the CNN. Then the representation of each region proposal in the feature maps of the last conv-layer can be determined and the 256 feature maps are cropped to the size of the current region-proposal representation in this layer. Spatial pyramid max pooling, as described above, is then applied to the region proposal representation of the feature maps. 

Compared to R-CNN, SPPnet achieves a slightly better mAP, whilst beeing much faster. The drawbacks of SPPnet are, that still 3 models must be trained. Moreover, the convolutional layers that precede the spatial pyramid pooling can not be fine-tuned, because the gradients of the error function can not be passed efficiently through the Spatial Pyramid Pooling). I.e. **End-to-End training is not possible**.

## Fast R-CNN

Fast R-CNN has been published 2015 in {cite}`GirshickFastRCNN`. It constitutes an modification and extension of R-CNN, which is much faster in training and inference. Moreover, it achieves a higher detection quality (mAP) than R-CNN and SPPnet. It also applies region proposals (RoIs), but the image must be passed only once through the CNN. Another important improvement is that only one model must be trained using task-specific loss-functions. 
Fast R-CNN also requires less memory, because there is no need to store for about 2000 RoIs per image a corresponding feature vector of length 4096.



 

### Inference Process Overview

<figure align="center"><img src="https://maucher.home.hdm-stuttgart.de/Pics/fastRCNN.PNG" style="width:500px" align="center">
	<figcaption>
		Image Source: Source: <a href="https://arxiv.org/pdf/1504.08083.pdf">https://arxiv.org/pdf/1504.08083.pdf</a>
	</figcaption>
</figure> 


### RoI Pooling Layer

A RoI of size $h \times w$ at the last convolutional layer is partitioned into a $H \times W$-grid, where each region is approximately of size $h/H \times w/W$. Typical values: $W=H=7$. This is like level-1 partitioning in SPP. The features in each region are **max-pooled**, independently for each feature map.  
 
 
<figure align="center"><img src="https://maucher.home.hdm-stuttgart.de/Pics/roiPoolingCombined.png" style="width:550px" align="center">
	<figcaption>
		RoI Partitioning of a (5x7)-feature map into a (2x2)-grid and ROI-pooling. Image Source: Source: <a href="https://deepsense.ai/region-of-interest-pooling-explained/">https://deepsense.ai/region-of-interest-pooling-explained/</a>
	</figcaption>
</figure> 
 	

### Training

#### Apply pretrained CNNs

In {cite}`GirshickFastRCNN` three CNNs of different size are poposed. No matter which of them is applied in it's pretrained version, three transformation steps are required to integrate it in Fast R-CNN:
1. the last max pooling layer is replaced by a RoI pooling layer that is configured by setting $H$ and $W$ to be compatible with the net’s first fully connected layer.
2. the network’s last fully connected layer (which was trained for 1000 objects) is replaced with the two sibling layers: a fully connected layer for $K + 1$ categories and a category-specific bounding-box regressors.
3. the network is modified to take two data inputs: a list of images and a list of RoIs in those images.   




#### Hierarchical Sampling enables Fine-Tuning

 
 In R-CNN and SPPnet minibatches of size $N=128$ were constructed by **sampling one RoI from 128 different images**. Sampling multiple RoIs from a single image was supposed to be inadequate, because RoIs from the same image are correlated, causing slow training convergence. Since at the end of the feature extractor each RoI has a large receptive field (sometimes covering the entire image), fine-tuning of the Feature Extractor would be very expensive, if each element of the batch comes from a different image. Therefore, **the Feature Extractor has not been fine-tuned in R-CNN and SPPnet**. 
 
 In Fast-RCNN minibatches are sampled hierarchically:
	 
1. sample $N$ images ($N=2$), 
2. sample $R/N$ RoIs from each image ($R=128$). 

**Samples from the same image share computation and memory in Forward- and Backwardpass**. This yields a $64 \times $ decrease in fine-tuning time compared to sampling $R=128$ different images. This is why **fine-tuning the feature extractor is possible in Fast R-CNN**. The concern of slow convergence due to correlated samples within a minibatch has appeared to be not true in the research on Fast R-CNN.
	 

#### Multi-Task Loss

Fast R-CNN has **Two output-layers**:

* Softmax-Output with $K+1$ neurons for class-probabilities $p=(p_0,p_1,\ldots,p_K)$
* Bounding-Box regressors $t^k=(t^k_x,t^k_y,t^k_w,t^k_h)$ for each of the $K$ classes
 
Each training RoI is labeled with a groundtruth class $u$ and a groundtruth bounding-box regresssion target $v$. The multi-task loss $L$ on each labeled RoI for **jointly training classification and bounding-box regression**:

$$
L(p,u,t^u,v)=L_{cls} +\lambda [u \geq 1] L_{loc}(t^u,v),
$$

where 
* the $[u \geq 1]$-operator evaluates to 1 for $u \geq 1$ and $0$ otherwise (class-index $u=0$ indicates *no-known class*). 
* $L_{cls}$ is the log-loss function for true class $u$ and $L_{loc}$ is a $L_1$ loss-function between the elements of the predicted- and the groundtruth bounding-box quadruple (see {cite}`GirshickFastRCNN`).
 

 


