# Convolutional Neural Networks for Object Recognition

## ImageNet Contest

The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) evaluates algorithms for **object detection** and **image classification** at large scale. It contains
* **Detection challenge** on fully labeled data for 200 categories of objects

* **Image classification** challenge with 1000 categories

* Image classification plus **object localization** challenge with 1000 categories 

The Training data, provided for the challenge is a subset of [ImageNet data](https://imagenet.stanford.edu). ImageNet is an image database organized according to the WordNet hierarchy, in which each node of the hierarchy is depicted by hundreds and thousands of images. (15 million images in 22000 categories). 

---
<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/imageNetTaxonomy.PNG" style="width:500px" align="center">
</figure>

### Detection Challenge

In the detection challenge 200 object categories must be distinguished. Training data consists of 456567 images with 478807 objects. For validation  20121 images with 55502 objects from flickr and other search engines are provided. The test-dataset consists of 40152 images from flickr and other search engines. The average image resolution in validation data is: $482 \times 415$ pixels. For each image, the algorithms must produce a set of annotations $(c_i,b_i,s_i)$ of class labels $c_i$, bounding boxes $b_i$ and confidence scores $s_i$.  

<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/ILSVRC2014detection.PNG" style="width:500px" align="center">
</figure>

### Classification Challenge

In the classification challenge 1000 object categories are distinguished **Training data** consists of 1.2 Million images from ImageNet. For **validation** 50 000 images from flickr and other search engines are provided. The **test-dataset** consists of 100 000 images from flickr and other search engines. For each image the algorithms must produce $5$ class labels $c_i, \, i \in \lbrace1,\ldots,5\rbrace$ in decreasing order of confidence and 5 bounding boxes $b_i, \, i \in \lbrace1,\ldots,5\rbrace$, one for each class label. The **Top-5 error rate** is the fraction of test images for which the correct label is not among the five labels considered most probable by the model. 



In the **localisation-task** object-category and bounding boxes must match. Bounding boxes are defined to match, if they have an overlap (Intersection over Union) of $>50\%$.

## AlexNet

AlexNet is considered to be an important milestone in Deeplearning. This CNN has been introduced in {cite}`KrizhevskySutskeverHinton` and won the   **ILSVRC 2012 classification- and localisation task**. It achieved a top-5 classification error of $15.4\%$ (next best achieved $26.2\%$!). 

The key elements of AlexNet are:

* ReLu activations decrease training time
* Local Response Normalisation (LRN) in order to limit the outputs of the Relu activations
* Overlapped Pooling regions
* For reducing overfitting Data augmentation and Dropout are applied.
  * **Data Augmentation**: For training not only the $224 \times 224$ crop at the center, but several crops within the $256 \times 256$ image are extracted. Moreover, also their horizontal flips are applied and the intensities of the RGB channels have been altered to provide more robustness w.r.t. color changes. The augmented training dataset consisted of 15 million annotated images
  * **Dropout** is applied in the first 2 fully-connected-layers 
* Training duration 6 days on **two parallel GTX 580 3GB GPUs**

<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/AlexNetArchitecture.png" style="width:500px" align="center">
</figure>

<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/AlexNetSpec.PNG" style="width:500px" align="center">
</figure>

At the input AlexNet requires images of size $256 \times 256$. Since ImageNet consists of variable-resolution images, in the preprocessing the rectangular images are first rescaled such that shorter side is of length $256$.
Then the central $224 \times 224$ patch is cropped from the resulting image. 
In addition to this scaling, the only **preprocessing** routine is subtraction of the mean value over the training set from each pixel.  

In the **test-phase** from each test - image 10 versions werecreated:

* the four $224 \times 224$ corner crops and the center crop
* from each of the 5 crops the horizontal flip

For the 10 versions the corresponding output of the softmax-layer is calculated and the class which yields highest average output over the 10 versions is selected.
<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/AlexNetEvaluation.PNG" style="width:500px" align="center">
</figure>

## VGG Net

In {cite}`KarenSimonyan2014` the *Visual Geometry Group* of the Universitiy of Oxford introduced the VGG Net. The main goal of the researches was to investigate the influence of depth in CNNs. VGG Net won the ILSVRC 2014 localisation-challenge and became $2nd$ in the classification task. 

The architecture of VGG-13 (13 learnable layers) is shown below:
<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/vgg19.png" style="width:500px" align="center">
</figure>

All VGG versions are summarized in this table:

<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/vggConfigurations.png" style="width:500px" align="center">
</figure>

The number of parameters in these versions are millions:
  *  A, A-LRN: 133 millions
 *  B: 133 millions
*  C: 134 millions
*  D: 138 millions
*  E: 144 millions

As can be seen only version A-LRN contains Local Response Normalisation. In the experiments this version did not perform better than the version A without LRN. 

The key facts of the VGG-architecture are:

*  **$224 \times 224 $** RGB images at the input
  *  **Preprocessing:** Subtract mean-RGB image computed on trainings set.
 *  **Small receptive fields of $3 \times 3$** in filters, particularly in the first conv-layers.
*  **Stride:** $1$
*  **Padding:** $1$ for $3 \times 3$-filters
*  **$2 \times 2$-Max-Pooling** with stride 2
*  **ReLu** activation in all hidden layers
*  **No Normalization**
*  **Feature Maps:** 64 in the first layer and increasing by a factor of 2 after each pooling layer.

All VGG versions apply only filters of size $3 \times 3$. The apparent drawback of small filters may be that only small features in the input can be extracted. This assumption is not valid, if the local receptive fields of a sequence of layers is considered: A stack of two conv-layers with $3 \times 3$ receptive field has an effective receptive field of $5 \times 5$ and a stack of three conv-layers with $3 \times 3$ receptive field has an effective receptive field of $7 \times 7$. The advantages of a stack of layers with smaller filters are:

* Stacked version has more ReLu-nonlinearities, which enables more discriminative decision function

  * Stacked version has less parameters: 

      *  $3 \cdot 3^2C^2 = 27C^2$ for a 3-layer stack of $(3 \times 3)$-filters, versus ...
      *  $7^2C^2=49C^2$ for a single layer with  $(3 \times 3)$-filters

    where $C$ is the number of channels (feature maps in the layer). Less parameters impose better generalisation.  

The VGG experiments prove that it is better to apply more layers with small filters than less layers with larger filters. 

The images passed to the input of VGG are of size $224 \times 224$. In the training phase this input is cropped from a training image, which is isotropically rescaled. The smallest side of the rescaled training image is called **training scale $S$**. There exist two options for setting $S$:

---
 *  Use constant $S$ (**single-scale training}**, e.g. $S=256$ or $S=384$.
*  **multi-scale training:** randomly sample $S$ from a range e.g. $[256,512]$.

Results of single- and multiscale-testing are listed in the tables below:



<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/vggPerfSingleScale.png" style="width:500px" align="center">
  <figcaption>
Single scale testing
</figcaption>
</figure>



<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/vggPerfMultiScale.png" style="width:500px" align="center">
  <figcaption>
Multi scale testing
</figcaption>
</figure>



## ResNet

