# Histogram-based Naive Bayes Recognition

In the previous sections 1-dimensional and multidimensional Color histograms and the generalisation towards multidimensional receptive field histograms, which count an kind of multidimensional features, have been introduced. Independent of the histogram-dimension and type of feature a simple probabilistic object recognition method, based on the **Naive Bayes Classifier**, has been introduced in {cite}`SchieleC00`. This method will be described below.

## Object Recognition in General

In Object Recognition the task is to determine which object $o_n \in \mathcal{O}$ is contained in a given input image $I$. Here, we assume the simple case, that only one object is prominent in each image. Before, the object recognition system is able to decide which object is contained in the image, the system must be trained. As in every other Machine Learning system, we must therefore distinguish

- Training phase
- Inference phase

of an object recognition system. Training requires a set of labeled training data, i.e. a set of images, which are labeled with the object, which is contained in the image. If $\mathcal{O}$ is the set of all objects, which appear in the training data, the training algorithm learns a model to distinguish objects from $\mathcal{O}$. This model is then applied in the inference phase in order to determine the object $o_n \in \mathcal{O}$ in a given input image.



## Naive Bayes Histogram-based Objectrecognition

The probabilistic object recognition as defined in {cite}`SchieleC00`, calculates the a-posteriori $p(o_n|R)$ by applying Bayes-rule:

$$
p(o_n|R) = \frac{p(R|o_n) \cdot p(o_n)}{P(R)}
$$ (bayesrule)

In the most simple (and not practical) setting, $R$ is just a measurement $m$ of the relevant features at a single pixel within the image $I$, where feature can be anything for which a one- or multi-dimensional histogram can be obtained, e.g. 

- single-channel pixel intensity
- multi-channel pixel intensity for color images
- arbitrary multidimensional receptive fields, such as gradient-magnitude, gradient-angle, etc. For the single measurement case equation {eq}`bayesrule` is  

$$
p(o_n|m) = \frac{p(m|o_n) \cdot p(o_n)}{P(m)} = \frac{p(m|o_n) \cdot p(o_n)}{\sum_i(p(m|o_i) \cdot p(o_i))},
$$ (bayesrulesingle)

where 

* $p(o_i)$ is the a-priori probability of object $o_i$
* $p(m|o_i)$ is the probability-density function of object $o_i$, which determines the probability, that measurement $m$ is obtained in an image with object $o_i$. 


Applying equation {eq}`bayesrulesingle` to calculate $p(o_n|m)$ requires that the a-priori probabilities $p(o_i)$ and the probability-density functions $p(m|o_i)$ are known. Where do they come from? They are estimated in the training-phase from training data:

1. The estimate for $p(o_i)$ is

    $$
    p(o_i)=\frac{N_i}{N},
    $$
    where $N$ is the total amount of labeled training images and $N_i$ is the number of training-images, which contain object $o_i$.
    
2. The estimate for $p(m|o_i)$ is actually the normalized one- or multidimensional histogram, which counts the frequency of feature-value-ranges or the frequency of joint features-value-ranges (in the multidimensional case), in the training-images, which belong to object $o_i$.

For example, assume that the we have only one feature (e.g. the pixel-intensities in a greyscale image) and all images, which belong to a certain object $o_n$, yield the following normalized histogram for this object. 

<img src="https://maucher.home.hdm-stuttgart.de/Pics/LikelihoodMeasureInObject.png" style="width:400px" align="center">

In this histogram only 8 different value-ranges (bins) are distinguished. From this histogram we can derive, for example
 - the probability that in an image, which contains object $o_n$, a value in the range of bin 0 is measured is 
 
     $$
     p(m \in bin_0|o_n) = 0.3
     $$
     
 - the probability that in an image, which contains object $o_n$, a value in the range of bin 4 is measured is
 
     $$ 
     p(m \in bin_4|o_n) = 0.05.
     $$
     
Hence, in the **training-phase** all a-priori probabilities $p(o_i)$ and all probability-density functions $p(m|o_i)$ are estimated from training-data. Then in the **inference-phase** equation {eq}`bayesrulesingle` is applied to determine the a-posteriori $p(o_n|m)$ for all objects $o_n$, and the object, which yields the largest $p(o_n|m)$ is the classification decision.


Now, we just have to go from the simple but unrealistic case, where only one measurement is taken in the image, to the practical case, where a set of $k$ measurements

$$
(m_1,m_2, \ldots, m_k)
$$

are obtained from the image, which shall be classified. In this case the Bayes-Rule is 

$$
p(o_n|m_1,m_2, \ldots, m_k) = \frac{p(m_1,m_2, \ldots, m_k|o_n) \cdot p(o_n)}{P(m_1,m_2, \ldots, m_k)} = \frac{p(m_1,m_2, \ldots, m_k|o_n) \cdot p(o_n)}{\sum_i(p(m_1,m_2, \ldots, m_k|o_i) \cdot p(o_i))},
$$ (bayesrulemulti)

With the **naive assumption** that the $k$ measurements are independent of each other, the joint conditional probability in the equation above can be represented as a product of conditional probabilities:

$$
p(o_n|m_1,m_2, \ldots, m_k) = \frac{ \prod_j p(m_j|o_n) \cdot p(o_n)}{\sum_i \prod_j p(m_j|o_i) \cdot p(o_i))}.
$$ (naivebayes)

This formula is calculated in the inference-phase in order to determine the most probable object $o_n$, given the set of $k$ measurements. The required a-priori probabilities $p(o_i)$ and probability-density functions $p(m|o_i)$ are the same as in the simple single-measurement case, described above.
     