# Multidimensional Receptive Field Histograms

Multidimensional receptive field histograms for object recognition have been introduced in {cite}`SchieleC00`. They can be understood as a **generalisation of the previously described color-histograms**. Recall that in color-histograms the frequency of pixel-values (or pixel-value-ranges) is determined. The corresponding distribution of pixel-intensity values is considered to constitute a relevant information, which is then passed to some object recognition algorithm (e.g. for CBIR). However, pixel-intensitiy is only one possible type of feature. There exist much more features, which can be relevant for a given task: For example, the gradient, it's magnitude and angle,... at each pixel. And for all of these features the corresponding joint frequency-distribution (histogram) can be determined. **Multidimensional Receptive Field Histograms** are multidimensional histograms, which count the frequency of feature-tuples in the image. 

The practical procedure is as follows:

1. Consider which features may be informative for your object recognition task, e.g. colour, gradients, low-frequencies, high frequencies, etc.

2. Apply filters to extract these features at each position $(x,y)$ in the image. 

   $$
   F(x,y) = \left( f_1(x,y), f_2(x,y), \ldots, f_Z(x,y) \right)
   $$

    is the feature-tuple at position $(x,y)$, which contains the $Z$ extracted features at this position.

3. Construct the Z-dimensional histogram by first defining bin-ranges for each of the $Z$ axes. Then count for each Bin the number of Z-tuples $F(x,y)$, which fall in this Bin. 

In the picture below the $Z=5$ features

- Partial derivation in x-direction $D_x$ of image $D$
- Partial derivation in y-direction $D_y$ of image $D$
- Partial derivation in y-direction $D_{xy}$ of $D_x$
- Partial derivation in x-direction $D_{xx}$ of $D_x$
- Partial derivation in y-direction $D_{yy}$ of $D_y$

are extracted at each pixel-position $(x,y)$. The corresponding feature tuple at the given positon is 

$$
   F(x,y) = \left( 1.2, -0.2, 0.5, 0.6,-1.3 \right)
$$
   


<img src="https://maucher.home.hdm-stuttgart.de/Pics/5dimHistogram.png" style="width:400px" align="center">


After calculating this feature-tuple at each position $(x,y)$, the corresponding 5-dimensional histogram can be determined.

Some commonly applied features and their properties are:

- **Gradient** $(D_x, D_y)$: Rotation variant; Can be applied to detect oriented structures, e.g. vertical lines.
- **Direction of Gradient** $Dir = \arctan\frac{D_y}{Dx}$: Rotation variant; Can be applied to detect oriented structures, e.g. vertical lines.
- **Magnitude of Gradient** $Mag = \sqrt{D_x^2+D_y^2}$: Rotation invariant.
- **Laplacian** $Lap=D_{xx}+D_{yy}$

In order to provide scale-invariant detection/recognition the features are extracted in more than one scale.



