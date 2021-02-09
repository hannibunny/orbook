## Exercise
Calculate Sobel-derivatives of the image *smallimg* as defined in the next row. Calculate magnitude and direction of sobel gradient. At which position of the image appears the maximum magnitude of the gradient?

import numpy as np
from scipy import ndimage as ndi

smallimg=np.array([[1,5,6],[1,6,6],[6,6,1]])
smallimg

imx = np.zeros(smallimg.shape,dtype=np.float64)
ndi.filters.sobel(smallimg,1,imx,mode="nearest")
imy = np.zeros(smallimg.shape,dtype=np.float64)
ndi.filters.sobel(smallimg,0,imy,mode="nearest")
print("Gradient in x-direction")
print(imx)
print("Gradient in y-direction")
print(imy)

magnitude=np.sqrt(imx**2+imy**2)
print("Magnitude of Gradient:")
print(magnitude)

direction =np.arctan(imy/imx)
print("Direction of Gradient:")
print(direction)

