# Rectangular- and Gaussian Low Pass Filtering 

* Author: Johannes Maucher
* Last Update: 28th January 2021

In section [Basic Filter Operations](02filtering.ipynb) and section [Gaussian Filters](04gaussianDerivatives.ipynb) the average- and the Gaussian filter have been described, respectively. Both of them blur their input by calculating at each position $i$ a weighted sum of the values $f(i+u)$ in the neighbouring positions. 

\begin{equation}
g(i) = \sum\limits_{u=-k}^{u=k} h(u) f(i+u) 
\end{equation}

The difference is that for the the average filters the weights $h(u)$ are all the same, whereas in the Gaussian filter the weights $h(u)$ decrease for increasing absolute values of $u$. 

Both filters reduce variations in neighbouring values by replacing original values by neighbourhood-averages. Filters with this property are called **Low-Pass Filters (LP)**. In order to understand this term, the term of *frequency in the context of images* must be explained: **In image-processing the term of frequency refers to pixel-value variations within a neighbourhood**. If their are strong variations within neighbouring pixel-values we speak of high-frequencies, whereas in homogenous regions with nearly no variations the frequency is low. A Low-Pass filter suppresses high-frequencies, whereas low-frequencies are left unchanged. Correspondingly, a **High-Pass Filter (HP)** suppresses low-frequencies but leaves high-frequencies unchanged. Both, Low- and High-Pass Filters have a wide range of applications in image-processing, for example noise-reduction (LP) and contour-extraction (HP). Noise reduction is subject of the [next section](06GaussianNoiseReduction.ipynb). 

In the current section we will demonstrate and compare the properties of the average- and Gauss-Filter. For this demonstration we apply 1-dimensional signals and filters. The adaptation to the 2-dimensional case should be obvious.     

We will apply both filters to a signal, which is the sum of three sinusoidal signals of different frequencies

$$
y(t)=sin(2\pi \cdot 15 \cdot t) + sin(2\pi \cdot 30 \cdot t) + sin(2\pi \cdot 45 \cdot t)
$$

Again, the [scipy ndimage package](http://docs.scipy.org/doc/scipy-0.13.0/reference/ndimage.html) is applied for convolution filtering.

%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
import scipy as sci
import scipy.ndimage as ndi

import warnings
warnings.filterwarnings("ignore")

## Function for calculating and plotting the spectral representation of a signal
The function below calculates the single-sided amplitude spectrum of a 1-dimensional time domain signal $y$. The sampling frequency $Fs$ is required for a correct scaling of the frequency-axis in the plot.

def plotSpectrum(y,Fs,title=""):
 """
 Plots a Single-Sided Amplitude Spectrum of y(t)
 """
 n = len(y) # length of the signal
 k = np.arange(n)
 T = n/Fs
 frq = k/T # two sides frequency range
 frq = frq[range(int(n/2))] # one side frequency range

 Y = sci.fft(y)/n # fft computing and normalization
 Y = Y[range(int(n/2))]

 plt.stem(frq,abs(Y),'r') # plotting the spectrum
 plt.title(title)
 plt.xlabel('Freq (Hz)')
 plt.ylabel('|Y(freq)|')

## Define signal that shall be filtered

Fs = 200.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,1,Ts) # time vector for signal
ff = 15;   # lowest frequency in the signal
y = np.sin(2*np.pi*(ff)*t)+np.sin(2*np.pi*(2*ff)*t)+np.sin(2*np.pi*(3*ff)*t)
sigTitle="Accumulation of 3 sinusoidal signals"

## Rectangular (Average) Low Pass Filtering

tf=np.arange(-0.5,0.5,Ts) # time vector for filter
filt=np.zeros(len(tf))
print(len(tf))
filt[int(0.45*len(tf)):int(0.55*len(tf))]=1.0
filtTitle="Rectangular Filter"

### Plot signal, filter and filtered signal in time- and frequency domain

plt.figure(num=None, figsize=(16,12), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2,3,1)
plt.plot(t,y)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title(sigTitle)
plt.subplot(2,3,2)
plt.plot(tf,filt)
plt.title(filtTitle)
plt.xlabel('Time')
plt.ylabel('Amplitude')
ymargin=0.05
ymin,ymax=plt.ylim()
plt.ylim(ymin-ymargin,ymax+ymargin)

plt.subplot(2,3,4)
plotSpectrum(y,Fs,title="Spectrum of "+sigTitle)
plt.subplot(2,3,5)
plotSpectrum(filt,Fs,title="Spectrum of "+filtTitle)
plt.subplot(2,3,6)
fo1=ndi.convolve1d(y,filt, output=np.float64,mode="wrap")
plotSpectrum(fo1,Fs,title="Spectrum of Filtered Signal")
plt.subplot(2,3,3)
plt.plot(t,fo1)
plt.title("Filtered Signal")

## Gaussian Filter

sig=0.01
m=0.0
filt=np.exp(-((tf-m)/sig)**2/2)/(sig*np.sqrt(2*np.pi))
filtTitle="Gaussian Filter"

### Plot signal, filter and filtered signal in time- and frequency domain

plt.figure(num=None, figsize=(16,12), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2,3,1)
plt.plot(t,y)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title(sigTitle)
plt.subplot(2,3,2)
plt.plot(tf,filt)
plt.title(filtTitle)
plt.xlabel('Time')
plt.ylabel('Amplitude')
ymargin=0.05
ymin,ymax=plt.ylim()
plt.ylim(ymin-ymargin,ymax+ymargin)

plt.subplot(2,3,4)
plotSpectrum(y,Fs,title="Spectrum of "+sigTitle)
plt.subplot(2,3,5)
plotSpectrum(filt,Fs,title="Spectrum of "+filtTitle)
plt.subplot(2,3,6)
fo1=ndi.convolve1d(y,filt, output=np.float64,mode="wrap")
plotSpectrum(fo1,Fs,title="Spectrum of Filtered Signal")
plt.subplot(2,3,3)
plt.plot(t,fo1)
plt.title("Filtered Signal")

**Questions:** 

* Compare Gaussian- and Rectangular Low-Pass filtering
* How to configure the Gaussian filter such that higher frequencies are surpressed less?


