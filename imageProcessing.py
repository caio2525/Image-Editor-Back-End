import numpy as np
#import matplotlib.pyplot as plt
import cv2

def addGaussianNoise(img, sigma):
  mu = 0;
  noisyImage = img.copy()
  noiseSize = img.shape #size of the noise
  noise =  np.random.normal(mu, sigma, noiseSize)
  noisyImage = noisyImage + noise;
  return(noisyImage)

def add_noise(img, num, noise=0):
  #print(type(num))
  for k in range(num):
    i = np.random.randint(0, img.shape[0])
    j = np.random.randint(0, img.shape[1])
    img[i, j] = noise
  return(img)

def add_salt_pepper_noise(img, amount=0.05, salt_vs_pepper_proportion=0.5): #amount representa a proporção da imagem que será afetada pelo noise, neste caso 5% da image
  high_val = 255;
  low_val = 0;
  output = img.copy()

  num_salt = np.ceil((amount * salt_vs_pepper_proportion) * img.size)
  num_pepper = np.ceil((amount *  (1.0 - salt_vs_pepper_proportion)) * img.size)

  output = add_noise(output.copy(), int(num_salt), high_val )
  output = add_noise(output.copy(), int(num_pepper), noise=low_val)

  return(output)

def boxFiltro(img, kernelSize=0):
    blurredImg = cv2.blur(img, (kernelSize * 2 + 1, kernelSize * 2 + 1))
    return(blurredImg)

def gaussianFiltro(img, kernelSize=0, sigma=1):
    blurredImg = cv2.GaussianBlur(img,(kernelSize * 2 + 1, kernelSize * 2 + 1), sigma)
    return(blurredImg)

def medianFiltro(img, kernelSize=0):
    median = cv2.medianBlur(img, kernelSize*2+1)
    return(median)

def unsharpMascara(img, kernelSize=0):
  blurredImg = cv2.blur(img, (kernelSize * 2 + 1, kernelSize * 2 + 1))
  partialOutput = 2.0 * img
  output = partialOutput - 1.0 * blurredImg
  return (output)
