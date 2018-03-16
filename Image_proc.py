# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:38:47 2017

@author: ASUS
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

LOW_FILTER = np.array([[1/9,1/9,1/9],
                       [1/9,1/9,1/9],
                       [1/9,1/9,1/9]])
SOBELX_FILTER = np.array([[-1,0,1],
                          [-2,0,2],
                          [-1,0,1]])
SOBELY_FILTER = np.array([[1,2,1],
                          [0,0,0],
                          [-1,-2,-1]])

def convolve(img, flt):
    if len(img.shape)==2:                               #BW
        print('##Start convolving on BW image')
        start = time.time()
        h,w = img.shape                                 #get image dimension
        tmp = np.zeros((h+2, w+2), dtype=np.float32)    #get padded image
        tmp[1:-1,1:-1] = img[:,:]
        new_im = np.zeros_like(img, dtype = np.float32) #new image as output
        for i in range(1,h+1):                          #Iterating on padded image
            for j in range(1,w+1):
                new_im[i-1,j-1] = np.sum(tmp[i-1:i+2,j-1:j+2]*flt) #element wise multiplication
        finish = time.time() - start
        print('##Finish convolving after {} secs'.format(finish))
        return new_im
    else:                                                   #BGR
        print('##Start convolving on color image')
        start = time.time()
        h,w,d = img.shape                                   #get image dimension
        new_im = np.zeros_like(img, dtype = np.float32)     #new output image
        for k in range(d):                                  #iterate on each color channel
            tmp = np.zeros((h+2, w+2), dtype=np.float32)    #get padded image
            tmp[1:-1,1:-1] = img[:,:,k]
            for i in range(1,h+1):                          #iterate on padded image
                for j in range(1,w+1):
                    new_im[i-1,j-1,k] = np.sum(tmp[i-1:i+2,j-1:j+2]*flt)
        finish = time.time() - start
        print('##Finish convolving after {} secs'.format(finish))
        del tmp
        return new_im

def sobel_filter(img):
    print('#Start Sobel Filtering')
    start = time.time()
    new_img_x = convolve(img, SOBELX_FILTER)        #apply sobel x
    new_img_y = convolve(img, SOBELY_FILTER)        #apply sobel y
    new_img = np.zeros_like(img, dtype=np.float32)  #as output
    new_img = np.sqrt(new_img_x**2 + new_img_y**2)  #calc gradient
    #Normalize pixel value to range 0-255
    if len(img.shape)==2:
        new_img = (new_img - np.min(new_img))/(np.max(new_img)-np.min(new_img))
        new_img = new_img*255
        plt.subplot(121)
        plt.imshow(img, cmap = 'gray')
        plt.title('original image')             # show original image
        plt.subplot(122)                    
        plt.imshow(np.uint8(new_img), cmap = 'gray')
        plt.title('filtered image')      # show original image
        plt.show()
        
        
    else:
        for k in range(3):
            tmp = new_img[:,:,k]
            new_img[:,:,k] = (tmp - np.min(tmp))/(np.max(tmp) - np.min(tmp))
            del tmp
        new_img = new_img*255
        plt.subplot(121)
        plt.imshow(switch_channel(img))
        plt.title('original image')             # show original image
        plt.subplot(122)                    
        plt.imshow(switch_channel(np.uint8(new_img)))
        plt.title('filtered image')      # show original image
        plt.show()
       
    del new_img_x, new_img_y                #free up some memory
    finish = time.time() - start
    
    print('#Finish filtering after {} secs'.format(finish))
    return np.uint8(new_img)

def low_filter(img):
    print('#Start Low Filtering')
    start = time.time()
    #new_img = convolve(img, LOW_FILTER)
    new_img = np.zeros_like(img)
    if len(img.shape)==2:
        new_img = convolve(img, LOW_FILTER)     #apply 3x3 low filter
        plt.subplot(121)
        plt.imshow(img, cmap = 'gray')
        plt.title('original image')             # show original image
        plt.subplot(122)                    
        plt.imshow(new_img, cmap = 'gray')
        plt.title('filtered image')      # show processed image
        plt.show()
    else:
        print('###Filtering on each layer')
        for i in range(3):
            new_img[:,:,i] = convolve(img[:,:,i], LOW_FILTER)
        plt.subplot(121)
        plt.imshow(switch_channel(img))
        plt.title('original image')             # show original image
        plt.subplot(122)                    
        plt.imshow(switch_channel(new_img))
        plt.title('filtered image')      # show processed image
        plt.show()
    finish = time.time() - start
    
    print('#Finish filtering after {} secs'.format(finish))
    return np.uint8(new_img)

def getbw(img):         #average sum of all color channel
    return np.uint8(np.sum(img, axis=2)/3.0)

def switch_channel(img):    #swith BGR-RGB or RGB-BGR
    new_img = np.zeros_like(img)
    new_img[:,:,0] = img[:,:,2]
    new_img[:,:,1] = img[:,:,1]
    new_img[:,:,2] = img[:,:,0]
    return new_img
    
def gethist(img):       #get histogram of image
    h,w = img.shape     #get image dimension
    hist_img = [0.0 for i in range (256)]  #freq of each pixel value
    for i in range(h):
        for j in range(w):
            hist_img[img[i,j]]+=1
    return hist_img


def getcdfnorm(hist, img):  #get normalized cdf
    h,w = img.shape         #get image dimension
    cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))]) #cumulative sum
    return np.uint8(cdf/(w*h)*255)              #scale back to 0-255

def map_cdf(img, cdfnorm):  #apply cdf mapping to image
    new_img = np.zeros_like(img)
    h,w = img.shape
    for i in range(h):
        for j in range(w):
            new_img[i,j] = cdfnorm[img[i,j]]
    return new_img

def plothist(img):                  #plot histogram of image
    if len(img.shape)==2:           #BW image
        plt.plot(gethist(img), color = 'black')
        plt.legend(('BW'))
    else:                           #color image
        plt.plot(gethist(getbw(img)), color = 'black')  
        plt.plot(gethist(img[:,:,0]), color = 'b')      
        plt.plot(gethist(img[:,:,1]), color = 'g')
        plt.plot(gethist(img[:,:,2]), color = 'r')
        plt.legend(('BW','blue', 'green', 'red'))
    plt.xlim([0,256])
    plt.show()

def hist_eq(img):
    print('#Start Histogram Equalization')
    start = time.time()
    new_img = None
    if len(img.shape)==2:           #BW image
        hist = gethist(img)
        new_img = map_cdf(img, getcdfnorm(hist, img))
        plt.subplot(121)                # show original image
        plt.imshow(img, cmap = 'gray')
        plt.title('original image')
        plt.subplot(122)                 # show original image
        plt.imshow(new_img, cmap = 'gray')
        plt.title('hist. equalized image')
        plt.show()
        plothist(img)
        plothist(new_img)
        del hist
    else:                           #color image
        imgbw = getbw(img)
        hist_bw = gethist(imgbw)
        nhist_bw = getcdfnorm(hist_bw, imgbw)       #mapping from bw image
        new_img = np.zeros_like(img)
        for i in range(3):                          #apply mapping to each channel
            new_img[:,:,i] = map_cdf(img[:,:,i], nhist_bw)
        plt.subplot(121)
        plt.imshow(switch_channel(img))
        plt.title('original image')             # show original image
        plt.subplot(122)                    
        plt.imshow(switch_channel(new_img))
        plt.title('hist. equalized image')      # show processed original image
        plt.show()
        plothist(img)
        plothist(new_img)
        del imgbw, hist_bw, nhist_bw           #free memory space
    finish = time.time() - start
    print('#Finish Histrogram Equalizing after {} secs'.format(finish))
    return new_img


CUR = os.getcwd()
IMG_DIR = os.path.join(CUR, 'house.jpg')
img = cv2.imread(IMG_DIR)  ##add 0 to read as black-white
new_img = sobel_filter(img)