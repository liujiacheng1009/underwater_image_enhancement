#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
import math
import numpy as np
import os
import time

def Dark_channel(img,r):
    win_size = 2*r + 1
    B,G,R = cv2.split(img)
    temp = cv2.min(cv2.min(B,G),R)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(win_size,win_size))
    dark = cv2.erode(temp,kernel)
    return dark

def AL_estimation(img,dark_channel):
    h,w = img.shape[:2]
    img_size = h*w
    num_pixel = int(max(math.floor(img_size/1000),1))
    
    img_temp = img.reshape(img_size,3)
    dark_temp = dark_channel.reshape(img_size,1)
    
    index = dark_temp[:,0].argsort()
    index_use = index[img_size-num_pixel:]
    
    AL_sum = np.zeros([1,3])
    for i in range(num_pixel):
        AL_sum = AL_sum + img_temp[index_use[i]]
        
    AL = AL_sum/num_pixel
    thread = np.array([[0.95,0.95,0.95]])
    A = cv2.min(AL,thread)
    return A

def Trans_estimation(img, A, r, omega):
    #omega = 0.95
    img_temp = np.empty(img.shape, img.dtype)
    for i in range(3):
        img_temp[:,:,i] = img[:,:,i]/A[0,i]
    trans = 1 - omega*Dark_channel(img_temp, r)
    return trans

def Guided_filter(I,p,r,eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r,r))
    corr_I = cv2.boxFilter(I*I, cv2.CV_64F, (r,r))
    corr_Ip = cv2.boxFilter(I*p, cv2.CV_64F, (r,r))
    
    var_I = corr_I - mean_I*mean_I
    cov_Ip = corr_Ip - mean_I*mean_p
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a*mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r,r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r,r))
    
    q = mean_a * I + mean_b
    
    return q

def supermaxmin(a, w):
    """
    # a: array to compute filter over
    # w: window width
    """
    maxpath, minpath = deque((0,)), deque((0,))
    lena = len(a)
    maxvalues = [None]*(lena-w+1)
    minvalues = [None]*(lena-w+1)
    for i in range(1, lena):
        if i >= w:
            maxvalues[i-w] = a[maxpath[0]]
            minvalues[i-w] = a[minpath[0]]
        if a[i] > a[i-1]:
            maxpath.pop()
            while maxpath:
                if a[i] <= a[maxpath[-1]]:
                    break
                maxpath.pop()
        else:
            minpath.pop()
            while minpath:
                if a[i] >= a[minpath[-1]]:
                    break
                minpath.pop()
        maxpath.append(i)
        minpath.append(i)
        if i == (w+maxfifo[0]):
            maxpath.popleft()
        elif i == (w + minpath[0]):
            minpath.popleft()
        maxvalues[lena-w] = a[maxpath[0]]
        minvalues[lena-w] = a[minpath[0]]
    
    return minvalues

def dehaze(img, r, n = 8, thre = 0.1, eps = 0.001, omega = 0.8):    
    #img_pro = img.astype('float64')/255
    img_pro = np.float64(img)/255
    J_dark = Dark_channel(img_pro, r)
    A = AL_estimation(img_pro, J_dark)
    t = Trans_estimation(img_pro, A, r, omega)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.float64(img_gray)/255
    t_ref = Guided_filter(img_gray,t,r*n,eps)
    
    t_thre = cv2.max(t_ref, thre)
    result = np.empty(img_pro.shape, img_pro.dtype)
    for i in range(3):
        result[:,:,i] = (img_pro[:,:,i]-A[0,i])/t_thre + A[0,i]
    result = cv2.normalize(result, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    return result

if __name__ == '__main__':
    saveProcessedImage = False
    showOutput = True
    showTime = True
    testImagePat= "data/test1.jpg"
    img = cv2.imread(testImagePat)
    if showTime:
        start = time.time()
    outputImage = dehaze(img, 5, n=8)
    if showTime:
        end = time.time()
        print (end-start)
    if saveProcessedImage:
        cv2.imwrite('data/darkChannelPrior.jpg', outputImage)
    if showOutput:
        concaImage = np.concatenate([img, outputImage], axis=1)
        cv2.imshow('dark channel prior result', concaImage)
        if cv2.waitKey(0) == 27: 
            cv2.destroyAllWindows()

