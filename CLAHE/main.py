#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import time

##CLAHE将超出阈值的灰度均分到所有像素值

if __name__ == '__main__':
    saveProcessedImage = False
    showOutput = True
    showTime = True
    testImagePat= "data/test1.jpg"
    img = cv2.imread(testImagePat)
    if showTime:
        start = time.time()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    outputImage = np.zeros(img.shape,np.uint8)
    for i in range(3):
        outputImage[:,:,i] = clahe.apply(img[:,:,i])
    if showTime:
        end = time.time()
        print (end-start)
    if saveProcessedImage:
        cv2.imwrite('data/CLAHE.jpg', outputImage)
    if showOutput:
        concaImage = np.concatenate([img, outputImage], axis=1)
        cv2.imshow('CLAHE result', concaImage)
        if cv2.waitKey(0) == 27: 
            cv2.destroyAllWindows()