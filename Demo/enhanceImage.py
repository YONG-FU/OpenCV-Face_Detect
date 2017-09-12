import cv2
import numpy as np
import operator
from matplotlib import pyplot as plt
import os, errno
import glob, os


def SharpenImage(): 

    img = cv2.imread('image\\image14.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Original", img)

    #Create the identity filter, but with the 1 shifted to the right!
    kernel = np.zeros( (9,9), np.float32)
    kernel[4,4] = 2.0   #Identity, times two! 

    #Create a box filter:
    boxFilter = np.ones( (9,9), np.float32) / 81.0

    #Subtract the two:
    kernel = kernel - boxFilter

    #Note that we are subject to overflow and underflow here...but I believe that
    # filter2D clips top and bottom ranges on the output, plus you'd need a
    # very bright or very dark pixel surrounded by the opposite type.

    custom = cv2.filter2D(img, -1, kernel)
    cv2.imshow("Sharpen Image", custom)
    cv2.waitKey(0)

SharpenImage()