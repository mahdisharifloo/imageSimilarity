# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

#-----------------------------------
# GLOBAL FEATURE EXTRACTION
#-----------------------------------

import numpy as np
import mahotas
import cv2


class Global_feature_extraction:

    bins = 8

    # feature-descriptor-1: Hu Moments
    def shape(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

    # feature-descriptor-2: Haralick Texture
    def texture(self,image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compute the haralick texture feature vector
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        # return the result
        return haralick

    # feature-descriptor-3: Color Histogram
    def color(self,image, mask=None):
        # convert the image to HSV color-space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # compute the color histogram
        hist  = cv2.calcHist([image], [0, 1, 2], None, [self.bins, self.bins, self.bins], [0, 256, 0, 256, 0, 256])
        # normalize the histogram
        cv2.normalize(hist, hist)
        # return the histogram
        return hist.flatten()

    def edge_detecetion(self,img):
        img = cv2.imread(img, 0)
        cv2.imwrite("canny.jpg", cv2.Canny(img, 200,300))
        
    def some_feature(self):
        """ mahdi : I have some features code in my laptop that can help us .
        so what if i add them in this place as method.
        """
        pass
    
        