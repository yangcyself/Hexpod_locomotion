# -^- coding:utf-8 -^-
from config import *
import os
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np

len_x = 6
len_y = 6

filedir = os.path.dirname(os.path.realpath(__file__))
filedir = os.path.dirname(filedir)
filedir = os.path.join(filedir,"scene")
filedir = os.path.join(filedir,"terrian.png")
img = mpimg.imread(filedir) 
img = img[:,:,:3]

mid_x = int(img.shape[0]/2)
mid_y = int(img.shape[1]/2)
scale_x = img.shape[0]/len_x
scale_y = img.shape[0]/len_y

# def heightMap(x,y):
#     return np.sum(img[int(mid_x + x*scale_x) ][int(mid_y-y*scale_y)])/3

def heightMap(x,y):
    return 0
    if(0.6<x<1.6 and -0.5<y<0.5):
        return 0.1
    return 0


#发现了heightmap和图片颜色的关系，是H = img[int(mid_x + x*100)][int(mid_y - y*100)]/3
