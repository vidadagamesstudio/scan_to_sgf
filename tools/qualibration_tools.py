"""
permit to qualibrate program
"""

import cv2
import numpy as np
import math

from config import (
    DEBUG,
    DEBUG_AUTOTUNE_PATH,
)

from tools.geometry_tools import (find_circle)


def find_radius(path,radius_margin=2):
    """
    guess stone radius
    """
    img_ori = cv2.imread(path) #binary image
    img = cv2.imread(path,0) #binary image


    ret, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    thresh=cv2.bitwise_not(thresh)
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    listArea = []
    list_boundingRect =[]
    for c in cnts:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if ar >= 0.95 and ar <= 1.05:
                list_boundingRect.append((x, y, w, h))


    median_cnt = np.median(list_boundingRect,0)
    rows = thresh.shape[0]

    radius = (median_cnt[2]+median_cnt[3])/4
    
    return math.floor(radius-radius_margin), math.ceil(radius+radius_margin)
