"""
permit to qualibrate program
"""
import cv2 as cv
import numpy as np

def find_radius(bw_img) -> int:
    """
    guess stone radius with median radius
    """

    thresh=cv.bitwise_not(bw_img)
    cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    listArea = []
    list_boundingRect =[]
    for c in cnts:

        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv.boundingRect(approx)
            ar = w / float(h)
            if ar >= 0.95 and ar <= 1.05:
                list_boundingRect.append((x, y, w, h))


    median_cnt = np.median(list_boundingRect,0)
    rows = thresh.shape[0]

    radius = (median_cnt[2]+median_cnt[3])/4
    
    return round(radius)
