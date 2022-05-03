"""
Function to transform a picture to a list of stones
"""

import functools
import cv2
import numpy as np


from config import (
    DEBUG,
    DEBUG_EDGE_PATH,
    MIN_LINE_LENGTH,
)
from tools.geometry_tools import (
    compare_line_horizontaly,
    compare_line_verticaly,
    get_coord,
    get_vector_of_line,
    is_near_horisontal,
    is_near_vertical,
  
)

def from_circle_to_stones(circles, gray_layer, farthest_insection,avg_radius, context):
    """
    return stones list
    """
    black_stones = []
    white_stones = []
    total_ray = 0
    nb_circles= 0

    if len(circles) != 0:
        for (circle_x, circle_y, cirlce_r) in circles[0, :]:
            nb_circles += 1    

            coord = get_coord(
                farthest_insection,
                (circle_x, circle_y),
                avg_radius             
            )
            mask = np.full((gray_layer.shape[0], gray_layer.shape[1]), 0, dtype=np.uint8)
            cv2.circle(mask, (circle_x, circle_y), cirlce_r, (255, 255, 255), -1)
            mean_color = cv2.mean(gray_layer, mask=mask)

            if mean_color[0] <= 150:
                if coord not in black_stones:
                    black_stones.append(coord)

            else:
                if coord not in white_stones:
                    white_stones.append(coord)

            if DEBUG and "debug_img" in context:
                cv2.circle(
                    context["debug_img"],
                    (circle_x, circle_y),
                    cirlce_r,
                    (0,0,255),
                    2
                )

    return black_stones, white_stones, context

def from_gray_layer_to_lines(gray_layer, context,lines_from_circles=[]):
    """
    scan a gray layer to return lists of lines
    """
    edges = cv2.Canny(gray_layer, 50, 150, apertureSize=3)
    if DEBUG:
        cv2.imwrite(DEBUG_EDGE_PATH, edges)
    lines_h = []
    lines_v = []

    lines = cv2.HoughLinesP( # find line
        image=edges,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        lines=np.array([]),
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=20
    )
    simple_lines = lines_from_circles
    size_of_shape, _, _ = lines.shape
    for i_line in range(size_of_shape):
        
        line = (
            (lines[i_line][0][0], lines[i_line][0][1]),
            (lines[i_line][0][2], lines[i_line][0][3])
        )
        simple_lines.append(line)
    for line in simple_lines:
    
        vector = get_vector_of_line(line)
        if is_near_horisontal(vector):
            lines_h.append(line)
        elif is_near_vertical(vector):
            lines_v.append(line)


    lines_h.sort(key=functools.cmp_to_key(compare_line_horizontaly))
    lines_v.sort(key=functools.cmp_to_key(compare_line_verticaly))

    return lines_h, lines_v
