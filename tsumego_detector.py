import os
import math

from distutils.log import debug
from email.mime import image
from statistics import median
from textwrap import fill

import cv2 as cv
import numpy as np

from config import (
    DEBUG,
    DEBUG_PATH,
    PATH_DELIMITER,
    PROBLEM_UNIT_DIRECTORY,
    SGF_OUTPUT_PATH,
)
from operator import index

from tools.geometry_tools import (
    find_circles,
    filter_intersection,
    circle_to_lines,
    from_line_list_to_instersections,
    get_corners,
    get_corners_without_problem_border,
    get_dif_from_center,
    get_farthest_insection,
    get_main_corner,
    get_coord_to_screen_pos,
    get_size_grid,
    deskew
)
from tools.goban_scan import (
    from_circle_to_stones,
    from_gray_layer_to_lines,
)
from tools.qualibration_tools import (find_radius)
from tools.sgf import generate_sgf_data

# global parameter for binary threshold
# maybe this can be computed based on the image ?
BINARY_THRESH_PARAM = 127
# same as above
HOUGH_PARAM1 = 150
HOUGH_PARAM2 = 16

# constants for readablity
VERTICAL = 0
HORIZONTAL = 1

def image_scan(image_path: str, debug_dir: str):
    """toplevel function to be called by the bot or debug tools"""
    gscale_img, bw_img = open_and_preprocess_image(image_path, debug_dir)

    # find stone radius for latter stone detection
    estimated_stone_radius = find_radius(bw_img)

    # detecting stones
    detected_stones, actual_stone_radius = find_circles(
        gscale_img,
        estimated_stone_radius-2, estimated_stone_radius+2,
        (HOUGH_PARAM1, HOUGH_PARAM2)
    )

    # detecting edge of the board

    # place the stones on an indexed grid

    # prepare the sgf
    # TODO
    sgf_data = None
    return sgf_data


def open_and_preprocess_image(image_path: str, debug_dir: str):
    """Returns a black and white image and a grayscalre image from the given source"""
    # gscale
    gscale_img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    # TODO blurring ? for now blurring degrades result

    # black and white from original image
    _ret, bw_img = cv.threshold(gscale_img, BINARY_THRESH_PARAM, 255, cv.THRESH_BINARY)
    
    # write for debug
    if DEBUG:
        cv.imwrite(os.path.join(debug_dir, 'bw_img.png'), bw_img)
        cv.imwrite(os.path.join(debug_dir, 'gc_img.png'), gscale_img)

    return gscale_img, bw_img

def detect_stones(gscale_img, debug_dir: str, stone_radius: int, houghparam1: int, houghparam2: int):
    """Detect all the stones it can on the given gray scale image"""
    detected_stones = find_circles(
        gscale_img,
        stone_radius-2, stone_radius+2,
        (houghparam1, houghparam2)
    )

    # compute actual stone radius from stone measure for later use
    if len(detected_stones) > 0:
        actual_stone_radius = int(np.median(detected_stones[0, :, 2:3]))
    else:
        return [], -1

    # display detected circles for debug
    if DEBUG:
        debug_circle_image = cv.cvtColor(gscale_img, cv.COLOR_GRAY2RGB)
        for pt in detected_stones[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv.circle(debug_circle_image, (a, b), r, (0, 0, 255), 2)
        cv.imwrite(os.path.join(debug_dir, "stones.png"), debug_circle_image)

        print("radius", actual_stone_radius)

    return detected_stones[0], actual_stone_radius

def detect_grid(gc_img, debug_dir: str, stones, stone_radius: int):
    """Detect the board grid using the previous discovered stones"""
    # compute a grid from the stones
    # grid is (vertical edges, horizontal edges)
    grid = detect_edges_from_stones(gc_img, debug_dir, stones, stone_radius)

    # extend the grid to match the border of the board (in place)
    
    # detect generic lines and select potential grid candidates
    lines = detect_lines(gc_img, debug_dir, grid, stone_radius)
    new_vertical_edges, new_horizontal_edges = split_and_merge_lines(lines, stone_radius)
    if DEBUG:
        debug_grid(gc_img, debug_dir, "global_detected_edges.png", new_vertical_edges, new_horizontal_edges)

    # merge the two grids together
    vertical_edges, horizontal_edges = grid
    vertical_grid = merge_grid(vertical_edges, new_vertical_edges, stone_radius)
    horizontal_grid = merge_grid(horizontal_edges, new_horizontal_edges, stone_radius)

    if DEBUG:
        debug_grid(gc_img, debug_dir, "final_grid.png", vertical_grid, horizontal_grid)

    return vertical_grid, horizontal_grid


def detect_lines(img, debug_dir: str, grid, stone_radius: int):
    """Detects the lines on the image after removing the bounding box of the current grid"""
    vertical_edges, horizontal_edges = grid

    # compute bounding box of the actual grid with a margin
    margin = stone_radius // 2
    bb_x = (vertical_edges[0]-margin, vertical_edges[-1]+margin)
    bb_y = (horizontal_edges[0]-margin, horizontal_edges[-1]+margin)

    vertical_min_line = vertical_edges[-1] - vertical_edges[0] - 2*margin
    horizontal_min_line = horizontal_edges[-1] - horizontal_edges[0] - 2*margin

    min_line_size = min(vertical_min_line, horizontal_min_line)

    # removes the bounding box into a temp image
    temp_img = img.copy()
    cv.rectangle(temp_img, (bb_x[0], bb_y[0]), (bb_x[1], bb_y[1]), color=(255, 255, 255), thickness=-1)
    if DEBUG:
        cv.imwrite(os.path.join(debug_dir, "removed_grid.png"), temp_img)

    # detect lines in the remaining image
    # lines should be about the same size as the bounding box
    canny = cv.Canny(temp_img, 80, 120)
    lines = cv.HoughLinesP(canny, 1, np.pi / 180, 50, None, min_line_size, 10)
    if lines is None:
        lines = []

    if DEBUG:
        cv.imwrite(os.path.join(debug_dir, "canny_lines.png"), canny)
        debug_line_image = cv.cvtColor(temp_img, cv.COLOR_GRAY2RGB)

        for line in lines:
            lin = line[0]
            cv.line(debug_line_image, (lin[0], lin[1]), (lin[2], lin[3]), (0,0,255), 1, cv.LINE_AA)
        cv.imwrite(os.path.join(debug_dir, "detected_lines.png"), debug_line_image)

    return lines


def merge_grid(strong_lines, weak_lines, stone_radius: int):
    """Merge the two subgrids together, the strong grid is used as a reference and elements from the weak grid try to be added"""
    # add all the strong grid in the middle
    merged_grid = list(strong_lines)

    # try to add every lower elements from the weak lines
    print("coucou", 2*stone_radius)
    print(strong_lines)
    print(weak_lines)
    for line in reversed(weak_lines):
        if line < merged_grid[0]:
            # if we can add the line to the grid
            if 3*stone_radius // 2 < merged_grid[0] - line < 5*stone_radius // 2:
                merged_grid.insert(0, line)

    # same for higher elements
    for line in weak_lines:
        if line > merged_grid[-1]:
            # if we can add the line to the grid
            if 3*stone_radius // 2 < line - merged_grid[-1] < 5*stone_radius // 2:
                merged_grid.append(line)

    print(strong_lines)
    return merged_grid


def split_and_merge_lines(lines, stone_radius: int):
    """returns two lists of vertical and horizontal lines.
    Lines that are close enough are merged together.
    Elements are put inside lists of size 1 for further merging"""
    vertical_lines = []
    horizontal_lines = []
    # split lines into vertical and horizontal
    for line in lines:
        lin = line[0]
        angle = math.atan2(lin[1] - lin[3], lin[0] - lin[2])
        # if vertical line (angle close to pi/2)
        if abs(angle - math.pi / 2) < 0.5:
            # store x coordinates
            vertical_lines.append([int((lin[0] + lin[2]) // 2)])
        # if horizontal line (angle close to pi and -1)
        elif abs(angle - math.pi) < 0.5 or abs(angle + math.pi) < 0.5:
            # store y coordinates
            horizontal_lines.append([int((lin[1] + lin[3]) // 2)])
        # otherwise we skip the line

    new_vertical_lines = sort_and_merge_lines(vertical_lines, stone_radius)
    new_horizontal_lines = sort_and_merge_lines(horizontal_lines, stone_radius)

    return new_vertical_lines, new_horizontal_lines


def sort_and_merge_lines(clusters, stone_radius: int):
    """Takes a list of coordinates over an axis and returns a grid from the given stone radius"""
    # we assume the image to be well formed
    # we can merge recursively lines together as long as the cordinates are close enough
    # we use a list of lists with a single element so we can pop and extend to build clusters

    # TODO it is better to merge it first...
    
    if DEBUG:
        print(clusters)

    cluster_to_build = 0
    while cluster_to_build < len(clusters):
        # merge every clusters with a single element that can be merged with the current one
        current_cluster = cluster_to_build + 1
        while current_cluster < len(clusters):
            # if a cluster has multiple elements it must contains all the elements by assumption
            # we can skip it
            if len(clusters[current_cluster]) > 1:
                current_cluster += 1
                continue
            # if the distance between the cluster to build and the current cluster is less than the stone radius
            # we can merge them
            if abs(clusters[current_cluster][0] - clusters[cluster_to_build][0]) < stone_radius:
                # we add the lonely cluster to the one we are building
                clusters[cluster_to_build].extend(clusters.pop(current_cluster))

                # we dont increase the cluster counter because we deleted the current one
            else:
                # if the cluster is too far skip it
                current_cluster += 1
        
        # build the next cluster
        cluster_to_build += 1

    print(clusters)
    edges = list(sorted([sum(l) // len(l) for l in clusters]))
    if DEBUG:
        print("computed grid", edges)
    return edges

def build_grid_indexes(indexes, stone_radius: int):
    """Builds a fake grid that approximately respects stone radius by filling lines with no stones"""
    edges = sort_and_merge_lines(indexes.tolist(), stone_radius)

    # fill holes
    fill_after_edge = 0
    while fill_after_edge < len(edges) - 1:
        # if the two lines are too far add an intermediate line
        space = edges[fill_after_edge+1] - edges[fill_after_edge]
        #print(space)
        
        if space >= 3*stone_radius:
            # compute how many edges we need to add
            n_new_edges = space // (2*stone_radius) - 1
            print(space, 2*stone_radius, n_new_edges)
            # if there is not many space between the last edge and the limit decrease the number of edges
            # FIXME do we have to do this ?
            if space - (n_new_edges+1)*2*stone_radius > stone_radius:
               n_new_edges += 1

            # add them at regular interval
            # we start from the rightmost one because we insert them on the left of the other edges
            if n_new_edges > 0:
                edge_space = space // (n_new_edges + 1)
                for i in range(n_new_edges, 0, -1):
                    edges.insert(fill_after_edge+1, edges[fill_after_edge] + i * edge_space)
                fill_after_edge += n_new_edges

        fill_after_edge += 1
    
    if DEBUG:
        print("edges", edges)
    return edges
        

def detect_edges_from_stones(bw_img, debug_dir: str, stones, stone_radius: int):
    """Returns a list of edges directly computed from the stone position"""
    vertical_edges = build_grid_indexes(stones[:,:1], stone_radius)
    horizontal_edges = build_grid_indexes(stones[:,1:2], stone_radius)

    # debug the grid built from the stones
    if DEBUG:
        debug_grid(bw_img, debug_dir, "simple_edges.png", vertical_edges, horizontal_edges)

    return vertical_edges, horizontal_edges


def debug_grid(src_img, debug_dir: str, img_name: str, vertical_lines, horizontal_lines):
    """Given two sets of lines, prints the grid resulting from them on the given image"""
    if not vertical_lines or not horizontal_lines:
        print("One of coordinate empty, grid is empty")
        return
    debug_simple_grid_image = cv.cvtColor(src_img, cv.COLOR_GRAY2RGB)
    for x in vertical_lines:
        cv.line(debug_simple_grid_image, (x, horizontal_lines[0]), (x, horizontal_lines[-1]), (0, 255, 0), 1)
    for y in horizontal_lines:
        cv.line(debug_simple_grid_image, (vertical_lines[0], y), (vertical_lines[-1], y), (0, 255, 0), 1)

    cv.imwrite(os.path.join(debug_dir, img_name), debug_simple_grid_image)

def from_image_to_sgf_data(
    image_path: str,
    debug_dir: str,
    circle_param1=100,
    circle_param2=20,
):
    """
    get an image and generate data to make a sgf
    """
    gscale_img, bw_img = open_and_preprocess_image(image_path, debug_dir)

    # find stone radius for latter stone detection
    stone_radius = find_radius(bw_img)

    # detecting stones
    detected_stones = find_circles(
        gscale_img,
        stone_radius-2, stone_radius+2,
        (circle_param1, circle_param2)
    )

    # detecting edge of the board

    # prepare the sgf

    top = int(0.05 * img_loaded.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.05 * img_loaded.shape[1])  # shape[1] = cols
    right = left
    img_loaded = cv2.copyMakeBorder(img_loaded, top, bottom, left, right, cv2.BORDER_CONSTANT, None,(255,255,255))
    debug_img = img_loaded.copy()
    

    lines_from_circles =circle_to_lines(detected_stones)
    lines_h, lines_v = from_gray_layer_to_lines(im_bw, {"debug_img": debug_img},lines_from_circles=lines_from_circles)
    '''
    intersections = from_line_list_to_instersections(lines_h, lines_v)
    cv2.imwrite('bw_debug.png', im_bw)
    for i in lines_h:
        cv2.line(debug_img,i[0],i[1],color=(0,255,0),thickness=2)
    for i in lines_v:
        cv2.line(debug_img,i[0],i[1],color=(255,255,0),thickness=2)
    good_intersection = filter_intersection(im_bw,intersections)
    for i in good_intersection:
        cv2.circle(debug_img,radius=2,center=i,color=(0,255,0))
    '''
    farthest_insection = get_corners(im_bw)
    farthest_insection = get_corners_without_problem_border(im_bw,farthest_insection)
    lines = lines_h+lines_v
    corner =get_main_corner(farthest_insection,lines)
    
    '''(xdif,ydif) =get_dif_from_center(farthest_insection,lines)
    
    x_corner = corner[0]
    y_corner = corner[1]
    if x_corner == 1:
        point = (farthest_insection[0][0]+x_corner,farthest_insection[0][1])
        farthest_insection = (point,farthest_insection[1],farthest_insection[2],farthest_insection[3])
    '''

    #pts1 = np.float32([farthest_insection[0],farthest_insection[1],farthest_insection[3],farthest_insection[2]])
    #pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    #matrix = cv2.getPerspectiveTransform(pts1, pts2)
    #gray_layer = cv2.warpPerspective(gray_layer, matrix, (500, 500))
    #gray_layer= cv2.copyMakeBorder(gray_layer.copy(),10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))

    
    if DEBUG:
        #debug_img = cv2.warpPerspective(debug_img, matrix, (500, 500))
        #debug_img= cv2.copyMakeBorder(debug_img.copy(),10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.line(debug_img, farthest_insection[0], farthest_insection[1], (0,255,255), 3)
        cv2.line(debug_img, farthest_insection[1], farthest_insection[2], (0,255,255), 3)
        cv2.line(debug_img, farthest_insection[2], farthest_insection[3], (0,255,255), 3)
        cv2.line(debug_img, farthest_insection[3], farthest_insection[0], (0,255,255), 3)

    black_stones = []
    white_stones = []
 

    black_stones, white_stones, context = from_circle_to_stones(
        detected_stones,
        gray_layer,
        farthest_insection,
        avg_radius,
        {"debug_img": debug_img}
    )
  


    if DEBUG:
       

        
        
        color_radius = 4
        lenght = get_size_grid(farthest_insection,avg_radius)
        for y in range(lenght[1]+1):
            p1 =get_coord_to_screen_pos(farthest_insection,(0,y),avg_radius)
            p2 =get_coord_to_screen_pos(farthest_insection, (lenght[0],y),avg_radius)
 
            cv2.line(debug_img, p1, p2, (0,0,255), 1)
        for x in range(lenght[0]+1):
            p1 =get_coord_to_screen_pos(farthest_insection,(x,0),avg_radius)
            p2 =get_coord_to_screen_pos(farthest_insection,(x,lenght[1]),avg_radius)
 
            cv2.line(debug_img, p1, p2, (0,0,255), 1)
        for black_stone in black_stones:
            screen_coord =get_coord_to_screen_pos(farthest_insection,black_stone,avg_radius)
            cv2.circle(
            context["debug_img"],
            (screen_coord[0], screen_coord[1]),color_radius,(0,255,0),4)
        for white_stone in white_stones:
            screen_coord =get_coord_to_screen_pos(farthest_insection,white_stone,avg_radius)
            cv2.circle(
            context["debug_img"],
            (screen_coord[0], screen_coord[1]),color_radius,(0,0,255),4)

        debug_img = context["debug_img"]

        cv2.imwrite(path_img_debug, debug_img)
    if (len(black_stones) == 0 and len(white_stones) == 0):
        return None
    return generate_sgf_data(black_stones, white_stones,corner,lenght)

'''
problem_index = 0
create_directory(DEBUG_PATH)
create_directory(SGF_OUTPUT_PATH)

for image_to_scan in get_file_list(PROBLEM_UNIT_DIRECTORY + PATH_DELIMITER + "*"):
    if DEBUG:
        print("let s play with" + image_to_scan)
    image_name = image_to_scan.split(PATH_DELIMITER)[-1]
    sgf_data = image_scan(image_to_scan,DEBUG_PATH + PATH_DELIMITER + image_name)
    create_file(SGF_OUTPUT_PATH + PATH_DELIMITER + str(problem_index) + ".sgf", sgf_data)
    problem_index += 1
'''

