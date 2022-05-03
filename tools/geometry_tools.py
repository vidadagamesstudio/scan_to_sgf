"""
function for manipulate geometrical entity
"""

from math import (floor, sqrt)
from operator import xor
from statistics import median,mean
import cv2
import numpy as np

def are_vector_in_line(vector1, vector2, marge=0):
    """
    check if two vector are aligned (with a potential marge)
    """
    return abs(dot_product(vector1, vector2)) >= 1-marge

def compare_line_horizontaly(line1, line2):
    """
    return 1 if the first item is sup
    return -1 if the first item is less
    return 0 else
    """
    if (line1[0][1] + line1[1][1]) < (line2[0][1] + line2[1][1]):
        return -1
    elif (line1[0][1] + line1[1][1]) > (line2[0][1] + line2[1][1]):
        return 1
    return 0

def compare_line_verticaly(line1, line2):
    """
    return 1 if the first item is sup
    return -1 if the first item is inf
    return 0 else
    """
    if (line1[0][0] + line1[1][0]) < (line2[0][0] + line2[1][0]):
        return -1
    elif (line1[0][0] + line1[1][0]) > (line2[0][0] + line2[1][0]):
        return 1
    return 0

def determinant(line1, line2):
    """
    calculate the determinant
    """
    return int(line1[0]) * int(line2[1]) - int(line1[1]) * int(line2[0])

def find_circle(img, min_radius=7, max_radius=14, circle_params=(100, 45)):
    """
    find circles in an image
    """
    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT,
        1, 10,
        param1=circle_params[0], param2=circle_params[1],
        minRadius=min_radius,maxRadius=max_radius
    )
    if circles is None:
        return []
    return np.uint16(np.around(circles))
def circle_to_lines(circles):
    lines = []
    for (first_circle_x, first_circle_y, first_cirlce_r) in circles[0, :]:
        for (second_circle_x, second_circle_y, second_cirlce_r) in circles[0, :]:
            
            if xor(first_circle_x != second_circle_x,   first_circle_y != second_circle_y):   
                line = ((first_circle_x,first_circle_y),(second_circle_x,second_circle_y))
                lines.append(line)
    return lines

def is_pos_in_bound(pos,boundMax,boundMin =(0,0)):
    (x,y)=pos
    (bmax_x,bmax_y)=boundMax
    (bmin_x,bmin_y)=boundMin
    if (x >=bmin_x and x <= bmax_x) and (y >=bmin_y and y <= bmax_y):
        return True
    return False

def filter_intersection(image,intersections):
    good_intersection = []
    for i in intersections:
        if is_pos_in_bound(i,(image.shape[1]-1,image.shape[0]-1)):
            color = image[i[0],i[1]]
            if color == 255:
                good_intersection.append(i)

    return good_intersection
            
def from_line_list_to_instersections(lines_h, lines_v):
    """
    return all intersections between a list of lines
    """
    intersections = []
    for horizontal_lign in lines_h:
        for vertical_lign in lines_v:
            intersec = what_is_two_lines_intersection(vertical_lign, horizontal_lign)
            if intersec is not None:
                intersections.append(intersec)
    return intersections

def get_coord(corners, position, radius, size=19):
     
    lenght = get_size_grid(corners,radius)
    origin = corners[0]
    position = (position[0] - origin[0],position[1] - origin[1])
    c1 =corners[0]
    c2 =corners[2]
    len_pixel =(abs(c1[0]-c2[0]),abs(c1[1]-c2[1]))
    len_pixel_between = (len_pixel[0]/(lenght[0]), len_pixel[1]/(lenght[1]))
    result = (round(position[0]/len_pixel_between[0]),round(position[1]/len_pixel_between[1]))
    
    return result
def get_coord2(corner, position, radius, size=19):
    """
    get coordonate from position
    """
    diameter = radius * 2
    return (
        size - 1 - floor((corner[0] - position[0]) / diameter),
        size - 1 - floor((corner[1] - position[1]) / diameter)
    )

def rotate_image(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage, maxAngle):
    angle = get_skew_angle(cvImage)
    if abs(angle)<=maxAngle:
        return rotate_image(cvImage, -1.0 * angle)
    return cvImage
# Calculate skew angle of an image
def get_skew_angle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle
def get_size_grid(corners,radius):
    """
    get size grid
    """
    c1 =corners[0]
    c2 =corners[2]
    diameter = radius * 2
    lx =abs(c1[0]-c2[0])/diameter
    ly=abs(c1[1]-c2[1])/diameter

    c3 =corners[1]
    c4 =corners[3]
    lx2 =abs(c3[0]-c4[0])/diameter
    ly2=abs(c3[1]-c4[1])/diameter
    lenght = (mean([lx,lx2]),mean([ly,ly2]))
    lenght = min(18,round(lenght[0])),min(18,round(lenght[1]))
    return lenght

def get_coord_to_screen_pos(corners, coord, radius):
    """
    get screen pos from coordinate
    """

    lenght = get_size_grid(corners,radius)
    origin = corners[0]
    c1 =corners[0]
    c2 =corners[2]
    len_pixel =(abs(c1[0]-c2[0]),abs(c1[1]-c2[1]))
    len_pixel_between = (len_pixel[0]/(lenght[0]), len_pixel[1]/(lenght[1]))
    result = (int(origin[0] + coord[0]*len_pixel_between[0]),int(origin[1] + coord[1]*len_pixel_between[1]))
    return result
def get_coord_to_screen_pos2(corner, coord, radius, size=19):
    """
    get screen pos from coordinate
    """
    
    diameter = radius * 2
    result = (int(corner[0] - coord[0]*diameter),int(corner[1] - (size - 1-coord[1])*diameter))
    return result
def get_dif_to_range(x,min,max):
    if(x>=min and x <=max):
        return 0
    
    if x <min:
        return x- min
    if x >max:
        return x- max
def clamp(num, min_value, max_value):
        num = max(min(num, max_value), min_value)
        return num
def get_dif_from_center(farthest_intersection, lines,treshold =5):
    list_points_x = []
    list_points_y = []
    list_dif_x = []
    list_dif_y = []
    dif_x_mean = 0
    dif_y_mean = 0
    min_x = min(farthest_intersection[0][0],farthest_intersection[3][0])
    max_x = max(farthest_intersection[1][0],farthest_intersection[2][0])
    min_y = min(farthest_intersection[1][1],farthest_intersection[0][1])
    max_y =max(farthest_intersection[2][1],farthest_intersection[3][1])
    
    for line in lines:
        for point in line:
            x = point[0]
            y = point[1]
            dif_x =get_dif_to_range(x,min_x,max_x)
            dif_y =get_dif_to_range(y,min_y,max_y)
            if(abs(dif_x)>treshold):
                list_points_x.append(x)
                list_dif_x.append(dif_x)
            if(abs(dif_y)>treshold):
                list_points_y.append(y)
                list_dif_y.append(dif_y)
                
    if(len(list_points_x)>0 and len(list_points_y)>0):
        point_mean =(mean(list_points_x),mean(list_points_y))
        corner_x = [farthest_intersection[0][0],farthest_intersection[1][0],farthest_intersection[2][0],farthest_intersection[3][0]]
        corner_y = [farthest_intersection[0][1],farthest_intersection[1][1],farthest_intersection[2][1],farthest_intersection[3][1]]
        corner_mean =(mean(corner_x),mean(corner_y))
        x = corner_mean[0]-point_mean[0]
        y = corner_mean[1]-point_mean[1]
        
            
        dif_x_mean =mean(list_dif_x)
        dif_y_mean =mean(list_dif_y)
    return (dif_x_mean,dif_y_mean)


def get_corners_without_problem_border(image,corners):
    left = corners[0][0]
    top = corners[0][1]
    bottom = corners[2][1]
    right = corners[2][0]
    pixel_white_treshold = 0.95
    result =[]
    width =right-left
    height = bottom-top
    list_range = [(left,right),(top,bottom)]
    for axis in range(2):
        for step in [1,-1]:
            ran = list_range[axis]
            if step==-1:
                ran = (ran[1],ran[0])

            for i in range(ran[0],ran[1],step):
                if axis == 1:
                    zone =image[i:i+1, left:left+width]
                else:
                    zone =image[top:top+height, i:i+1]
                nb_pixel_black =np.sum(zone == 255)  
                if (axis == 0 and nb_pixel_black < round(pixel_white_treshold*height))or (axis == 1 and nb_pixel_black < round(pixel_white_treshold*width)) :
                    result.append(i)
                    break

    retour = ((result[0],result[2]),(result[0],result[3]),(result[1],result[3]),(result[1],result[2]))
    
    return retour    
    

def get_main_corner(farthest_intersection, lines,treshold =5):
    list_points_x = []
    list_points_y = []
    list_dif_x = []
    list_dif_y = []
    min_x = min(farthest_intersection[0][0],farthest_intersection[3][0])
    max_x = max(farthest_intersection[1][0],farthest_intersection[2][0])
    min_y = min(farthest_intersection[1][1],farthest_intersection[0][1])
    max_y =max(farthest_intersection[2][1],farthest_intersection[3][1])
    
    for line in lines:
        for point in line:
            x = point[0]
            y = point[1]
            dif_x =get_dif_to_range(x,min_x,max_x)
            dif_y =get_dif_to_range(y,min_y,max_y)
            if(abs(dif_x)>treshold):
                list_points_x.append(x)
                list_dif_x.append(dif_x)
            if(abs(dif_y)>treshold):
                list_points_y.append(y)
                list_dif_y.append(dif_y)
                
    if(len(list_points_x)>0 and len(list_points_y)>0):
        point_mean =(mean(list_points_x),mean(list_points_y))
        corner_x = [farthest_intersection[0][0],farthest_intersection[1][0],farthest_intersection[2][0],farthest_intersection[3][0]]
        corner_y = [farthest_intersection[0][1],farthest_intersection[1][1],farthest_intersection[2][1],farthest_intersection[3][1]]
        corner_mean =(mean(corner_x),mean(corner_y))
        x = corner_mean[0]-point_mean[0]
        y = corner_mean[1]-point_mean[1]
        
            
        dif_x_mean =mean(list_dif_x)
        dif_y_mean =mean(list_dif_y)
        return (clamp(-dif_x_mean,-1,1),clamp(-dif_y_mean,-1,1))

    return (-1,-1)
def get_corners(bw_img):
    inversed_image = cv2.bitwise_not(bw_img)
    contours,hierarchy = cv2.findContours(inversed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    c=contours[max_index]
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    (x, y, w, h) = cv2.boundingRect(approx)
    return ((x,y),(x,y+h),(x+w,y+h),(x+w,y))

    


def get_farthest_insection(intersections, maxsize):

    corners = [
        (0,0),
        (maxsize[0], 0),
        maxsize,
        (0, maxsize[1])
    ]

    farthest_corner = [None, None, None, None]
    for intersec in intersections:
        if (intersec[0] >= 0) \
            and (intersec[1] >= 0) \
            and (intersec[0] <= maxsize[0]) \
            and (intersec[1] <= maxsize[1]) \
        :
            for i, i_corner in enumerate(corners):
                if farthest_corner[i] is None:
                    farthest_corner[i] = intersec
                else:
                    if (
                        measure_distance(intersec, i_corner)
                        < measure_distance(farthest_corner[i], i_corner)
                    ):
                        farthest_corner[i] = intersec
    return (farthest_corner)

def get_vector_of_line(line):
    """
    return vector defining a line
    """

    x =int(line[1][0]) - int(line[0][0])
    y =int(line[1][1]) - int(line[0][1])
    return (x,y)

def is_near_horisontal(vector, marge = 0.01):
    """
    check if a vector is near vertical
    """
    return are_vector_in_line(vector, (1,0), marge)

def is_near_vertical(vector, marge = 0.01):
    """
    check if a vector is near vertical
    """
    return are_vector_in_line(vector, (0,1), marge)

def measure_distance(point1, point2):
    """
    return distance between two point
    """
    return sqrt(
        pow(point2[0]-point1[0], 2)
        + pow(point2[1]-point1[1],2)
    )

def find_nearest_pixel(img, target):
    nonzero = cv2.find(img)
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index]

def what_is_two_lines_intersection(line1, line2):
    """
    return the intersection between two lines
    """
    xdiff = (int(line1[0][0]) - int(line1[1][0]), int(line2[0][0]) - int(line2[1][0]))
    ydiff = (int(line1[0][1]) - int(line1[1][1]), int(line2[0][1]) - int(line2[1][1]))

    div = determinant(xdiff, ydiff)
    if div == 0:
        return None

    d_determinant = (determinant(*line1), determinant(*line2))
    x_determinant = determinant(d_determinant, xdiff) / div
    y_determinant = determinant(d_determinant, ydiff) / div

    return int(x_determinant), int(y_determinant)

def normalize_vector(vector):
    """
    return vector normalized
    """
    power = (pow(vector[0], 2) + pow(vector[1],2))
    lenght = sqrt(power)
    return (
        vector[0] / lenght,
        vector[1] / lenght
    )

def dot_product(vector1, vector2):
    """
    @Todo don t know
    """
    normalize_vector1 = normalize_vector(vector1)
    normalize_vector2 = normalize_vector(vector2)
    return normalize_vector1[0] * normalize_vector2[0] + normalize_vector1[1] * normalize_vector2[1]
