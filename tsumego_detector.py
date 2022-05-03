
from distutils.log import debug
from email.mime import image
from statistics import median
import cv2

from config import (
    DEBUG,
    DEBUG_PATH,
    PATH_DELIMITER,
    PROBLEM_UNIT_DIRECTORY,
    SGF_OUTPUT_PATH,
)
from tools.file_tools import (
    create_directory,
    create_file,
    get_file_list,
)
from tools.geometry_tools import (
    find_circle,
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


def image_scan(path_img,path_debug):
    radius_min, radius_max = find_radius(path_img)
    sgf_data = from_image_to_sgf_data(path_img,path_debug, radius_min, radius_max, 150, 16)
    return sgf_data



def from_image_to_sgf_data(
    image_path,
    path_img_debug,
    stone_min_radius=20,
    stone_max_radius=30,
    circle_param1=100,
    circle_param2=20,
    
    
):

    """
    get an image and generate data to make a sgf
    """
    avg_radius = (stone_min_radius + stone_max_radius)/2
    image_name = image_path.split(PATH_DELIMITER)[-1]
    img_loaded = cv2.imread(image_path)
    img_loaded =deskew(img_loaded,15.0)
    

    top = int(0.05 * img_loaded.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.05 * img_loaded.shape[1])  # shape[1] = cols
    right = left
    img_loaded = cv2.copyMakeBorder(img_loaded, top, bottom, left, right, cv2.BORDER_CONSTANT, None,(255,255,255))
    debug_img = img_loaded.copy()

    gray_layer = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2GRAY)
    im_bw = cv2.threshold(gray_layer, 127, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('bw_image.png', im_bw)
    
    detected_circles = find_circle(
        gray_layer,
        stone_min_radius, stone_max_radius,
        (circle_param1, circle_param2)
    )
    lines_from_circles =circle_to_lines(detected_circles)
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
        detected_circles,
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

