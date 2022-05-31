"""Debug module to be ran with:
python debug.py <path_to_image>
"""
import sys
import os

# import cv2 as cv

from tsumego_detector import open_and_preprocess_image, detect_stones, detect_grid
from tools.qualibration_tools import find_radius

BINARY_THRESH_PARAM = 127

def process_image(image_path: str):
    """process a single image for debugging"""
    # creates debug output directory
    debug_dir = os.path.splitext(image_path)[0]
    try:
        os.makedirs(debug_dir)
    except FileExistsError:
        pass

    print(f"debugging {image_path}")

    gscale_img, bw_img = open_and_preprocess_image(image_path, debug_dir)

    # stone radius
    stone_radius = find_radius(bw_img)
    print(stone_radius)

    stones, radius = detect_stones(gscale_img, debug_dir, stone_radius, 150, 16)

    if len(stones) == 0:
        print("Error no stones")
        return

    detect_grid(gscale_img, debug_dir, stones, radius)

if len(sys.argv) > 2:
    print("Usage: python debug.py <path_to_img>")
    sys.exit(1)

if len(sys.argv) == 2:
    process_image(sys.argv[1])
else:
    files_and_dirs = os.listdir(os.path.join(os.getcwd(), "images"))
    for name in files_and_dirs:
        img_name = os.path.join("images", name)
        if os.path.isfile(img_name):
            process_image(img_name)
