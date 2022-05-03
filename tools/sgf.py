"""
function to manage sgf
"""

def coordonate_to_sgf_coordonate(point):
    """
    transform coordonate to sgf coordonate system
    """
    base = ord('a')
    return chr(base++point[0]) + chr(base++point[1])
def modify_coord_by_corner(coord,corner,grid_size,size=19):
    x = 0
    y = 0
    if(corner[0]==1):
        x+=size-grid_size[0]-1
    if(corner[1]==1):
        y+=size-grid_size[1]-1
    coord = (x+coord[0],y+coord[1])
    return coord
def generate_sgf_data(black_stones, white_stones,corner,grid_size):
    """
    prepare data to sgf
    """
    sgf_begining = "(;GM[1]FF[4]CA[UTF-8]AP[SgfHelperBot]KM[6.5]SZ[19]"
    sgf_end = ")"
    black_moves ="AB"
    white_moves = "AW"


    for i_black_stone in black_stones:
        coord = modify_coord_by_corner(i_black_stone,corner,grid_size,19)
        black_moves = black_moves + "[" + coordonate_to_sgf_coordonate(coord) + "]"

    for i_white_stone in white_stones:
        coord = modify_coord_by_corner(i_white_stone,corner,grid_size,19)
        white_moves = white_moves + "[" + coordonate_to_sgf_coordonate(coord) + "]"

    if len(black_moves) == 2:
        black_moves = ""

    if len(white_moves) == 2:
        white_moves = ""

    return sgf_begining + black_moves + white_moves + sgf_end
