"""
Module to manage files
"""
import glob
import os

def create_file(file_path, data):
    """
    generate a file with specific data
    """
    
    file_loaded = open(file_path, "w")
    if data != None:
        file_loaded.write(data)
    file_loaded.close()

def create_directory(dir_path):
    """
    create needed directory
    """
    # @Todo check if this function work on windobe
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError:
            print ("Creation of the directory %s failed" % dir_path)
        else:
            print ("Successfully created the directory %s" % dir_path)
    else:
        print ("%s already exist" % dir_path)

def get_file_list(dir_path):
    """
    return all file in a directory
    """
    return glob.glob(dir_path)
