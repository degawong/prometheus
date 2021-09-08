'''
Author: your name
Date: 2021-09-07 16:22:44
LastEditTime: 2021-09-07 18:47:29
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \color_transfer\dega_package\dega_filesystem.py
'''
# -*- coding: utf-8 -*-

import os
import re
import cv2
import numpy as np
import scipy as sp

class dega_filesystem():
    __image_format = [
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        '.jpg', '.JPG', '.jpeg', '.JPEG','.dng', '.DNG',
    ]
    __image_regex = ".*\.(bmp|BMP|jpg|JPG|jpeg|JPEG|png|PNG)"
    def __init__(self, match_regex=None):
        self.__regex = self.__image_regex if match_regex is None else match_regex
    def walk_path(self, path):
        assert os.path.exists(path), "directory {} does not exist".format(path)
        return [os.path.join(path, _) for _ in os.listdir(path) if re.match(self.__regex, _)]

if __name__ == '__main__':
    path_list = dega_filesystem().walk_path(path='g:/ProjectDocu/Python')
    print(path_list)