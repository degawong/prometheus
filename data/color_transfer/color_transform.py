'''
Author: your name
Date: 2021-09-07 16:16:48
LastEditTime: 2021-09-08 17:02:23
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \prometheus\data\color_transfer\color_transform.py
'''

import os
import cv2
import sys
# sys.path.append('../.')
# sys.path.append('../dega_package/')
import dega_package.dega_filesystem as df
from color_transfer import color_transfer as ct

class color_transfer():
    def __init__(self) -> None:
        self.__df = df.dega_filesystem()
    def apply(self, source_directory, target_directory, transfer_directory):
        source_list = self.__df.walk_path(source_directory)
        target_list = self.__df.walk_path(target_directory)
        for source_path, target_path in zip(source_list, target_list):
            source_image = cv2.imread(source_path)
            target_image = cv2.imread(target_path)
            transfer_image = ct(source_image, target_image)
            # transfer_image = cf(source_image, target_image, clip=True, preserve_paper=False)
            (_, image_name) = os.path.split(source_path)
            cv2.imwrite(transfer_directory + '/' + image_name, transfer_image)

if __name__ == '__main__':
    color_transfer().apply(
        source_directory='g:/ProjectDocu/Python',
        target_directory='g:/ProjectDocu/Python',
        transfer_directory='g:/share/image_io/color_transfer'
    )