'''
Author: your name
Date: 2021-09-07 18:43:11
LastEditTime: 2021-09-07 20:10:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \dega_package\dega_imageio.py
'''
# -*- coding: utf-8 -*-

import os
import re
import cv2
import numpy as np
import scipy as sp

import dega_filesystem

class dega_imageio():
    def __init__(self):
        self.__df = dega_filesystem.dega_filesystem()
        self.__four_cc={
            '.avi' : cv2.VideoWriter_fourcc('F','M','P','4'),
            '.mp4' : cv2.VideoWriter_fourcc('M','J','P','G'),
        }
    def image_video(self, image_directory, video_path, fps=24):
        image_list = self.__df.walk_path(image_directory)
        assert(len(image_list) > 1)
        size = cv2.imread(image_list[0]).shape[0:2]
        (_, extension) = os.path.splitext(video_path)
        video_writer = cv2.VideoWriter(video_path, self.__four_cc[extension], fps, size)
        for image_path in image_list:
            video_writer.write(cv2.imread(image_path))
        video_writer.release()
    def video_image(self, video_path, image_directory, image_extension='.bmp'):
        video = cv2.VideoCapture(video_path)
        for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, frame = video.read()
            cv2.imwrite(os.path.join(image_directory,'{:06d}'.format(i) + image_extension), frame) if _ is not None else None

if __name__ == '__main__':
    dega_imageio().video_image(image_directory='g:/share/image_io/image/', video_path='g:/share/image_io/000.mp4')
    # dega_imageio().image_video(image_directory='g:/share/image_io/image/', video_path='g:/share/image_io/video/0.avi', fps=24)