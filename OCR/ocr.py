import os
import sys
import numpy as np
import cv2
from OCR.Segmentation import seg


def ocr(imgs_folder, masks_folder):
    plates = [int(addr.split('.')[0]) for addr in os.listdir(imgs_folder)]
    imgs = [imgs_folder + "{}.jpg".format(p) for p in plates]
    masks = [masks_folder + "{}.jpg".format(p) for p in plates]
    imgs = [[cv2.imread(img) for img in imgs],
            [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in masks]]
    seg(imgs, plates)