import cv2
import numpy as np
import sys

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # img = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
    # img = cv2.equalizeHist(img)
    # img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    # img[:, :, 1] = clahe.apply(img[:, :, 1])
    # img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    img[:, :, 2] = clahe.apply(img[:, :, 2])

    # img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    # img[:, :, 0] = clahe.apply(img[:, :, 0])
    # cv2.equalizeHist(img)
    # img = cv2.equalizeHist(img)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    # img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    cv2.imshow("image", img)
    cv2.waitKey()

