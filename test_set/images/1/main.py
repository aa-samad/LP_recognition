import cv2
import numpy as np
import sys

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
    img[:, :, 2] = clahe.apply(img[:, :, 2])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow("image", img)
    cv2.waitKey()

