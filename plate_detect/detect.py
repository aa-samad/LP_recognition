import cv2 as cv
import sys
import os
import shutil
import numpy as np


def detect(img_addrs, out_folder):
    model_addr = "model/frozen_inference_graph.pb"
    model_addr_txt = "model/graph.pbtxt"

    # LABELS = ["null","plate"]
    cvNet = cv.dnn.readNetFromTensorflow(model_addr, model_addr_txt)
    # --- pre process
    imgs = [cv.imread(addr) for addr in img_addrs]
    # img = cv.imread(image_addr)
    for i in range(len(imgs)):
        imgs[i] = cv.cvtColor(imgs[i], cv.COLOR_BGR2HSV)
        clahe = cv.createCLAHE(clipLimit=4.0,tileGridSize=(4,4))
        imgs[i][:, :, 2] = clahe.apply(imgs[i][:, :, 2])
        imgs[i] = cv.cvtColor(imgs[i], cv.COLOR_HSV2BGR)
    # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    # ---- feeding the net
    cvOut = np.zeros((len(imgs), 1, 100, 7))
    a = cv.dnn.blobFromImages(imgs, size=(300, 300), swapRB=True, crop=False)
    for i in range(len(imgs)):
        cvNet.setInput(a[i:i+1, :, :, :])
        cvOut[i:i+1, :, :, :] = cvNet.forward()
        print("item {} / {} Done!".format(i, len(imgs)))

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    else:
        shutil.rmtree(out_folder)
        os.mkdir(out_folder)

    score_thr = 0.5
    max_area = 0

    for i in range(cvOut.shape[0]):
        for detection in cvOut[i, 0, :, :]:
            score = float(detection[2])
            label = int(detection[1])
            if score > score_thr and label == 1:  # confident plate
                rows = imgs[i].shape[0]
                cols = imgs[i].shape[1]
                left = int(detection[3] * cols)
                top = int(detection[4] * rows)
                right = int(detection[5] * cols)
                bottom = int(detection[6] * rows)
                area = (bottom - top) * (right - left)
                # --- detect of bounding box has some ratio and bigger area than others
                if 0.15 < (bottom - top) / (right - left) < 0.35:
                    # max_area = area
                    # --- if box was outside of image
                    left = 0 if left < 0 else left
                    top = 0 if top < 0 else top
                    right = imgs[i].shape[1] if right > imgs[i].shape[1] else right
                    bottom = imgs[i].shape[0] if bottom > imgs[i].shape[0] else bottom
                    # --- create a margin
                    left = left - 10 if left - 10 > 0 else left
                    top = top - 10 if top - 10 > 0 else top
                    right = right + 10 if right + 10 < imgs[i].shape[1] else right
                    bottom = bottom + 10 if bottom + 10 < imgs[i].shape[0] else bottom

                    # for i in range(3):
                    cv.imwrite(out_folder + '{}.png'.format(i), imgs[i][top:bottom, left:right])
                    # print("left:{} top:{} right:{} bottom:{}".format(left,top,right,bottom))

                    # cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
        #             left = left - 15 if left - 15 > 15 else left + 15
        #             top = top - 5
        #             cv.putText(img, label, (int(left), int(top)),
        #                 cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)
    #     cv.imwrite(sys.argv[2], img)

if __name__ == "__main__":
    img_folder = os.listdir(sys.argv[1])
    out_folder = sys.argv[2]