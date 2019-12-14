# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc


# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls


def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files


def find_corners(poly):
    if len(poly) <= 4:
        return poly
    box = []
    box.append(poly[0])

    if float(poly[0][1]) - float(poly[1][1]) == 0:
        prev_slope = (float(poly[1][1]) - float(poly[0][1])) * 1000
    else:
        prev_slope = (float(poly[1][1]) - float(poly[0][1])) / (float(poly[1][0]) - float(poly[0][0]))

    for i in range(1, len(poly)):
        if float(poly[i - 1][1]) - float(poly[i][1]) == 0:
            slope = (float(poly[i][1]) - float(poly[i - 1][1])) * 1000
        else:
            slope = (float(poly[i][1]) - float(poly[i - 1][1])) / (float(poly[i][0]) - float(poly[i - 1][0]))
        if abs(prev_slope - slope) > 2:
            box.append(poly[i - 1])
        prev_slope = slope

    box.append(poly[-1])
    return np.array(box)


def perspective_trans(img, poly, max_width=350, max_height=100, resize=False, size=None):
    img0 = np.copy(img)
    if resize:
        img0 = cv2.resize(img0, (size[1], size[0]))
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(poly.astype(np.float32), dst)
    return cv2.warpPerspective(img0, M, (max_width, max_height))


def saveResult(img_file, img, boxes, mask_file, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)
        # img = np.zeros(img.shape, dtype=np.uint8)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        # res_file = dirname + filename + '.txt'
        res_img_file = dirname + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        box_exits = False
        # with open(res_file, 'w') as f:
        for i, box in enumerate(boxes):
            box_exits = True
            poly = np.array(box).astype(np.int32).reshape((-1))

            # strResult = ','.join([str(p) for p in poly]) + '\r\n'
            # f.write(strResult)

            poly = poly.reshape(-1, 2)
            poly = find_corners(poly)
            # print(poly)
            # cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)


        # Save result image
        # cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance, useHarrisDetector=True)
        if box_exits:
            # print(img.shape)
            cv2.imwrite(res_img_file, perspective_trans(img, poly))
            mask_addr = "heat_maps/" + filename + '.jpg'
            # print(mask_file.shape)
            cv2.imwrite(mask_addr, perspective_trans(mask_file, poly, resize=True, size=img.shape))
