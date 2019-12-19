import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


def refine_cut(cuts, no_recursion=False):
    """" refine created cuts (multiple cuts into one) """
    prev_c = 0
    c_bank = []
    temp_c = []
    for c in cuts:
        temp_c.append(c)
        if (c - prev_c) > 10:
            c_bank.append(temp_c)
            temp_c = []
        prev_c = c
    cuts0 = [0, ]
    if no_recursion:
        for i in c_bank:
            cuts0.append(int(np.max(np.array(i))))
    else:
        for i in c_bank:
            cuts0.append(int(np.mean(np.array(i))))
    if no_recursion:
        return cuts0
    else:
        return refine_cut(cuts0, no_recursion=True)


def cut_image(img0, cut_points):
    """ cut image with the given cut points"""
    img = np.copy(img0)
    imgs = []
    for i in range(len(cut_points)):
        imgs.append(img[:, cut_points[i - 1]: cut_points[i], :])
        if i == len(cut_points) - 1:
            imgs.append(img[:, cut_points[i]:, :])
    return imgs


def save_images(folder, imgs):
    """ save blob of images into its folder """
    for i in range(len(imgs)):
        cv2.imwrite("{}/{}.jpg".format(folder, i), imgs[i])


def seg(img_and_masks, plates):
    for i in range(len(plates)):
        img = img_and_masks[0][i]
        msk = img_and_masks[1][i]
        blur = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 0)
        ret2, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # cv2.imshow("plate_thresh", th2)
        # cv2.waitKey(10)

        th2 = th2.astype(np.uint8) // 255
        horizontal_hist = np.sum(th2, axis=0)
        seg_thresh = np.max(horizontal_hist) * 0.87
        cut_points2 = np.where(horizontal_hist > seg_thresh)[0]
        # # plt.plot(np.sum(th2, axis=0))
        # # plt.show()
        #
        # ret, th1 = cv2.threshold(msk, 180, 255, cv2.THRESH_BINARY)
        # nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(th1.astype(np.uint8),
        #                                                                      connectivity=8)
        # # find cut points
        # cut_points = []
        # print(nLabels - 1)
        # centroids_col = sorted([centroids[i][0] for i in range(len(centroids))])
        # for i in range(1, len(centroids_col)):
        #     cut_points.append(int((centroids_col[i] + centroids_col[i - 1]) / 2))

        # refine cut points
        # try:
        #     char_space = cut_points[1] - cut_points[0]
        # except:
        #     print("segmentation unsuccessful")
        # for i in range(1, len(cut_points)):
        #     pass

        cut_points2 = refine_cut(cut_points2)
        # segment images
        seged_imgs = cut_image(img, cut_points2)

        # save segmented images
        if not os.path.exists("out/{}/".format(i)):
            os.mkdir("out/{}/".format(i))

        for j in range(len(seged_imgs)):
            cv2.imwrite("out/{}/{}.jpg".format(i, j), seged_imgs[j])

        # cv2.imshow("mask", msk)
        # cv2.imshow("threshold", th1)
        # cv2.imshow("frame", img)
        # cv2.imshow("cutted", cutted)
        # cv2.waitKey()


if __name__ == "__main__":
    imgs_folder = 'result/'
    masks_folder = 'heat_maps/'

    plates = [int(addr.split('.')[0]) for addr in os.listdir(imgs_folder)]
    img0 = [imgs_folder + "{}.jpg".format(p) for p in plates]
    mask0 = [masks_folder + "{}.jpg".format(p) for p in plates]
    imgs = [[cv2.imread(img) for img in img0],
            [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in mask0]]

    seg(imgs, plates)
