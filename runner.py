import os
import sys
from multiprocessing import Pool
from time import perf_counter
import shutil
import plate_detect.detect as detector
import numpy as np


def f(imgs):
    # print("module:1 - img:{}".format(x[1]))
    # os.system("python3 detect.py ../{} output/{}.png".format(x[0], x[1]))
    # img_addrs = os.listdir("../{}".format(x[0]))
    detector.detect(imgs, "output/")


# def f2(x):
#     print("module:3 - img:{}".format(x[0]))
#     os.system("python3 real_plates.py result/{}.jpg heat_maps/{}.jpg".format(x[0], x[1]))


if __name__ == '__main__':
    # ---- input files
    t0 = perf_counter()
    a = ['../test_set/images/' + addr for addr in os.listdir('test_set/images')]
    b = ['../test_set/Normal_plate_net/' + addr for addr in os.listdir('test_set/Normal_plate_net')]
    c = ['../test_set/siyasi/' + addr for addr in os.listdir('test_set/siyasi/')]
    d = ['../test_set/gozar_movaghat/' + addr for addr in os.listdir('test_set/gozar_movaghat/')]
    imgs = a + b + c + d
    # ---- module1 plate detection
    print("========= module1 =========")
    if os.path.exists('plate_detect/output'):
        shutil.rmtree('plate_detect/output')
        os.mkdir('plate_detect/output')
    os.chdir('plate_detect')
    f(imgs)
    os.chdir('../')
    print()
    # p = Pool(8)
    # jobs = zip(imgs, list(range(1, len(imgs) + 1)))
    # p.map(f, jobs)

    # ---- module2 OCR detection
    print("========= module2 =========")
    if os.path.exists('OCD/result/'):
        shutil.rmtree('OCD/result/')
    os.mkdir('OCD/result/')
    if os.path.exists('OCD/plates/'):
        shutil.rmtree('OCD/plates/')
    os.mkdir('OCD/plates/')
    if os.path.exists('OCD/heat_maps/'):
        shutil.rmtree('OCD/heat_maps/')
    os.mkdir('OCD/heat_maps/')
    os.chdir("OCD/")
    os.system("python3 test.py --test_folder=../plate_detect/output")

    # ---- module3 post-process and OCR
    # f_list = sorted(os.listdir("result"))
    # f_list = sorted([int(f.split(".")[0]) for f in f_list])
    # p = Pool(8)
    # jobs = zip(f_list, f_list)
    # p.map(f2, jobs)

    # ---- get the results
    # f_list = sorted(os.listdir("result"))
    # plates = sorted([int(f.split(".")[0]) for f in f_list])
    # f_list = sorted(os.listdir("plates"))
    # true_plates = sorted([int(f.split(".")[0]) for f in f_list])

    # os.chdir("../")
    # classes = []
    # # print(plates)
    # # print(true_plates)
    # for i in range(1, len(imgs) + 1):
    #     if i in true_plates:
    #         if np.random.rand() > 0.5:
    #             classes.append("{} 1\r\n".format(i))
    #         else:
    #             classes.append("{} 2\r\n".format(i))
    #     elif i in plates:
    #         if np.random.rand() > 0.5:
    #             classes.append("{} 2\r\n".format(i))
    #         else:
    #             classes.append("{} 1\r\n".format(i))
    #     else:
    #         classes.append("{} 3\r\n".format(i))
    # # print(classes)
    # with open('out.txt', 'w') as FILE0:
    #     FILE0.writelines(classes)

    print("took {:.3f} sec, # images={}".format(perf_counter() - t0, len(imgs)))
    print("-----------")
    print("output file is in {}".format(os.path.join(os.getcwd(), 'out.txt')))
    print("-----------")