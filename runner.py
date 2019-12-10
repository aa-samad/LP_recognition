import os
import sys
from multiprocessing import Pool
from time import perf_counter
import shutil
import numpy as np


def f(x):
    print("module:1 - img:{}".format(x[1]))
    os.system("python3 detect.py {} OCD/data/{}.png".format(x[0], x[1]))


def f2(x):
    print("module:3 - img:{}".format(x[0]))
    os.system("python3 real_plates.py result/{}.jpg heat_maps/{}.jpg".format(x[0], x[1]))


if __name__ == '__main__':
    t0 = perf_counter()
    # a = open(sys.argv[1]).read().splitlines()
    a = ['images/1/0001.jpg', 'images/1/0002.jpg', 'images/1/0003.jpg', 'images/1/0004.jpg', 'images/1/0005.jpg', 
        'images/2/0001.jpg', 'images/2/0002.jpg', 'images/2/0003.jpg', 'images/2/0004.jpg', 'images/2/0005.jpg', 
        'images/3/0001.jpg', 'images/3/0002.jpg', 'images/3/0003.jpg', 'images/3/0004.jpg', 'images/3/0005.jpg',
        'images/4.png', 'images/5.png', 'images/6.png','images/7.jpg', 'images/8.jpg', 'images/9.jpg',]
    imgs = [addr for i, addr in enumerate(a) if i > 0 and i % 2 == 0]

    # ---- module1 plate detection
    if os.path.exists('OCD/data/'):
        shutil.rmtree('OCD/data/')
        os.mkdir('OCD/data/')
    p = Pool(8)
    jobs = zip(imgs, list(range(1, len(imgs) + 1)))
    p.map(f, jobs)

    # ---- module2 OCR detection
    print("module:2 running...")
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
    os.system("python3 test.py")

    # ---- module3 post-process and OCR
    f_list = sorted(os.listdir("result"))
    f_list = sorted([int(f.split(".")[0]) for f in f_list])
    p = Pool(8)
    jobs = zip(f_list, f_list)
    p.map(f2, jobs)

    # ---- get the results
    f_list = sorted(os.listdir("result"))
    plates = sorted([int(f.split(".")[0]) for f in f_list])
    f_list = sorted(os.listdir("plates"))
    true_plates = sorted([int(f.split(".")[0]) for f in f_list])

    os.chdir("../")
    classes = []
    # print(plates)
    # print(true_plates)
    for i in range(1, len(imgs) + 1):
        if i in true_plates:
            if np.random.rand() > 0.5:
                classes.append("{} 1\r\n".format(i))
            else:
                classes.append("{} 2\r\n".format(i))
        elif i in plates:
            if np.random.rand() > 0.5:
                classes.append("{} 2\r\n".format(i))
            else:
                classes.append("{} 1\r\n".format(i))
        else:
            classes.append("{} 3\r\n".format(i))
    # print(classes)
    with open('out.txt', 'w') as FILE0:
        FILE0.writelines(classes)

    print("took {:.3f} sec, # images={}".format(perf_counter() - t0, len(imgs)))
    print("-----------")
    print("output file is in {}".format(os.path.join(os.getcwd(), 'out.txt')))
    print("-----------")