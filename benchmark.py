# import os
# import sys
import numpy as np

a = open("labels.txt").read().splitlines()
b = open("out.txt").read().splitlines()

a = np.array([int(i.split(" ")[1]) for i in a if len(i.split(" ")) > 1])
b = np.array([int(i.split(" ")[1]) for i in b if len(i.split(" ")) > 1])

correct = [1 for i in range(a.shape[0]) if a[i] == b[i]]

acc = len(correct) / a.shape[0]
print("accuracy = {:.2f}%".format(acc * 100))