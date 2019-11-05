import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import matplotlib.pyplot as plt


t1 = torch.tensor([[[[0.3699, 0.3584, 0.4940, 0.8618], ##########here max value
          [0.6767, 0.7439, 0.5984, 0.5499],
          [0.8465, 0.7276, 0.3078, 0.3882],
          [0.1001, 0.0705, 0.2007, 0.4051]]],


        [[[0.7520, 0.4528, 0.0525, 0.9253],
          [0.6946, 0.0318, 0.5650, 0.7385],
          [0.0671, 0.6493, 0.3243, 0.2383],
          [0.6119, 0.7762, 0.9687, 0.0896]]],           ##########here is argmax at place 30


        [[[0.3504, 0.7431, 0.8336, 0.0336],
          [0.8208, 0.9051, 0.1681, 0.8722],
          [0.5751, 0.7903, 0.0046, 0.1471],
          [0.4875, 0.1592, 0.2783, 0.6338]]],


        [[[0.9398, 0.7589, 0.6645, 0.8017],
          [0.9469, 0.2822, 0.9042, 0.2516],
          [0.2576, 0.3852, 0.7349, 0.2806],
          [0.7062, 0.1214, 0.0922, 0.1385]]]])



n, _, w, h = t1.shape
#x = torch.rand(n, 1, w, h)
m = t1.view(n, -1).argmax(1)
indices = torch.cat(((m / w).view(-1, 1), (m % h).view(-1, 1)), dim=1)

for i, row in enumerate(indices):
#        print(i, int(row[0]), int(row[1]))
        t1[i][0, int(row[0])][int(row[1])] = 1000


#t1[3][0,3][0] = 1000


