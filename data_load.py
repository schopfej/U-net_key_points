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


def make_1d_gauss(length, std, x0):

    x = np.arange(length)
    y = np.exp(-0.5 * ((x - x0) / std)**2)

    return y / np.sum(y)


def make_2d_gauss(shape, std, center):
    """
    Make object prior (gaussians) on center
    """

    g = np.zeros(shape)
    g_x = make_1d_gauss(shape[1], std, center[1])
    g_x = np.tile(g_x, (shape[0], 1))
    g_y = make_1d_gauss(shape[0], std, center[0])
    g_y = np.tile(g_y.reshape(-1, 1), (1, shape[1]))

    g = g_x * g_y

    return g / np.sum(g)


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, csv_file, root_dir, sig_map=15, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.sig_map = sig_map

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                  self.key_pts_frame.iloc[idx, 0])

        image = mpimg.imread(image_name)

        # if image has an alpha color channel, get rid of it
        #if (image.shape[2] == 4):
        #    image = image[:, :, 0:3]

        shape = image.shape

        key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1, 2)

        # apply augmentations
        if (self.augmentations is not None):
            # make deterministic so that all data have same transformatoin
            aug_det = self.augmentations.to_deterministic()

            key_pts = ia.KeypointsOnImage(
                [ia.Keypoint(x=p[1], y=p[0]) for p in key_pts],
                shape=(shape[0], shape[1]))
            keypoints = aug_det.augment_keypoints([keypoints])[0]

            # generate univariate gaussian
            kp_maps = np.array([
                make_2d_gauss((shape[0], shape[1]),
                              self.sig_map * np.max(shape), (kp.y, kp.x))
                for kp in keypoints.keypoints
            ])

            image = aug_det.augment_images([image])[0]

        sample = {'image': image, 'truth': kp_maps}

        return sample
