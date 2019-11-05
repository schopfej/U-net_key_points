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

    # normalize to [0, 1]
    return g / np.max(g)


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, csv_file, root_dir, sig_kp=0.05, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = (pd.read_csv(csv_file)).dropna(axis=0, how="any").reset_index(drop=True)

        self.root_dir = root_dir
        self.transform = transform
        self.sig_kp = sig_kp

        # TODO: Filter out self.key_pts_frame for rows without wanted key points (pupil, left, right, ...)

    def __len__(self):

        #print(self.key_pts_frame)
        return len(self.key_pts_frame)



    def __getitem__(self, idx):


            image_name = os.path.join(self.root_dir,
                                      self.key_pts_frame.iloc[idx, 0])

            image = mpimg.imread(image_name)[..., None]
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            shape = image.shape

            # we take all key points xr, yr is the
            kx = self.key_pts_frame['xp'].loc[idx]
            ky = self.key_pts_frame['yp'].loc[idx]
            xr = self.key_pts_frame['xr'].loc[idx]
            yr = self.key_pts_frame['yr'].loc[idx]
            xl = self.key_pts_frame['xl'].loc[idx]
            yl = self.key_pts_frame['yl'].loc[idx]


            # apply augmentations
            if (self.transform is not None):
                # make deterministic so that all data have same transformation
                aug_det = self.transform.to_deterministic()

                key_pts = ia.KeypointsOnImage(
                    [ia.Keypoint(x=int(kx*image.shape[1]), y=int(ky * image.shape[0])),
                    ia.Keypoint(x=int(xr*image.shape[1]), y=int(yr * image.shape[0])),
                    ia.Keypoint(x=int(xl*image.shape[1]), y=int(yl * image.shape[0]))],
                    shape=(shape[0], shape[1]))
                image, key_pts = aug_det(image=image, keypoints=key_pts)

            # generate univariate gaussian
            kp_map = np.array([
                make_2d_gauss((image.shape[0], image.shape[1]),
                                self.sig_kp * np.max(shape), (kp.y, kp.x))
                for kp in key_pts.keypoints
            ])

            # put channel first
            image = np.rollaxis(image, -1, 0)
            sample = {'image': image, 'truth': kp_map}


            return sample
