import yaml
from os.path import join as pjoin
import configargparse
import scipy.ndimage.filters as fi
import numpy as np
from sklearn import preprocessing
import glob
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import cm
from torch.utils.data import Dataset, DataLoader
from scratch import UNet
from torchvision import transforms, utils
from data_load import Rescale, RandomCrop, Normalize, ToTensor, FacialKeypointsDataset
from imgaug import augmenters as iaa
from imgaug import parameters as iap

def rescale_images(images, random_state, parents, hooks):

    result = []
    for image in images:
        image_aug = np.copy(image)
        if (image.dtype == np.uint8):
            image_aug = image_aug / 255
        result.append(image_aug)
    return result


void_fun = lambda x, random_state, parents, hooks: x

rescale_augmenter = iaa.Lambda(
    func_images=rescale_images,
    func_heatmaps=void_fun,
    func_keypoints=void_fun)


def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0],
                predicted_key_pts[:, 1],
                s=20,
                marker='.',
                c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


# visualize the output
# by default this shows a batch of 10 images
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    for i in range(batch_size):
        plt.figure(figsize=(20, 10))
        ax = plt.subplot(1, batch_size, i + 1)
        # un-transform the image data
        image = test_images[i].data  # get the image from it's Variable wrapper
        image = image.numpy()  # convert to numpy array from a Tensor
        image = np.transpose(
            image, (1, 2, 0))  # transpose to go from torch to numpy image
        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints
        predicted_key_pts = predicted_key_pts * 50.0 + 100

        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts * 50.0 + 100

        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts,
                           ground_truth_pts)

        plt.axis('off')
    plt.show()


## TODO: Define the loss and optimization
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

for i, sample_size in enumerate(transformed_dataset):
    if i == 0:
        images = sample['image'].squeeze_()
        size_x, size_y = images.shape[:2]
        print(size_x)
        resized_coord = sample['keypoints'].squeeze_()
        break

for i, sample in enumerate(transformed_dataset):
    # get sample data: images and ground truth keypoints
    images = sample['image']
    # key_pts = sample['keypoints'].float()
    key_pts = sample['keypoints']
    # convert images to FloatTensors
    # images = images.type(torch.FloatTensor)
    # print("this sample has the size:", sample['keypoints'].shape)
    print("now a little bit further:", sample['keypoints'])
    # forward pass to get net output
    if i > 0:
        break
    keypoints_3 = np.array(sample['keypoints'])
    prob = gkern2(tensor_coord=keypoints_3.astype("int"),
                  size_x=cfg.in_shape,
                  size_y=cfg.in_shape,
                  sig_x=cfg.sig_x,
                  sig_y=cfg.sig_y)

    nx, ny = prob.shape[1], prob.shape[0]
    x = range(nx)
    y = range(ny)
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    ha.view_init(30, 100)
    X, Y = np.meshgrid(x, y)
    ha.plot_surface(X, Y, prob, cmap=cm.summer)
    plt.title('Normalized gauss filter of size 21x21')
    plt.show()

#criterion = nn.BCEWithLogitsLoss
#criterion1 = nn.Sigmoid()
criterion = nn.CrossEntropyLoss()
#criterion = nn.L1Loss
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
#optimizer = optim.Adam(params=net.parameters(), lr=0.0025)
softm = nn.Softmax(dim=1)
net = UNet(2, in_channels=3, depth=5, merge_mode='concat')


def train(cfg):

    if(not os.path.exists(cfg.out_dir)):
        os.makedirs(cfg.out_dir)

    with open(pjoin(cfg.out_dir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)


    net = UNet(cfg.out_channels,
               in_channels=cfg.in_channels,
               depth=cfg.depth,
               merge_mode=cfg.merge_mode)

    # make augmenter
    transf = iaa.Sequential([
        iaa.SomeOf(3, [
            iaa.Affine(scale={
                "x": (1 - cfg.aug_scale,
                      1 + cfg.aug_scale),
                "y": (1 - cfg.aug_scale,
                      1 + cfg.aug_scale)},
            iaa.Affine(rotate=iap.Uniform(-cfg.aug_rotate, cfg.aug_rotate)),
            iaa.Affine(shear=iap.Uniform(-cfg.aug_shear, cfg.aug_shear)),
            iaa.Fliplr(1.),
            iaa.Flipud(1.),
            iaa.GaussianBlur(sigma=iap.Uniform(0.0, cfg.aug_gaussblur))
        ]),
        iaa.Resize(cfg.in_shape), rescale_augmenter
    ])

    my_transforms = transforms.Compose(
        [Rescale(cfg.in_shape), Normalize(),
         ToTensor()])
    base_loader = FacialKeypointsDataset(csv_file=cfg.csv_file,
                                       root_dir=cfg.root_dir,
                                       transform=my_transforms)

    # build train, val and test sets randomly with given split ratios
    idxs = np.random.shuffle(np.arange(len(base_loader)))
    train_idxs = idxs[:int(len(base_loader) * cfg.train_split)]
    others_idxs = idxs[int(len(base_loader) * cfg.train_split):]
    val_idxs = others_idxs[:int(len(base_loader) * cfg.val_split)]
    test_idxs = others_idxs[int(len(base_loader) * cfg.val_split):]

    train_loader = torch.utils.data.Subset(base_loader, train_idxs)
    val_loader = torch.utils.data.Subset(base_loader, val_idxs)
    test_loader = torch.utils.data.Subset(base_loader, test_idxs)

    loaders = {'train': DataLoader(train_loader,
                                   batch_size=cfg.batch_size,
                                   num_workers=cfg.n_workers)
               'val': DataLoader(train_loader,
                                   batch_size=cfg.batch_size,
                                   num_workers=cfg.n_workers),
               'test': DataLoader(train_loader,
                                  num_workers=cfg.n_workers)}
    train_loader = torch
    # prepare the net for training
    for epoch in range(cfg.n_epochs):  # loop over the dataset multiple times
        for phase in ['train', 'val', 'test']:
            running_loss = 0.0
            # train on batches of data, assumes you already have train_loader
            for batch_i, data in enumerate(loaders[phase]):
                # get the input images and their corresponding labels
                images = data['image']
                prob = preprocessing.minmax_scale(prob, feature_range=(0, 1))

                prob = torch.from_numpy(prob.reshape(-1))
                #prob = prob.long()

                # x = Variable(torch.FloatTensor(1, 1, images))
                images = images.type(torch.FloatTensor)
                # print("images after torch",images)
                # forward pass to get outputs)
                optimizer.zero_grad()
                out = net(images)

            ################## Shape of output #########################################################
            ############################################################################################
            # permute is like np.transpose: (N, C, H, W) => (H, W, N, C)
            # contiguous is required because of this issue: https://github.com/pytorch/pytorch/issues/764
            # view: reshapes the output tensor so that we have (H * W * N, num_class)
            # NOTE: num_class == C (number of output channels)
            # output = output.permute(2, 3, 0, 1).contiguous().view(-1, num_classes)
            out = out.permute(2, 3, 0, 1).contiguous().view(-1, 2)
            ### ###########################
            ## Cross entropy (input,target/labels)
            ###################################
            ###### Input: (N,C) where C = number of classes
            ########## Target: (N) where each value is 0≤targets[i]≤C−1
            ########## Output: scalar
            prob = prob.squeeze_().long()
            out = out.squeeze_()
            # out = out.reshape(-1, 1)
            # out = out.permute(2, 3, 0, 1).contiguous().view(-1, 3)
            #out = softm(out)
            # torch.sum(out)

            #out = criterion1(out)
            #out = softm(out)
            ###########activation

            #out = F.sigmoid(out)
            loss = criterion(out, prob)
            # loss.backward()
            #optimizer.zero_grad()  # otherwise the parameters would be cumulated...
            # backward pass to calculate the weight gradients
            loss.backward(
            )  ###############################################calculate gradients
            # update the weights
            optimizer.step()
            net.eval()
            # print loss statistics
            print(loss.item)
            running_loss += loss.item()
            if batch_i % 10 == 9:  # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(
                    epoch + 1, batch_i + 1, running_loss))
                running_loss = 0.0
                params = list(net.parameters())
                print("the len of the param", len(params))
                print("the param size is", params[0].size())  # conv1's .weight
                print(params[0])
    print('Finished Training')


if __name__ == "__main__":

    p = configargparse.ArgParser()

    #Paths, dirs, names ...
    p.add('--csv-file', type=str, required=True)
    p.add('--root-dir', type=str, required=True)
    p.add('--out-dir', type=str, required=True)
    p.add('--checkpoint-file', type=str)

    cfg = p.parse_args()

    main(cfg)
