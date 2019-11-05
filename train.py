import yaml
from os.path import join as pjoin
import configargparse
import numpy as np
import glob
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import cm
from torchvision import utils as tutls
from torch.utils.data import Dataset, DataLoader
from unet import UNet
from data_load import FacialKeypointsDataset
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import datetime
from tensorboardX import SummaryWriter
import tqdm
import params
from skimage import draw


def save_checkpoint(dict_,
                    is_best,
                    path,
                    fname_cp='checkpoint.pth.tar',
                    fname_bm='best_model.pth.tar'):

    cp_path = os.path.join(path, fname_cp)
    bm_path = os.path.join(path, fname_bm)

    if (not os.path.exists(path)):
        os.makedirs(path)

    try:
        state_dict = dict_['model'].module.state_dict()
    except AttributeError:
        state_dict = dict_['model'].state_dict()

    torch.save(state_dict, cp_path)

    if (is_best):
        shutil.copyfile(cp_path, bm_path)


def load_checkpoint(path, model, gpu=False):

    if (gpu):
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage)
    else:
        # checkpoint = torch.load(path)
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    return model


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


def train(cfg):

    # make run_dir with date

    d = datetime.datetime.now()
    run_dir = pjoin(cfg.out_dir, 'exp_{:%Y-%m-%d_%H-%M}'.format(d))

    if(not os.path.exists(run_dir)):
        os.makedirs(run_dir)

    with open(pjoin(run_dir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    # generate tensorboard object
    writer = SummaryWriter(run_dir)

    net = UNet(cfg.out_channels,
               in_channels=cfg.in_channels,
               depth=cfg.depth,
               merge_mode=cfg.merge_mode)

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    optimizer = optim.Adam(params=net.parameters(), lr=0.0025, weight_decay=0.00001)
    #softm = nn.Softmax(dim=1)
    softm = nn.Sigmoid()                                     ################################

    # make augmenter
    transf = iaa.Sequential([
        iaa.SomeOf(3, [
            iaa.Affine(rotate=iap.Uniform(-cfg.aug_rotate, cfg.aug_rotate)),
            iaa.Affine(shear=iap.Uniform(-cfg.aug_shear, cfg.aug_shear)),
            iaa.Fliplr(1.),
            iaa.Flipud(1.),
            iaa.GaussianBlur(sigma=iap.Uniform(0.0, cfg.aug_gaussblur))
        ]),
        iaa.Resize(cfg.in_shape), rescale_augmenter
    ])

    base_loader = FacialKeypointsDataset(csv_file=cfg.csv_file,
                                         root_dir=cfg.root_dir,
                                         sig_kp=cfg.sig_kp,
                                         transform=transf)

    # build train, val and test sets randomly with given split ratios
    idxs = np.arange(len(base_loader))
    np.random.shuffle(idxs)
    train_idxs = idxs[:int(len(base_loader) * cfg.train_split)]
    others_idxs = idxs[int(len(base_loader) * cfg.train_split):]
    val_idxs = others_idxs[:int(others_idxs.size * cfg.val_split)]
    test_idxs = others_idxs[int(others_idxs.size * cfg.val_split):]

    train_loader = torch.utils.data.Subset(base_loader, train_idxs)
    val_loader = torch.utils.data.Subset(base_loader, val_idxs)
    test_loader = torch.utils.data.Subset(base_loader, test_idxs)

    loaders = {'train': DataLoader(train_loader,
                                   batch_size=cfg.batch_size,
                                   num_workers=cfg.n_workers),
               'val': DataLoader(train_loader,
                                   batch_size=cfg.batch_size,
                                   num_workers=cfg.n_workers),
               'test': DataLoader(train_loader,
                                  batch_size=cfg.batch_size,
                                  num_workers=cfg.n_workers)}
    # convert batch to device
    device = torch.device('cuda' if cfg.cuda else 'cpu')

    net.to(device)

    batch_to_device = lambda batch: {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    for epoch in range(cfg.n_epochs):  # loop over the dataset multiple times

        for phase in loaders.keys():
            if phase == 'train':
                net.train()

            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0
            # train on batches of data, assumes you already have train_loader
            pbar = tqdm.tqdm(total=len(loaders[phase]))
            for i, data in enumerate(loaders[phase]):
                data = batch_to_device(data)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #import pdb; pdb.set_trace()

                    out = softm(net(data['image'].float()))

                    #data_truth = softm(torch.squeeze(data['truth'])) ## here I am not sure, bc the entropy fct already has softmax
                    #data_truth = data_truth.long().reshape(4, 224*224)
                    #out = out.reshape(4, 1, 224*224)
                    # data_truth should have shape 4, 244,244
                    # input matrix is in the shape: (Minibatch, Classes, H, W)
                    # the target is in size (Minibatch, H, W)
                    loss = criterion(out, data['truth'].float())
                                                   #############################
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.cpu().detach().numpy()
                loss_ = running_loss / ((i + 1) * cfg.batch_size)
                pbar.set_description('[{}] loss: {:.4f}'.format(phase, loss_))
                pbar.update(1)

            pbar.close()
            writer.add_scalar('{}/loss'.format(phase),
                              loss_,
                              epoch)

        # make preview images
        if phase == 'test':
            data = next(iter(loaders[phase]))
            data = batch_to_device(data)
            pred_ = softm(net(data['image'].float())).cpu()

            im_ = data['image'].float().cpu().detach()
            im_ = [torch.cat(3*[im_[i, ...]]) for i in range(im_.shape[0])]
            truth_ = data['truth'].float().cpu()
            truth_ = [
                torch.cat([truth_[i, ...]])
                for i in range(truth_.shape[0])
            ]

            # normalize prediction maps in-place
            for b in range(pred_.shape[0]):
                for n in range(pred_.shape[1]):
                    pred_[b, n, ...] = (pred_[b, n, ...] - pred_[b, n, ...].min())
                    pred_[b, n, ...] = pred_[b, n, ...] / pred_[b, n, ...].max()

            # find max location on each channel of each batch element
            pos = []
            for b in range(pred_.shape[0]):
                pos.append([])
                for n in range(pred_.shape[1]):
                    idx_max = pred_[b, n, ...].argmax()
                    i, j = np.unravel_index(idx_max, pred_[b, n, ...].shape)
                    pos[-1].append((i, j))
                    # draw circle on image through numpy :(
                    rr, cc = draw.circle(i, j, 5, shape=im_[b][n, ...].shape)
                    im__ = np.rollaxis(im_[b].detach().numpy(), 0, 3)
                    im__[rr, cc, ...] = (1., 0., 0.)
                    im_[b] = torch.from_numpy(np.rollaxis(im__, -1, 0))

            pred_ = [torch.cat([pred_[i, ...]]) for i in range(pred_.shape[0])]
            #import pdb; pdb.set_trace()
            all_ = [
                tutls.make_grid([im_[i], truth_[i], pred_[i]],
                                nrow=len(pred_),
                                padding=10,
                                pad_value=1.)
                for i in range(len(truth_))
            ]
            all_ = torch.cat(all_, dim=1)
            writer.add_image('test/img', all_, epoch)

        # save checkpoint
        if phase == 'val':
            is_best = False
            if (loss_ < best_loss):
                is_best = True
                best_loss = loss_
            path = pjoin(run_dir, 'checkpoints')
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': net,
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict()
                },
                is_best,
                path=path)


if __name__ == "__main__":

    #import pdb; pdb.set_trace() ## DEBUG ##
    p = params.get_params()

    #Paths, dirs, names ...
    p.add('--csv-file', type=str, required=True)
    p.add('--root-dir', type=str, required=True)
    p.add('--out-dir', type=str, required=True)
    p.add('--checkpoint-file', type=str)

    cfg = p.parse_args()

    train(cfg)
