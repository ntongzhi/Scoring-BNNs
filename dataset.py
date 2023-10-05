import numpy as np
import cv2
import random
from skimage.io import imread
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = imread(img_path)

        image = cv2.resize(image, (256, 256))
        mask_path = 'input/' + self.args.dataset + '/masks/' + img_path.split('\\')[-1].split('.')[0] + '.png'
        mask = imread(mask_path)

        mask = cv2.resize(mask, (256, 256))

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = mask[:, :, np.newaxis]

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=-1)

        image = image.astype('float32') / 255
        mask = mask.astype('float32') / 255

        if self.aug:
            if random.uniform(0, 1) > 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if random.uniform(0, 1) > 0.5:
                image = image[::-1, :, :].copy()
                mask = mask[::-1, :].copy()

        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        return torch.Tensor(image), torch.Tensor(mask)
