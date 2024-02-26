from torch.utils.data import Dataset
import numpy as np
import torch
import random
import re
import os
from utils import loadTxt, loadDict
from PIL import Image
# from augment import CoronaryPolicy  # 原作者的


def default_loader(path):
    return Image.open(path)


class ImageDataSet(Dataset):
    def __init__(self, txt_path, img_dir, label_dir, loader=default_loader,transform=False, sort=False, seed=42):
        fileNames = [name for name in loadTxt(str(txt_path))]
        self.sort = sort
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.loader = loader
        self.seed = seed
        self.transform = transform

    def preprocess(self, img, size, label=False):
        img = img.resize(size, Image.BILINEAR)
        img = np.array(img)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]

        assert img.shape == (512, 512, 1)

        if label:
            img_trans = img.transpose((2, 0, 1))
            return img_trans
        else:
            img = img / 255.0
            return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]

        img = self.loader(str(self.img_dir / fileName))
        label = self.loader(str(self.label_dir / fileName))

        # if self.sort == False: # RandAug
        #     img,label = CoronaryPolicy()(img,label) # RandAug

        img = self.preprocess(img, (512, 512))
        label = self.preprocess(label, (512, 512), label=True)

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        

        return img,label

    def __len__(self):
        return len(self.fileNames)


class ImageDataSet1(Dataset):
    def __init__(self, labeled_txtPath, un_txtPath, imgDir, labelDir, loader=default_loader, transform=False,
                 sort=False):
        fileNames = [name for name in loadTxt(str(labeled_txtPath))] + [name for name in loadTxt(str(un_txtPath))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.imgDir = imgDir
        self.labelDir = labelDir
        self.loader = loader
        self.transform = transform

    def preprocess(self, img, size, label=False, DA_s=False, DA_w=False):
        img = img.resize(size, Image.BILINEAR)
        if DA_s:
            img = self.transform(img)

        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]
        
        if DA_w:
            if random.random() > 0.5:
                img = dataAugment(img, "gaussian_noise", (0.01, 0.0))

        assert img.shape == (512, 512, 1)

        if label:
            img = self.one_hot(img)
            return img.transpose((2, 0, 1))
        else:
            img = img / 255.0
            return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.imgDir / fileName))

        img1 = self.preprocess(img, (512, 512), label=False)
        img2 = self.preprocess(img, (512, 512), label=False, DA_s=True)

        if os.path.exists(str(self.labelDir / fileName)):
            label = self.loader(str(self.labelDir / fileName))
            label = self.preprocess(label, (512, 512), label=True)
        else:
            label = np.zeros((3, 512, 512))

        return torch.from_numpy(img1).float(), torch.from_numpy(img2).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.fileNames)


class ImageDataSet2(Dataset):
    def __init__(self, labeled_txt_path, un_txtPath, img_dir, loader=default_loader, transform=False, sort=False):
        fileNames = [name for name in loadTxt(str(un_txtPath))] + [name for name in loadTxt(str(labeled_txt_path))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.loader = loader
        self.transform = transform

    def preprocess(self, img, size, DA_s=False):
        img = img.resize(size, Image.BILINEAR)
        if DA_s:
            img = self.transform(img)

        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]

        assert img.shape == (512, 512, 1)

        img = img / 255.0
        return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))

        img1 = self.preprocess(img, (512, 512))
        img2 = self.preprocess(img, (512, 512), DA_s=True)

        return torch.from_numpy(img1).float(), torch.from_numpy(img2).float()

    def __len__(self):
        return len(self.fileNames)

class ImageDataSet_supervised_aug(Dataset):
    def __init__(self, txt_path, img_dir, label_dir, loader=default_loader, transform=False, sort=False, seed=42):
        fileNames = [name for name in loadTxt(str(txt_path))]
        self.sort = sort
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.loader = loader
        self.seed = seed
        self.transform = transform

    def preprocess(self, img, size, DA_s=False, label=False):
        img = img.resize(size, Image.BILINEAR)
        if DA_s:
            img = self.transform(img)
        img = np.array(img)
        
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]

        assert img.shape == (512, 512, 1)

        if label:
            img_trans = img.transpose((2, 0, 1))
            return img_trans
        else:
            img = img / 255.0
            return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]

        img = self.loader(str(self.img_dir / fileName))
        label = self.loader(str(self.label_dir / fileName))

        img = self.preprocess(img, (512, 512), DA_s=True)
        label = self.preprocess(label, (512, 512), label=True)

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        

        return img, label

    def __len__(self):
        return len(self.fileNames)
