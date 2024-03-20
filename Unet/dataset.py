"""
這段程式碼定義了幾個不同的自定義資料集類別（ImageDataSet、ImageDataSet1、ImageDataSet2 和 ImageDataSet_supervised_aug），用於加載圖像資料和標籤，並將它們準備成 PyTorch 模型可以處理的格式。
主要功能包括：
1.	ImageDataSet：用於加載帶有圖像和標籤的資料集。可以根據設置的參數對圖像進行大小調整和預處理。
2.	ImageDataSet1：用於加載帶有圖像但不一定帶有標籤的資料集。同樣可以進行圖像的大小調整和預處理。
3.	ImageDataSet2：與 ImageDataSet1 類似，但是用途不同。可能用於不需要標籤的情況。
4.	ImageDataSet_supervised_aug：用於加載帶有圖像和標籤的資料集，並進行圖像的大小調整和預處理，同時也可以對圖像進行額外的資料增強。
這些資料集類別能夠將圖像和標籤封裝成 PyTorch 的 Dataset 對象，以便於在模型訓練過程中進行數據的加載和處理。

"""
import img as img
from torch.utils.data import Dataset# 導入 PyTorch Dataset 類
import numpy as np
import torch
import random
import re
import os
from utils import loadTxt, loadDict  # 導入自定義的工具函數
from PIL import Image  # 導入 PIL 库中的 Image 模組
# from augment import CoronaryPolicy  # 原作者的


def default_loader(path):
    return Image.open(path)  # 默認的圖像加載函數



class ImageDataSet(Dataset):
    def __init__(self, txt_path, img_dir, label_dir, loader=default_loader,transform=False, sort=False, seed=42):
        fileNames = [name for name in loadTxt(str(txt_path))]  # 讀取文本文件中的檔案名稱列表
        self.sort = sort  # 是否對檔案名稱列表進行排序
        if sort:
            fileNames.sort()
        self.fileNames = fileNames  # 檔案名稱列表
        self.img_dir = img_dir  # 圖像檔案所在目錄
        self.label_dir = label_dir  # 標籤檔案所在目錄
        self.loader = loader  # 圖像加載函數
        self.seed = seed  # 隨機種子
        self.transform = transform  # 是否進行圖像轉換

    def preprocess(self, img, size, label=False):
        img = img.resize(size, Image.BILINEAR))  # 調整圖像大小
        img = np.array(img)  # 將圖像轉換為 NumPy 陣列


        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)  # 擴展圖像的通道數

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis] # 如果圖像具有多個通道，只保留第一個通道


        assert img.shape == (512, 512, 1)  # 確保圖像形狀符合預期

        # 將圖像轉換為 PyTorch 預期的形狀
        # 返回處理後的圖像
        if label:
            img_trans = img.transpose((2, 0, 1))
            return img_trans
        # 將圖像像素值標準化
        # 返回處理後的圖像
        else:
            img = img / 255.0
            return img.transpose((2, 0, 1))


    def __getitem__(self, idx):
        fileName = self.fileNames[idx]  # 獲取指定索引的檔案名稱

        img = self.loader(str(self.img_dir / fileName))  # 加載圖像
        label = self.loader(str(self.label_dir / fileName))  # 加載標籤

        # if self.sort == False: # RandAug
        #     img,label = CoronaryPolicy()(img,label) # RandAug

        img = self.preprocess(img, (512, 512)) # 對圖像進行預處理
        label = self.preprocess(label, (512, 512), label=True)  # 對標籤進行預處理

        img = torch.from_numpy(img)  # 將 NumPy 陣列轉換為 PyTorch 張量
        label = torch.from_numpy(label)  # 將 NumPy 陣列轉換為 PyTorch 張量
        

        return img,label # 返回圖像和標籤對

    def __len__(self):
        return len(self.fileNames)  # 返回資料集的大小


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

    def preprocess(self, img, size, label=False, DA_s=False, DA_w=False): # 將圖片調整為指定大小
        img = img.resize(size, Image.BILINEAR)
        if DA_s:  # 若需要進行增強操作，則進行增強
            img = self.transform(img)

        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]
        
        if DA_w:# 若需要進行另一種增強操作，則進行增強
            if random.random() > 0.5:
                img = dataAugment(img, "gaussian_noise", (0.01, 0.0))

        assert img.shape == (512, 512, 1)

        if label:   # 對標籤圖像進行 one-hot 編碼
            img = self.one_hot(img)
            return img.transpose((2, 0, 1))
        else:  # 正規化圖片數據並轉換維度順序
            img = img / 255.0
            return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]     # 獲取文件名
        img = self.loader(str(self.imgDir / fileName)) # 讀取圖片和標籤

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
        img = img.resize(size, Image.BILINEAR)   # 將圖片調整為指定大小
        if DA_s: # 若需要進行增強操作，則進行增強
            img = self.transform(img)

        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]

        assert img.shape == (512, 512, 1)

        img = img / 255.0  # 正規化圖片數據並轉換維度順序
        return img.transpose((2, 0, 1))

    def __getitem__(self, idx):   # 獲取文件名
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))  # 讀取圖片

        # 處理圖片數據
        img1 = self.preprocess(img, (512, 512))
        img2 = self.preprocess(img, (512, 512), DA_s=True)

        return torch.from_numpy(img1).float(), torch.from_numpy(img2).float()

    def __len__(self):
        return len(self.fileNames)

class ImageDataSet_supervised_aug(Dataset):
    def __init__(self, txt_path, img_dir, label_dir, loader=default_loader, transform=False, sort=False, seed=42):
        fileNames = [name for name in loadTxt(str(txt_path))] # 讀取檔案名稱列表
        self.sort = sort
        if sort:   # 如果需要排序，則對文件名進行排序
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.loader = loader
        self.seed = seed
        self.transform = transform

    def preprocess(self, img, size, DA_s=False, label=False):
        img = img.resize(size, Image.BILINEAR)   # 將圖片調整為指定大小
        if DA_s:   # 若需要進行增強操作，則進行增強
            img = self.transform(img)
        img = np.array(img)
        
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]

        assert img.shape == (512, 512, 1)

        if label:# 將圖片轉換維度順序
            img_trans = img.transpose((2, 0, 1))
            return img_trans
        else:  # 正規化圖片數據並轉換維度順序
            img = img / 255.0
            return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]  # 獲取文件名

        # 讀取圖片和標籤
        img = self.loader(str(self.img_dir / fileName))
        label = self.loader(str(self.label_dir / fileName))

        # 處理圖片數據
        img = self.preprocess(img, (512, 512), DA_s=True)
        label = self.preprocess(label, (512, 512), label=True)

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        

        return img, label

    def __len__(self):
        return len(self.fileNames)
