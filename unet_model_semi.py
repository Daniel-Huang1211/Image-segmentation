""" Full assembly of the parts to form the complete network """
"""這段程式碼定義了一個名為UNet的類別，該類別繼承自PyTorch中的nn.Module類別，用於構建UNet神經網絡模型。"""
#import torch.nn.functional as F
import torch.nn as nn
#from torch.distributions.uniform import Uniform
from unet_parts_semi import *
#from unet_parts_m import *

# 定義UNet類，繼承自nn.Module
class UNet(nn.Module):
    # 初始化函數
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()  # 調用父類的初始化函數

        self.n_channels = n_channels # 輸入圖像通道數
        self.n_classes = n_classes # 輸出類別數
        self.bilinear = bilinear # 是否使用雙線性插值
        self.inc = DoubleConv(n_channels, 64)  # 定義編碼器中的第一個卷積塊
        self.down1 = Down(64, 128) # 定義編碼器中的第一個下採樣步驟
        self.down2 = Down(128, 256) # 定義編碼器中的第二個下採樣步驟
        self.down3 = Down(256, 512) # 定義編碼器中的第三個下採樣步驟
        self.down4 = Down(512, 512) # 定義編碼器中的第四個下採樣步驟
        self.up1 = Up(1024, 256, bilinear) # 定義解碼器中的第一個上採樣步驟
        self.up2 = Up(512, 128, bilinear)  # 定義解碼器中的第二個上採樣步驟
        self.up3 = Up(256, 64, bilinear) # 定義解碼器中的第三個上採樣步驟
        self.up4 = Up(128, 64, bilinear) # 定義解碼器中的第四個上採樣步驟
        self.outc = OutConv(64, n_classes) # 定義輸出卷積層

    # 前向傳播函數
    def forward(self, x):
        x1 = self.inc(x)  # 第一個卷積塊的輸出
        x2 = self.down1(x1)  # 第一個下採樣步驟的輸出
        x3 = self.down2(x2)  # 第二個下採樣步驟的輸出
        x4 = self.down3(x3)  # 第三個下採樣步驟的輸出
        x5 = self.down4(x4)  # 第四個下採樣步驟的輸出
        x = self.up1(x5, x4) # 第一個上採樣步驟的輸出
        x = self.up2(x, x3) # 第二個上採樣步驟的輸出
        x = self.up3(x, x2) # 第三個上採樣步驟的輸出
        x = self.up4(x, x1) # 第四個上採樣步驟的輸出
        return self.outc(x) # 返回輸出卷積層的輸出

