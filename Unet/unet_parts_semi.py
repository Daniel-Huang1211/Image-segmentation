""" Parts of the U-Net model """
"""這段程式碼定義了 U-Net 模型的各個組件，包括雙重卷積（DoubleConv）、下採樣（Down）、上採樣（Up）和輸出卷積（OutConv）"""
import torch   # 導入 PyTorch 模組
import torch.nn as nn  # 導入 PyTorch 中的神經網絡模組
import torch.nn.functional as F # 導入 PyTorch 中的函數操作模組


class DoubleConv(nn.Module):  # 定義雙重卷積類別，繼承自 nn.Module
    """(convolution => [BN] => ReLU) * 2"""# 註釋：雙重卷積結構，包含兩個卷積操作以及可選的批量標準化和ReLU激活函數

    def __init__(self, in_channels, out_channels): # 初始化函數，設定輸入通道數和輸出通道數
        super().__init__()  # 繼承父類的初始化函數
        self.double_conv = nn.Sequential( # 定義一個由多個層組成的順序容器
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 2D 卷積層，用於特徵提取，設定卷積核大小為 3x3，填充為 1
            nn.BatchNorm2d(out_channels), # 批量標準化層，用於加速收斂和穩定模型訓練
            nn.ReLU(inplace=True),  # ReLU 激活函數，用於增加模型的非線性能力，inplace=True 表示原地操作，節省內存
            nn.Dropout(0.1),  # Dropout 正則化層，用於減少過擬合，防止神經元的過度依賴
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # 2D 卷積層，同樣用於特徵提取
            nn.BatchNorm2d(out_channels),  # 批量標準化層
            nn.ReLU(inplace=True),  # ReLU 激活函數
            nn.Dropout(0.1) # Dropout 正則化層
        )

    def forward(self, x):  # 定義前向傳播函數
        return self.double_conv(x)  # 返回雙重卷積結果


class Down(nn.Module): # 定義下採樣類別，繼承自 nn.Module
    """Downscaling with maxpool then double conv"""  # 註釋：下採樣操作，包含最大池化和雙重卷積

    def __init__(self, in_channels, out_channels):  # 初始化函數，設定輸入通道數和輸出通道數
        super().__init__()  # 繼承父類的初始化函數
        self.maxpool_conv = nn.Sequential(  # 定義一個由多個層組成的順序容器
            nn.MaxPool2d(2),  # 最大池化層，用於降低特徵圖尺寸，採樣因子為 2
            DoubleConv(in_channels, out_channels)  # 雙重卷積，用於特徵提取
        )

    def forward(self, x):  # 定義前向傳播函數
        return self.maxpool_conv(x)  # 返回下採樣結果


class Up(nn.Module):  # 定義上採樣類別，繼承自 nn.Module
    """Upscaling then double conv""" # 註釋：上採樣操作，包含上採樣和雙重卷積

    def __init__(self, in_channels, out_channels, bilinear=True):  # 初始化函數，設定輸入通道數、輸出通道數和是否使用雙線性插值上採樣
        super().__init__()  # 繼承父類的初始化函數

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:  # 如果使用雙線性插值上採樣
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 使用雙線性插值上採樣
        else:  # 如果不使用雙線性插值上採樣
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)  # 使用轉置卷積進行上採樣

        self.conv = DoubleConv(in_channels, out_channels)  # 雙重卷積，用於特徵提取
        self.dropout = nn.Dropout(0.1)  # Dropout 正則化層


    def forward(self, x1, x2):  # 定義前向傳播函數，接收上一層和跳躍連接的特徵作為輸入
        x1 = self.up(x1)  # 上採樣操作
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])  # 計算特徵圖尺寸之間的差異
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])  # 計算特徵圖尺寸之間的差異

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])  # 對特徵圖進行填充以匹配尺寸
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)  # 將上一層的特徵和上採樣後的特徵進行連接
        x = self.conv(x)  # 雙重卷積操作
        return self.dropout(x)  # 返回結果並進行 Dropout 正則化


class OutConv(nn.Module):  # 定義輸出卷積類別，繼承自 nn.Module
    def __init__(self, in_channels, out_channels):  # 初始化函數，設定輸入通道數和輸出通道數
        super(OutConv, self).__init__()  # 繼承父類的初始化函數
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) # 2D 卷積層，用於將特徵映射到輸出類別的分佈

    def forward(self, x):  # 定義前向傳播函數
        x = self.conv(x)  # 卷積操作
        return x # 返回結果
