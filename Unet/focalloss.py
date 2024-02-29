"""
這段程式定義了一個名為 Focal Loss 的 PyTorch 自定義損失函數。
Focal Loss 是一種針對類別不平衡問題設計的損失函數，通常用於解決
二元分類任務中存在嚴重類別不平衡的情況。它引入了兩個參數，即 alpha
和 gamma，用於調整正負樣本的權重和調整易分類樣本的權重。該程式計算
了二元交叉熵損失（Binary Cross Entropy Loss），並根據 Focal
Loss 的公式計算最終的損失值。根據 reduce 參數的設置，可以返回平均
損失或每個樣本的損失值。
"""
from torch import nn
import torch
from torch.nn import functional as F

class focal_loss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(focal_loss, self).__init__()
        # 初始化 Focal Loss 的參數
        self.alpha = alpha  # alpha 參數用於平衡正負樣本的權重
        self.gamma = gamma  # gamma 參數用於調整易分類樣本的權重
        self.logits = logits # 指定是否輸入 logits（預測值）還是概率值
        self.reduce = reduce  # 指定是否對每個批次進行損失縮減

    def forward(self, inputs, targets):
        # 計算 Binary Cross Entropy Loss
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        # 計算正樣本的預測概率
        pt = torch.exp(-BCE_loss)
        # 計算 Focal Loss
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss


        # 根據 reduce 參數返回損失值
        if self.reduce:
            return torch.mean(F_loss)  # 返回平均損失
        else:
            return F_loss # 返回每個樣本的損失值
