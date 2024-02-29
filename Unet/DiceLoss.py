"""
這段程式定義了一個名為 BinaryDiceLoss 的自定義 PyTorch 損失函數。
該損失函數用於計算二元分類任務中的 Dice 損失。Dice 損失是衡量預測分割
結果與真實分割標籤之間相似度的一種常用指標。該程式可以根據預測值和目標值
計算出損失值，並支持不同的損失縮減方法，包括平均值、總和或不進行縮減。
"""
import self as self
import torch.nn as nn
import torch


class BinaryDiceLoss(nn.Module):
    """二元分類的 Dice 損失函數
       Args:
           smooth: 用於平滑損失的浮點數值，避免 NaN 錯誤，默認值: 1
           p: 分母值：\sum{x^p} + \sum{y^p}，默認值: 2
           predict: 預測張量，形狀為 [N, *]
           target: 與預測張量形狀相同的目標張量
           reduction: 應用的縮減方法，如果是 'mean'，則返回批次的平均值，
               如果是 'sum'，則返回總和，如果是 'none'，則返回形狀為 [N,] 的張量
       Returns:
           根據 reduction 參數返回損失張量
       Raise:
           如果縮減方法不符合預期，則引發異常
       """
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, k=0):
        """確保預測和目標張量的批次大小匹配"""
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

       """將預測和目標張量轉換為相同的形狀"""
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        """ 計算 Dice 損失的分子和分母"""
        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        """計算 Dice 損失"""
        loss = 1 - num / den
        loss = loss / (1 + k * (1 - loss))

        """根據指定的縮減方法返回損失值"""
        if self.reduction == 'mean':
            return loss.mean()

        elif self.reduction == 'sum':
            return loss.sum()

        elif self.reduction == 'none':
            return loss

        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
