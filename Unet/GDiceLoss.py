"""
這段程式定義了幾種不同版本的廣義 Dice 損失函數（Generalized Dice Loss），
用於分割任務中的模型訓練。這些不同版本的損失函數主要針對不同的情況進行優化和改進。
"""
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable

def flatten(tensor):
    """
    將給定的張量展平，使通道軸位於第一個位置。
    形狀轉換如下：
          (N, C, D, H, W) -> (C, N * D * H * W)
    """

    # 通道數量
    C = tensor.size(1)
    #新的軸順序
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # 轉置：(N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # 展平：(C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class _AbstractDiceLoss(nn.Module):
    """
    不同版本 Dice 損失的基礎類。
    """

    def __init__(self, weight=None, sigmoid_normalization=True):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # 網絡輸出的預期是未正規化的概率，我們想對 logits 進行正規化。
        # 由於 Dice（或在這種情況下的軟 Dice）通常用於二元數據，
        # 因此即使對於多類分割問題，使用 Sigmoid 進行通道正規化也是默認選擇。
        # 但是，如果希望應用 Softmax 以從輸出獲取正確的概率分佈，只需指定 sigmoid_normalization=False。
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def dice(self, input, target, weight):
        # 實際的 Dice 分數計算；由子類實現
        raise NotImplementedError

    def forward(self, input, target):
        # 從 logits 獲取概率
        input = self.normalization(input)

        #  計算每個通道的 Dice 系數
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # 平均所有通道/類別的 Dice 分數
        return 1. - torch.mean(per_channel_dice)

class GeneralizedDiceLoss(_AbstractDiceLoss):
    """計算廣義 Dice 損失 (GDL)，參考 https://arxiv.org/pdf/1707.03237.pdf。 """

    def __init__(self, sigmoid_normalization=True, epsilon=1e-6):
        super().__init__(weight=None, sigmoid_normalization=sigmoid_normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' 和 'target' 必須具有相同的形狀"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # 為了讓 GDL 有意義，我們需要至少 2 個通道(see https://arxiv.org/pdf/1707.03237.pdf)
            # 將前景和背景像素放在不同的通道中
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        #GDL 加權：每個標籤的貢獻由其體積的倒數校正
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())

class pGeneralizedDiceLoss(_AbstractDiceLoss):
    """計算修改版的廣義 Dice 損失 (GDL)，參考 https://arxiv.org/pdf/1707.03237.pdf。    """

    def __init__(self, sigmoid_normalization=True, epsilon=1e-6):
        super().__init__(weight=None, sigmoid_normalization=sigmoid_normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' 和 'target' 必須具有相同的形狀"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # 為了讓 GDL 有意義，我們需要至少 2 個通道(see https://arxiv.org/pdf/1707.03237.pdf)
            #  將前景和背景像素放在不同的通道中
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL 加權：每個標籤的貢獻由其體積的倒數校正
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)
        gdice = 2 * (intersect.sum() / denominator.sum())

        return gdice / (1 + 2.5*(1-gdice))

class GeneralizedDiceLoss_high_confidence(_AbstractDiceLoss):
    """計算修正的廣義 Dice 損失 (GDL)，參考 https://arxiv.org/pdf/1707.03237.pdf。  """

    def __init__(self, th, sigmoid_normalization=True, epsilon=1e-6):
        super().__init__(weight=None, sigmoid_normalization=sigmoid_normalization)
        self.epsilon = epsilon
        self.th = th

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' 和 'target' 必須具有相同的形狀"

        input = flatten(input)
        target = flatten(target)
        target = target.float()
        target = torch.sigmoid(target)
        target[target >= self.th] = 1
        target[target <= (1-self.th)] = 0
        statement = torch.logical_or(target >= self.th, target <= (1-self.th))
        input_1 = input[statement]
        target_1 = target[statement]
        # print(input_1)
        # print(target_1)

        if input.size(0) == 1:
            # 為了讓 GDL 有意義，我們需要至少 2 個通道(see https://arxiv.org/pdf/1707.03237.pdf)
            # 將前景和背景像素放在不同的通道中
            input_1 = torch.cat((input_1, 1 - input_1), dim=0)
            target_1 = torch.cat((target_1, 1 - target_1), dim=0)

        # GDL 加權：每個標籤的貢獻由其體積的倒數校正
        w_l = target_1.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input_1 * target_1).sum(-1)
        intersect = intersect * w_l

        denominator = (input_1 + target_1).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        score_1 = 2 * (intersect.sum() / denominator.sum())

        return score_1