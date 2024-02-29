"""
這段程式是用於計算影像分割模型的評估指標，包括 IOU（Intersection over Union）、灵敏度（Sensitivity）、特異度（Specificity）、精度（Precision）和 F1 分數。具體功能如下：
1.	iou_pytorch: 計算 IOU（Intersection over Union）指標，用於衡量預測結果和真實標籤的重疊程度。
2.	get_sensitivity: 計算灵敏度（Sensitivity），也稱為召回率（Recall），衡量預測中正類樣本被正確檢測出的比例。
3.	get_specificity: 計算特異度（Specificity），衡量預測中負類樣本被正確檢測出的比例。
4.	get_precision: 計算精度（Precision），衡量所有被預測為正類的樣本中，真正為正類的比例。
5.	get_F1: 計算 F1 分數，是精度和灵敏度的調和平均，用於綜合評估模型的性能。
這些指標對於評估影像分割模型的性能非常重要，可以幫助我們了解模型在預測中的準確度、召回率以及整體的分割效果。

"""

import torch

"""平滑值，用於避免除以零的情況"""
SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
	# You can comment out this line if you are passing tensors of equal shape
	# But if you are passing output from UNet or something it will most probably
	# be with the BATCH x 1 x H x W shape
	"""如果輸入張量的形狀不相等，將其壓縮為相等形狀"""
	outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

	"""計算交集和聯集"""
	intersection = (outputs & labels).float().sum((1, 2))    # 如果 Truth=0 或 Prediction=0 將為零
	union = (outputs | labels).float().sum((1, 2))    # 如果兩者都為 0 將為零

	"""計算 IoU"""
	iou = (intersection + SMOOTH) / (union + SMOOTH)   # 平滑處理以避免 0/0

	"""使用閾值進行二值化，用於分類"""
	thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # 相當於與閾值比較

	return thresholded

def get_sensitivity(SR, GT, threshold=0.5):
	"""獲取預測值的激活函數的 Sigmoid 輸出，並根據閾值進行二值化"""
	SR = torch.sigmoid(SR)
	SR = SR > threshold
	GT = GT == torch.max(GT)# 轉換成 one-hot 表示

	# TP : True Positive
	# FN : False Negative
	"""# 計算真正例（True Positive）和假負例（False Negative）"""
	TP = ((SR == 1).float() + (GT == 1).float()) == 2
	FN = ((SR == 0).float() + (GT == 1).float()) == 2


	"""計算灵敏度（Sensitivity，也稱為召回率）"""
	SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

	return SE


def get_specificity(SR, GT, threshold=0.5):
	"""獲取預測值的激活函數的 Sigmoid 輸出，並根據閾值進行二值化"""
	SR = torch.sigmoid(SR)
	SR = SR > threshold
	GT = GT == torch.max(GT)  # 轉換成 one-hot 表示

	# TN : True Negative
	# FP : False Positive
	"""計算真反例（True Negative）和假正例（False Positive）"""
	TN = ((SR == 0).float() + (GT == 0).float()) == 2 
	FP = ((SR == 1).float() + (GT == 0).float()) == 2

	"""計算特異度（Specificity）"""
	SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

	return SP


def get_precision(SR, GT, threshold=0.5):
	""" 獲取預測值的激活函數的 Sigmoid 輸出，並根據閾值進行二值化"""
	SR = torch.sigmoid(SR)
	SR = SR > threshold
	GT = GT == torch.max(GT)# 轉換成 one-hot 表示

	# TP : True Positive
	# FP : False Positive
	"""計算真正例（True Positive）和假正例（False Positive）"""
	TP = ((SR == 1).float() + (GT == 1).float()) == 2
	FP = ((SR == 1).float() + (GT == 0).float()) == 2

	"""計算精確度（Precision）"""
	PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

	return PC


def get_F1(SR, GT, threshold=0.5):
	"""計算灵敏度（Sensitivity，也稱為召回率）和精確度（Precision）"""
	# Sensitivity == Recall
	SE = get_sensitivity(SR, GT, threshold=threshold)
	PC = get_precision(SR, GT, threshold=threshold)

	"""計算 F1 分數"""
	F1 = 2 * SE * PC / (SE + PC + 1e-6)

	return F1
