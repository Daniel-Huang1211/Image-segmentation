import torch

SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
	# You can comment out this line if you are passing tensors of equal shape
	# But if you are passing output from UNet or something it will most probably
	# be with the BATCH x 1 x H x W shape
	outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

	intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
	union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

	iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

	thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

	return thresholded

def get_sensitivity(SR, GT, threshold=0.5):
	# Sensitivity == Recall
	SR = torch.sigmoid(SR)
	SR = SR > threshold
	GT = GT == torch.max(GT)

	# TP : True Positive
	# FN : False Negative
	TP = ((SR == 1).float() + (GT == 1).float()) == 2
	FN = ((SR == 0).float() + (GT == 1).float()) == 2

	SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

	return SE


def get_specificity(SR, GT, threshold=0.5):
	SR = torch.sigmoid(SR)
	SR = SR > threshold
	GT = GT == torch.max(GT)

	# TN : True Negative
	# FP : False Positive
	TN = ((SR == 0).float() + (GT == 0).float()) == 2 
	FP = ((SR == 1).float() + (GT == 0).float()) == 2

	SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

	return SP


def get_precision(SR, GT, threshold=0.5):
	SR = torch.sigmoid(SR)
	SR = SR > threshold
	GT = GT == torch.max(GT)

	# TP : True Positive
	# FP : False Positive
	TP = ((SR == 1).float() + (GT == 1).float()) == 2
	FP = ((SR == 1).float() + (GT == 0).float()) == 2

	PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

	return PC


def get_F1(SR, GT, threshold=0.5):
	# Sensitivity == Recall
	SE = get_sensitivity(SR, GT, threshold=threshold)
	PC = get_precision(SR, GT, threshold=threshold)

	F1 = 2 * SE * PC / (SE + PC + 1e-6)

	return F1
