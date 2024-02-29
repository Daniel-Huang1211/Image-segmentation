"""
1.處理文本文件的讀取和保存。
2.對字典對象進行序列化和反序列化操作。
3.定義不同類型的學習率調整函數，包括指數型和線性型的 rampup 函數。
4.對模型參數進行指數滑動平均。
5.獲取帶有特定參數設置的 SGD 優化器。
6.更新模型中的 Batch Normalization 層參數。
7.將預測結果與標籤進行比較，並生成彩色圖像以可視化差異。
8.將圖像、標籤和預測結果組合成一個網格並保存為圖像文件。
"""
import pickle # 導入 pickle 模塊，用於對象的序列化和反序列化操作
import torch  # 導入 PyTorch 框架
import math  # 導入數學庫，用於數學計算
import numpy as np  # 導入 NumPy 模塊，用於數組操作
from torch.optim.lr_scheduler import LambdaLR  # 從 PyTorch 導入學習率調整器 LambdaLR
from PIL import Image  # 導入圖像處理庫 PIL
import matplotlib.pyplot as plt  # 導入 matplotlib 用於繪圖
import os  # 導入操作系統庫，用於文件路徑等操作

""" 定義創建訓練歷史的函數，初始化訓練和驗證過程中的關鍵詞 """
def createTrainHistory(keywords):
    history = {"train": {}, "valid": {}}  # 創建包含訓練和驗證歷史的字典
    for words in keywords:
        history["train"][words] = list()  # 初始化訓練過程中關鍵詞的歷史列表
        history["valid"][words] = list()  # 初始化驗證過程中關鍵詞的歷史列表
    return history  # 返回歷史字典

"""加載文本文件，返回文件內容列表"""
def loadTxt(filename):
    f = open(filename)
    context = list()
    for line in f:
        context.append(line.replace("\n", ""))
    return context

"""將字典對象保存到文件中"""
def saveDict(filename, data):
    file = open(filename, 'wb')
    pickle.dump(data, file)
    file.close()

""" 從文件中加載字典對象"""
def loadDict(fileName):
    with open(fileName, 'rb') as handle:
        data = pickle.load(handle)
    return data

"""將文件名列表保存到指定文件中"""
def saveTxt(filenamesList, saveName):
    fp = open(saveName, "a")
    for name in filenamesList:
        fp.write(name + "\n")
    fp.close()

"""定義指數型的學習率 rampup 函數"""
def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""

    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0

    return warpper

"""解碼輸入數據，將其分為有監督和無監督部分"""
def decode_inputs(inputs):
    imgs_s, imgs_t, labels = inputs  # 解析輸入數據
    batch_size = labels.shape[0]  # 獲取批量大小
    imgs_s_s = list()  # 存儲有監督源圖像的列表
    imgs_t_s = list()  # 存儲有監督目標圖像的列表
    label_s = list()  # 存儲有監督標籤的列表
    imgs_s_u = list()  # 存儲無監督源圖像的列表
    imgs_t_u = list() # 存儲無監督目標圖像的列表

    for batch_idx in range(batch_size):
        if torch.sum(labels[batch_idx]).item() != 0:
            label_s.append(labels[batch_idx].unsqueeze(0))
            imgs_s_s.append(imgs_s[batch_idx].unsqueeze(0))
            imgs_t_s.append(imgs_t[batch_idx].unsqueeze(0))
        else:
            imgs_s_u.append(imgs_s[batch_idx].unsqueeze(0))
            imgs_t_u.append(imgs_t[batch_idx].unsqueeze(0))

    if len(imgs_s_u) == 0 and len(imgs_s_u) == 0:

        return (torch.cat([img for img in imgs_s_s], 0),
                torch.cat([img for img in imgs_t_s], 0),
                torch.cat([label for label in label_s], 0))

    elif len(imgs_s_s) == 0 and len(imgs_s_s) == 0:
        return (torch.cat([img for img in imgs_s_u], 0),
                torch.cat([img for img in imgs_t_u], 0))

    else:
        return (torch.cat([img for img in imgs_s_s], 0),
                torch.cat([img for img in imgs_t_s], 0),
                torch.cat([label for label in label_s], 0),
                torch.cat([img for img in imgs_s_u], 0),
                torch.cat([img for img in imgs_t_u], 0))

"""獲取帶有特定參數設置的 SGD 優化器"""
def get_SGD(net, name='SGD', lr=0.1, momentum=0.9, \
            weight_decay=5e-4, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    optim = getattr(torch.optim, name)

    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]

    optimizer = optim(per_param_args, lr=lr,
                      momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer


"""使用指數滑動平均更新模型參數"""
def update_ema(ema_model, model, alpha):
	# alpha = min(1 - 1 / (epoch + 1), alpha)
	for ema_param, param in zip(ema_model.parameters(), model.parameters()):
		ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
'''
def update_ema(ema_model, model, alpha):
    for params_train, params_eval in zip(model.parameters(), ema_model.parameters()):
        params_eval.copy_(params_eval * alpha + params_train.detach() * (1 - alpha))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)
'''

"""#定義線性型的學習率 rampup 函數"""
def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

""" 更新模型中的 Batch Normalization 層參數"""
def update_bn(model, ema_model):
    for m2, m1 in zip(ema_model.named_modules(), model.named_modules()):
        if ('bn' in m2[0]) and ('bn' in m1[0]):
            bn2, bn1 = m2[1].state_dict(), m1[1].state_dict()
            bn2['running_mean'].data.copy_(bn1['running_mean'].data)
            bn2['running_var'].data.copy_(bn1['running_var'].data)
            bn2['num_batches_tracked'].data.copy_(bn1['num_batches_tracked'].data)


"""返回帶有 warmup 的余弦退火學習率調整器"""
def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    
    
    #Get cosine scheduler (LambdaLR).
    #if warmup is needed, set num_warmup_steps (int) > 0.
    

    def _lr_lambda(current_step):
        
        #_lr_lambda returns a multiplicative factor given an interger parameter epochs.
        #Decaying criteria: last_epoch
        

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)
'''
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
'''

"""對預測結果應用空間增強"""
def space_augment_on_predicted(pred, work_q_list):
    batch = pred.shape[0]
    tmp = []
    for i in range(batch):
        op, val = work_q_list[i]
        pred1 = op(Image.fromarray(np.uint8(pred[i, 0, :, :])), val)
        tmp.append(np.array(pred1))
#         pred2 = Image.fromarray(np.array(pred1) * 255)
#         pred2.save("pred1.png")
#         input()
    pred = np.array(tmp)
    pred = np.expand_dims(pred, axis=1)

    return torch.tensor(pred, requires_grad=True, dtype=torch.float32)

"""將預測結果與標籤進行比較，生成彩色圖像以可視化差異"""
def predicted_color_result(img, pred, label, threshold=0.5):
    img = img[0, 0, :, :]
    pred = pred[0, 0, :, :]
    label = label[0, 0, :, :]
    pred = torch.sigmoid(pred)
    pred = pred > threshold
    label = label == torch.max(label)

    TP_mask = (pred == 1) * (label == 1)
    FP_mask = (pred == 1) * (label == 0)
    FN_mask = (pred == 0) * (label == 1)

    img.float()
    img_r, img_g, img_b = img.clone(), img.clone(), img.clone()

    # TP
    img_r[TP_mask] = 1.0
    img_g[TP_mask] = 70 / 255
    img_b[TP_mask] = 70 / 255
    # FP
    img_r[FP_mask] = 0.7
    img_g[FP_mask] = 1.0
    img_b[FP_mask] = 0.4588
    # FN
    img_r[FN_mask] = 1.0
    img_g[FN_mask] = 0.96
    img_b[FN_mask] = 0.3922

    img = torch.stack([img_r, img_g, img_b], 0)
    img = img[None, :, :, :].float()

    pred = pred.float()
    pred_r, pred_g, pred_b = pred.clone(), pred.clone(), pred.clone()
    
    # FP is red color so set 0 to g and b color channel
    pred_r[FP_mask] = 0 # 0.67
    pred_g[FP_mask] = 1.0
    pred_b[FP_mask] = 0 # 0.67
    # FN is yellow color so set 0 to g and b color channel
    pred_r[FN_mask] = 1.0
    pred_g[FN_mask] = 0 # 0.67 
    pred_b[FN_mask] = 0 # 0.67 
    
    pred = torch.stack([pred_r, pred_g, pred_b], 0)
    pred = pred[None, :, :, :].float()

    return pred # img, pred

"""將圖像、標籤和預測結果組合成一個網格並保存為圖像文件"""
def make_grid(img, labels, pred, data, save_dir):
    img, labels, pred = img.cpu(), labels.cpu(), pred.cpu()
    pred = pred.permute(1, 2, 0)
    
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.suptitle('Recall: {:.3f}   Precision: {:.3f}   F1: {:.3f}'.format(float(data["SE"]),
        float(data["PR"]),
        float(data["F1"])),
        fontsize=20, x=0.5, y=0.1, horizontalalignment='center')
    ax[0].set_axis_off()
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Image", fontsize=18)

    ax[1].set_axis_off()
    ax[1].imshow(labels, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Label", fontsize=18)

    ax[2].set_axis_off()
    ax[2].imshow(pred, vmin=0, vmax=1)
    ax[2].set_title("Predicted", fontsize=18)
    
    file_name = data["file_name"][:-4] + "_compare" + data["file_name"][-4:]
    fig.savefig(os.path.join(save_dir, file_name), pad_inches = 0)
    plt.close()