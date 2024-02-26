import random
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
from PIL import Image, ImageFilter
"""
這段程式碼是用Python實現的圖像增強函數。這些函數可用於對圖像進行各種操作，
如對比度增強、銳化、模糊等。其中的RandAugment類是一個隨機圖像增強類，它從
一個預定義的操作列表中隨機選擇操作，並將其應用於圖像。這些操作還具有可調整的
參數，這些參數在應用操作之前根據指定的範圍進行隨機選擇。
"""
PARAMETER_MAX = 10

#def _float_parameter(v, max_v):
#    return float(v) * max_v / PARAMETER_MAX


#def _int_parameter(v, max_v):
#    return int(v * max_v / PARAMETER_MAX)

def hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Sharpness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Identity(img, _):
    return img


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Posterize(img, v):
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Solarize(img, v):
    return PIL.ImageOps.solarize(img, v)


def Edge_enhance(img, _):
    return img.filter(ImageFilter.EDGE_ENHANCE)


def Edge_enhance_more(img, _):
    return img.filter(ImageFilter.EDGE_ENHANCE_MORE)


def Smooth(img, _):
    return img.filter(ImageFilter.SMOOTH)


def Smooth_more(img, _):
    return img.filter(ImageFilter.SMOOTH_MORE)


def Detail(img, _):
    return img.filter(ImageFilter.DETAIL)


def Blur(img, _):
    return img.filter(ImageFilter.BLUR)


def Sharpen(img, _):
    return img.filter(ImageFilter.SHARPEN)


def Emboss(img, _):
    return img.filter(ImageFilter.EMBOSS)


def Contour(img, _):
    return img.filter(ImageFilter.CONTOUR)


def Find_edges(img, _):
    return img.filter(ImageFilter.FIND_EDGES)


def SolarizeAdd(img, v, threshold=128):
    v = int(v)
    if random.random() > 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def gauss_noise(image, v):
    # row, col, ch = image.shape
    row, col, ch = 512, 512, 1
    mean, var = 0, v
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy

def Original_img(img, _):
    return img

def augment_list():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 1.2),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            #(Emboss, 0, 1),
            #(Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            ]
    return augs

def augment_list1():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.1, 2.0), # (Sharpness, 0.5, 5.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            ]
    return augs

def augment_list_16():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            # (Posterize, 4, 8),
            (Sharpness, 0.5, 5.0), # (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            ]
    return augs

def augment_list_17():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            # (Posterize, 4, 8),
            (Sharpness, 0.5, 5.0), # (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            (gauss_noise, 0, 0.01),
            ]
    return augs

def augment_list_18():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.5, 5.0), # (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            (gauss_noise, 0, 0.01),
            ]
    return augs


def augment_list_best_9Aug():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            # (Posterize, 4, 8),
            # (Sharpness, 0.5, 5.0), # (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            # (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            # (Edge_enhance, 0, 1),
            # (Edge_enhance_more, 0, 1),
            # (Smooth, 0, 1),
            # (Smooth_more, 0, 1),
            # (Sharpen, 0, 1),
            #(Identity, 0, 1),
            ]
    return augs

def augment_list_best_1Aug():
    # Test
    augs = [(Contour, 0, 1),]
    return augs

def augment_list_best_2Aug():
    # Test
    augs = [(Blur, 0, 1),
            (Contour, 0, 1),
            ]
    return augs

def augment_list_best_2Aug_with_ori_img():
    # Test
    augs = [(Blur, 0, 1),
            (Contour, 0, 1),
            (Original_img, 0, 1)
            ]
    return augs

def augment_list_best_3Aug():
    # Test
    augs = [(Blur, 0, 1),
            (Contour, 0, 1),
            (SolarizeAdd, 0, 80),
            ]
    return augs

def augment_list_best_4Aug():
    # Test
    augs = [(Blur, 0, 1),
            (Contour, 0, 1),
            (SolarizeAdd, 0, 80),
            (Invert, 0, 1),
            ]
    return augs

def augment_list_3Aug():
    # Test
    augs = [(Blur, 0, 1),
            (Contour, 0, 1),
            (Posterize, 4, 8),
            ]
    return augs

def augment_list_gauss_noise():
    # Test
    augs = [(gauss_noise, 0, 0.01),
            ]
    return augs

def augment_list_gauss_noise_Contour():
    # Test
    augs = [(gauss_noise, 0, 0.01),
            (Contour, 0, 1),
            ]
    return augs

def augment_list_gauss_noise_Contour_Blur():
    # Test
    augs = [(gauss_noise, 0, 0.01),
            (Contour, 0, 1),
            (Blur, 0, 1),
            ]
    return augs

def augment_list_Detail():
    # Test
    augs = [(Detail, 0, 1),
            ]
    return augs

def augment_list_M():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 1.2),
            ]
    return augs


class RandAugment:
    def __init__(self, n):
        self.n = n
        #self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            if random.random() > 0.5:
                val = round(min_val + float(max_val - min_val) * random.random(), 2)
                img = op(img, val)

        return img

class RandAugment1:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list1()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img
    
class RandAugment_16:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_16()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_17:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_17()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

    
class RandAugment_18:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_18()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img


class RandAugment_single:
    def __init__(self, n, m,aug_num):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list1()
        self.aug_num = aug_num

    def __call__(self, img):
        # ops = random.choices(self.augment_list, k=self.n)
        ops = [self.augment_list[self.aug_num]]
        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_1aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_best_1Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_best_2aug_with_ori_img:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_best_2Aug_with_ori_img()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_best_3aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_best_3Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_best_4aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_best_4Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_3aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_3Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_gauss_noise:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_gauss_noise()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_gauss_noise_Contour:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_gauss_noise_Contour()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_gauss_noise_Contour_Blur:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_gauss_noise_Contour_Blur()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_Detail:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_Detail()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img


class RandAugment_best9aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_best_9Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img


class Without_RandAugment:
    def __call__(self, img):
        return img

class RandAugmentM:
    def __init__(self, n):
        self.n = n
        #self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_M()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:            
            factor1 = np.random.randint(0, (max_val - min_val)/0.1)
            factor2 = np.random.randint(0, (max_val - min_val)/0.1)
            
            val1 = min_val + 0.1 * factor1
            val2 = min_val + 0.1 * factor2
            img1 = op(img, val1)
            img2 = op(img, val2)

        return img1, img2