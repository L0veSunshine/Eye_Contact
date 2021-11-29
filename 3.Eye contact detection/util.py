import math
import random
import cv2
import numpy as np


def random_crop(src, shrink):
    h, w, _ = src.shape
    h_shrink = int(h * shrink)
    w_shrink = int(w * shrink)
    bimg = cv2.copyMakeBorder(src, h_shrink, h_shrink, w_shrink, w_shrink, borderType=cv2.BORDER_CONSTANT,
                              value=(0, 0, 0))
    start_h = random.randint(0, 2 * h_shrink)
    start_w = random.randint(0, 2 * w_shrink)
    target_img = bimg[start_h:start_h + h, start_w:start_w + w, :]
    return target_img.reshape(h, w, 3)


def random_rotate(src, angle, center=None, scale=1.0):
    image = src
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    for i in range(image.shape[2]):
        image[:, :, i] = cv2.warpAffine(image[:, :, i], M, (w, h), flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=[0., 0., 0.])
    return image


def pixel_jitter(src, max_=5.):
    src = src.astype(np.float32)
    pattern = (np.random.rand(src.shape[0], src.shape[1], src.shape[2]) - 0.5) * 2 * max_
    img = src + pattern
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)
    return img


def calculate_mcc(tp, fp, fn, tn):
    numerator = (tp * tn) - (fp * fn)  # 马修斯相关系数公式分子部分
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))  # 马修斯相关系数公式分母部分
    result = numerator / denominator
    return result
