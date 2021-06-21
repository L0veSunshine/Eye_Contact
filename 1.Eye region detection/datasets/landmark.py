import numpy as np
import copy
import json
import random
import cv2

from torch.utils.data import Dataset
from utils.visual_augmentation import ColorDistort, pixel_jitter
from utils.augmentation import Rotate_aug, Affine_aug, mirror, Padding_aug, Img_dropout
from utils.turbo.turbo import reader

symmetry = [(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), (8, 8),
            (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
            (31, 35), (32, 34),
            (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
            (48, 54), (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63), (65, 67)]
base_extend_range = [0.2, 0.3]


class JsonData:
    def __init__(self, json_file: str):
        if not json_file.endswith(".json"):
            json_file += '.json'
        with open(json_file, 'r') as f:
            train_json_list = json.load(f)
        self.metas = train_json_list

    def get_all_sample(self):
        random.shuffle(self.metas)
        return self.metas


class Landmark(Dataset):
    def __init__(self, ann_file, input_size=(160, 160), training_flag=True):
        super(Landmark, self).__init__()
        self.counter = 0
        self.time_counter = 0
        self.raw_dataset_size = 0
        self.training_flag = training_flag
        self.color_augmentor = ColorDistort()
        self.lst = self.parse_file(ann_file)
        self.input_size = input_size

    def __getitem__(self, item):
        """Data augmentation function."""
        dp = self.lst[item]
        fname: str = dp['image_path']
        keypoints = dp['keypoints']
        bbox = dp['bbox']
        if fname.endswith('.jpg'):
            image = reader.imread(fname)
        else:
            image = cv2.imread(fname)
        label = np.array(keypoints, dtype=np.float32).reshape((-1, 2))  # (68,2)
        bbox = np.array(bbox)
        crop_image, label = self.augmentationCropImage(image, bbox, label, self.training_flag)
        # for point in label:
        #     crop_image = cv2.circle(crop_image, tuple(point), 2, (255, 255, 0), -1)
        # cv2.imshow("", crop_image)
        # print(crop_image.shape)
        # cv2.waitKey(0)

        if self.training_flag:
            if random.uniform(0, 1) > 0.5:
                crop_image, label = mirror(crop_image, label=label, symmetry=symmetry)
            if random.uniform(0, 1) > 0.2:
                angle = random.uniform(-10, 10)
                crop_image, label = Rotate_aug(crop_image, label=label, angle=angle)
            if random.uniform(0, 1) > 0.5:
                strength = random.uniform(0, 50)
                crop_image, label = Affine_aug(crop_image, strength=strength, label=label)
            if random.uniform(0, 1) > 0.5:
                crop_image = self.color_augmentor(crop_image)
            if random.uniform(0, 1) > 0.5:
                crop_image = pixel_jitter(crop_image, 15)
            if random.uniform(0, 1) > 0.5:
                crop_image = Img_dropout(crop_image, 0.2)
            if random.uniform(0, 1) > 0.5:
                crop_image = Padding_aug(crop_image, 0.3)

        # reprojectdst, euler_angle = get_head_pose(label, crop_image)
        # PRY = euler_angle.reshape([-1]).astype(np.float32) / 90.

        crop_image_height, crop_image_width, _ = crop_image.shape

        label = label.astype(np.float32)

        # for point in label:
        #     crop_image = cv2.circle(crop_image, tuple(point), 2, (255, 0, 0), -1)
        # cv2.imshow("", crop_image)
        # cv2.waitKey(0)

        label[:, 0] = label[:, 0] / crop_image_width
        label[:, 1] = label[:, 1] / crop_image_height
        label = label.reshape([-1]).astype(np.float32)

        crop_image = crop_image.astype(np.float32)
        crop_image = (crop_image - 127.0) / 127.0
        crop_image = np.transpose(crop_image, (2, 0, 1)).astype(np.float32)

        return crop_image, label

    def __len__(self):
        return len(self.lst)

    def parse_file(self, ann_file):
        ann_info = JsonData(ann_file)
        all_samples = ann_info.get_all_sample()
        self.raw_dataset_size = len(all_samples)
        print("原始样本数量: " + str(self.raw_dataset_size))
        return all_samples

    def augmentationCropImage(self, img, bbox, joints, is_training: bool):
        bbox = np.array(bbox).reshape(4, ).astype(np.float32)  # [top,left,right,bottom]
        add = max(img.shape[0], img.shape[1])
        bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=[127., 127., 127.])
        objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])
        bbox += add
        objcenter += add
        joints[:, :2] += add
        gt_width = (bbox[2] - bbox[0])
        gt_height = (bbox[3] - bbox[1])
        crop_width_half = gt_width * (1 + base_extend_range[0] * 2) // 2
        crop_height_half = gt_height * (1 + base_extend_range[1] * 2) // 2
        if is_training:
            min_x = int(objcenter[0] - crop_width_half +
                        random.uniform(-base_extend_range[0], base_extend_range[0]) * gt_width)
            max_x = int(objcenter[0] + crop_width_half +
                        random.uniform(-base_extend_range[0], base_extend_range[0]) * gt_width)
            min_y = int(objcenter[1] - crop_height_half +
                        random.uniform(-base_extend_range[1], base_extend_range[1]) * gt_height)
            max_y = int(objcenter[1] + crop_height_half +
                        random.uniform(-base_extend_range[1], base_extend_range[1]) * gt_height)
        else:
            min_x = int(objcenter[0] - crop_width_half)
            max_x = int(objcenter[0] + crop_width_half)
            min_y = int(objcenter[1] - crop_height_half)
            max_y = int(objcenter[1] + crop_height_half)
        joints[:, 0] = joints[:, 0] - min_x
        joints[:, 1] = joints[:, 1] - min_y
        img = bimg[min_y:max_y, min_x:max_x, :]
        crop_image_height, crop_image_width, _ = img.shape
        joints[:, 0] = joints[:, 0] / crop_image_width
        joints[:, 1] = joints[:, 1] / crop_image_height
        # 使用随机的插值法进行缩放
        interp_methods = [cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = random.choice(interp_methods)
        img = cv2.resize(img, self.input_size, interpolation=interp_method)
        joints[:, 0] = joints[:, 0] * self.input_size[0]
        joints[:, 1] = joints[:, 1] * self.input_size[1]
        return img, joints
