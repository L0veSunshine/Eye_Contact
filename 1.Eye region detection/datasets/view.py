import cv2
import json
import numpy as np


def preview_dataset(json_file):
    with open(json_file) as f:
        labels = json.load(f)
    for label in labels:
        img = cv2.imread(label['image_path'])
        landmarks = np.array(label['keypoints']).reshape((68, 2)).astype(np.integer)
        for point in landmarks:
            img = cv2.circle(img, tuple(point), 2, (255, 0, 0), -1, 1)
        cv2.imshow("", img)
        cv2.waitKey()


if __name__ == '__main__':
    preview_dataset('../val.json')
