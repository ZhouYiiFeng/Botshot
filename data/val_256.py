# from trans1920size import val_bbox_in_path
import glob
import os.path
from tqdm import tqdm
import cv2
import numpy as np

def val_bbox_in_img(img, labels, id):
    h, w, c = img.shape

    for label in labels:
        class_id, x_center, y_center, width, height = label
        class_id = float(class_id)
        x_center = float(x_center)
        y_center = float(y_center)
        width = float(width) * w
        height = float(height) * h
        anchor_abs_pixel_pos_x = x_center * w
        anchor_abs_pixel_pos_y = y_center * h

        cv2.rectangle(img, (int(anchor_abs_pixel_pos_x-width//2), int(anchor_abs_pixel_pos_y-height//2)),
                      (int(anchor_abs_pixel_pos_x+width//2), int(anchor_abs_pixel_pos_y+height//2)), color=(0,0,255), thickness=1)
    # cv2.imshow("a", img)
    cv2.imwrite("%d_b.png" % id, img)
    # cv2.waitKey(500)

def val_bbox_in_path(img_p, label_p, id):
    img = cv2.imread(img_p)
    labels = []
    with open(label_p, "r") as f:
        while True:
            line = f.readline()
            if line:
                labels.append(line.strip().split(" "))
            else:
                break
    val_bbox_in_img(img, labels, id)

if __name__ == '__main__':
    # list_dir = "../models/datasets/cf256"
    list_dir = "../models/datasets/cf1920"
    imgs_path = sorted(glob.glob(os.path.join(list_dir, "ALLtest", "Images", "*.jpg")))[0:100]
    label_path = sorted(glob.glob(os.path.join(list_dir, "ALLtest", "labels", "*.txt")))[0:100]
    # imgs_path = sorted(glob.glob(os.path.join(list_dir, "Images", "*.png")))
    # label_path = sorted(glob.glob(os.path.join(list_dir, "labels", "*.txt")))
    for i in range(len(imgs_path)):
        val_bbox_in_path(imgs_path[i], label_path[i], i)