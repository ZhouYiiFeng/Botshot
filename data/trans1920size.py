"""
Generate the crop size training patches.

the patches is followed by these standers:
    1. pos ratio consistency
    2. integration
"""

import glob
import os.path
from tqdm import tqdm
import cv2
import numpy as np


def val_bbox_in_img(img, labels):
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
    cv2.imshow("a", img)
    # cv2.imwrite("test.png", img)
    cv2.waitKey(500)


def val_bbox_in_path(img_p, label_p):
    img = cv2.imread(img_p)
    labels = []
    with open(label_p, "r") as f:
        while True:
            line = f.readline()
            if line:
                labels.append(line.strip().split(" "))
            else:
                break
    val_bbox_in_img(img, labels)


def include_bbox(win, label, crop_wind_size, orig_h=1080, orig_w=1920):
    """
    :param win:
    :param label:
    :param crop_wind_size:
    :param orig_h:
    :param orig_w:
    :return: judge whether the label is contained in the given win.
    """

    win_left_abs_pixel_pos_x, win_top_abs_pixel_pos_y = win
    win_right_abs_pixel_pos_x, win_btn_abs_pixel_pos_y = win_left_abs_pixel_pos_x+crop_wind_size, \
                                                         win_top_abs_pixel_pos_y + crop_wind_size
    class_id, x_center, y_center, width, height = label
    x_center = float(x_center)
    y_center = float(y_center)
    width = float(width)
    height = float(height)
    anchor_abs_pixel_pos_x = x_center * orig_w
    anchor_abs_pixel_pos_y = y_center * orig_h
    width, height = orig_w * width, orig_h * height
    bbox_top = anchor_abs_pixel_pos_y - height // 2
    bbox_btn = anchor_abs_pixel_pos_y + height // 2
    bbox_left = anchor_abs_pixel_pos_x - width // 2
    bbox_right = anchor_abs_pixel_pos_x + width // 2
    if bbox_right > win_right_abs_pixel_pos_x or bbox_left < win_left_abs_pixel_pos_x or bbox_top < \
            win_top_abs_pixel_pos_y or bbox_btn > win_btn_abs_pixel_pos_y:
        return False
    return True


def get_abs_win(label, orig_h=1080, orig_w=1920, crop_wind_size=256):
    """
    :param label:
    :param orig_h:
    :param orig_w:
    :param crop_wind_size:
    :return: the top and left pos of cropped window.
    maintain the same pos of original img with that of cropped patch. e.g. x;y in Img.   x;y in patch.
    """
    class_id, x_center, y_center, width, height = label
    x_center = float(x_center)
    y_center = float(y_center)
    win_x_offset = crop_wind_size * x_center
    win_y_offset = crop_wind_size * y_center
    anchor_abs_pixel_pos_x = x_center * orig_w
    anchor_abs_pixel_pos_y = y_center * orig_h
    win_left_abs_pixel_pos_x = int(anchor_abs_pixel_pos_x - win_x_offset)
    win_top_abs_pixel_pos_y = int(anchor_abs_pixel_pos_y - win_y_offset)
    return win_left_abs_pixel_pos_x, win_top_abs_pixel_pos_y


def classify_label(labels, crop_wind_size=256, orig_h=1080, orig_w=1920):
    c_label = []
    visited = []
    for i in range(len(labels)):
        cluster = []
        if i not in visited:
            cluster.append(labels[i])
            visited.append(i)
            win_left_abs_pixel_pos_x, win_top_abs_pixel_pos_y = get_abs_win(labels[i])
            win = (win_left_abs_pixel_pos_x, win_top_abs_pixel_pos_y)
            for j in range(i+1, len(labels)):
                """
                for other boxes.
                """
                if j not in visited:
                    """
                    if current window include other boxes?
                    """
                    if include_bbox(win, labels[j], crop_wind_size, orig_h, orig_w):
                        visited.append(j)
                        class_id, x_center, y_center, width, height = labels[j]
                        x_center = float(x_center)
                        y_center = float(y_center)
                        anchor_abs_pixel_pos_x = x_center * orig_w
                        anchor_abs_pixel_pos_y = y_center * orig_h
                        x_center_win = abs(anchor_abs_pixel_pos_x - win_left_abs_pixel_pos_x) / crop_wind_size
                        y_center_win = abs(anchor_abs_pixel_pos_y - win_top_abs_pixel_pos_y) / crop_wind_size
                        cluster.append([class_id, x_center_win, y_center_win, width, height])
        if len(cluster) >0:
            c_label.append(cluster)
    return c_label


def trans1(crop_wind_size=256):
    list_dir = "../models/datasets/cf1920"

    out_put_path = "../models/datasets/cf256/"
    output_img_path = os.path.join(out_put_path, "images")
    output_label_path = os.path.join(out_put_path, "labels")
    if not os.path.exists(out_put_path):
        os.mkdir(out_put_path)
        os.mkdir(output_img_path)
        os.mkdir(output_label_path)

    imgs_path = sorted(glob.glob(os.path.join(list_dir, "ALLtest", "Images", "*.jpg")))
    label_path = sorted(glob.glob(os.path.join(list_dir, "ALLtest", "labels", "*.txt")))

    """
    label : class id; x->, y|, Width, Height. 
    """

    # val_bbox_in_path(imgs_path[0], label_path[0])

    for id in tqdm(range(len(label_path))):
        # img_p = imgs_path[id]
        label_p = label_path[id]
        name = label_p.split('\\')[-1].split('.')[0]
        img_p = os.path.join(list_dir, "ALLtest", "Images", name+".jpg")
        img = cv2.imread(img_p)
        labels = []
        with open(label_p, "r") as f:
            while True:
                line = f.readline()
                if line:
                    labels.append(line.strip().split(" "))
                else:
                    break
        h, w, c = img.shape
        labels_win = []
        c_label = classify_label(labels, crop_wind_size, h, w)
        cluster_id = 0
        for cluster in c_label:

            win_left_abs_pixel_pos_x, win_top_abs_pixel_pos_y = get_abs_win(cluster[0], h, w)
            win = img[win_top_abs_pixel_pos_y:win_top_abs_pixel_pos_y + crop_wind_size,
                  win_left_abs_pixel_pos_x:win_left_abs_pixel_pos_x + crop_wind_size, :]
            for cluster_itm in cluster:
                class_id, x_center, y_center, width, height = cluster_itm
                width = float(width)
                height = float(height)
                # the width and height is ratio for orignial image size, then transform to the patch size.
                labels_win.append([int(class_id), float(x_center), float(y_center), width*w/crop_wind_size, height*h/crop_wind_size])

            # val_bbox_in_img(win, labels_win)
            image_name = img_p.split('\\')[-1].split('.')[0]+"_%d" % cluster_id+".png"
            label_name = label_p.split('\\')[-1].split('.')[0]+"_%d" % cluster_id+".txt"
            cv2.imwrite(os.path.join(output_img_path, image_name), win)
            with open(os.path.join(output_label_path, label_name), "w") as f:
                for line in labels_win:
                    strings = "%d %f %f %f %f\n" % (line[0], line[1], line[2], line[3], line[4])
                    f.write(strings)
            labels_win = []
            cluster_id += 1

        # cv2.imshow("a", win)
        # cv2.waitKey()


if __name__ == '__main__':
    crop_wind_size = 256
    trans1(crop_wind_size)