import glob
import os.path
from tqdm import tqdm
import cv2
import numpy as np
import json

def main():
    filename = r"E:\PycharmProjects\autoGame\models\datasets\cf1920\ALLtest\{}.json"
    img_id = 20211030192214
    filename = filename.format(img_id)
    img_H, img_W = 1080, 1920
    with open(filename, ) as f:
        js = json.load(f)
    shape = js["shapes"][0]
    lab = shape["label"]
    points = shape["points"]
    class_id = 0
    p1 = points[0]
    p2 = points[1]
    x_1, y_1 = p1[0], p1[1]
    x_2, y_2 = p2[0], p2[1]
    x_center = (x_1 + abs(x_1-x_2)/2.0) / img_W
    y_center = (y_1 + abs(y_1-y_2)/2.0) / img_H
    width = abs(x_1 - x_2) / img_W
    height = abs(y_1 - y_2) / img_H

    with open('./%s.txt' % img_id, 'w') as f:
        f.write("%d %f %f %f %f" % (class_id, x_center, y_center, width, height))



if __name__ == '__main__':
    main()


