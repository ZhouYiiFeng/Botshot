# from models.yolo import Model
import numpy as np
from utils.general import (LOGGER, NCOLS, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
import torch

from models.common import DetectMultiBackend
import cv2

if __name__ == '__main__':
    conf_thres = 0.001  # confidence threshold
    iou_thres = 0.6
    single_cls = False
    yolo = DetectMultiBackend(r'E:\PycharmProjects\autoGame\models\yolov5-master\runs\train\cfauto\weights\last.pt',
                              device='cuda')
    yolo.eval()
    img = cv2.imread(r"E:\PycharmProjects\autoGame\models\datasets\cf256\images\20211030192214_0.png")
    img = img.transpose((2, 0, 1))[::-1].astype(np.float32) / 255
    img_tsr = torch.from_numpy(img.copy()).unsqueeze(0).cuda()
    out = yolo(img_tsr)
    out = non_max_suppression(out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=single_cls)
    for si, pred in enumerate(out):
        # labels = targets[targets[:, 0] == si, 1:]
        # nl = len(labels)
        # tcls = labels[:, 0].tolist() if nl else []  # target class
        # path, shape = Path(paths[si]), shapes[si][0]
        # seen += 1

        # if len(pred) == 0:
        #     if nl:
        #         stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
        #     continue

        # Predictions
        if single_cls:
            pred[:, 5] = 0
        predn = pred.clone()
        p1 = pred[:, 4].cpu()
        p2 = pred[:, 5].cpu()
        print("a")
        # scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
