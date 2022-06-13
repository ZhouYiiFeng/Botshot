import time
import threading
import pygame
import sys
import cv2
import pyautogui
from ctypes import windll
import pydirectinput
import ctypes
import win32gui
import win32api
import win32con
from PIL import ImageGrab
import numpy as np
import torch
import pynput
from pynput import mouse, keyboard
import sys
sys.path.append('../models/yolov5-master')
import pynput
from models.yolo import Model
from train import parse_opt
from models.common import DetectMultiBackend
import yaml
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync


SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def on_mouse_move(x, y):
    print("\rX:%d Y:%d" % (x, y), end=" ", flush=True)


def mouse_click(x, y, button, pressed):
    global waiting_click
    if pressed and button == mouse.Button.left:
        waiting_click = False

def start_mouse_listen():  # 用于开始鼠标的监听
    # 进行监听
    with pynput.mouse.Listener(on_move=on_mouse_move, on_click=mouse_click) as listener:
        listener.join()


def set_pos(x, y):

    x = 1 + int(x * 65536. / 1920)
    y = 1 + int(y * 65536. / 1080)
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0,
                                           ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    command = pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

def MoveMouse(x, y):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    # x = int(x*(65536/ctypes.windll.user32.GetSystemMetrics(0))+1)
    # y = int(y*(65536/ctypes.windll.user32.GetSystemMetrics(1))+1)
    ii_.mi = MouseInput(x, y, 0, 0x0001, 0, ctypes.pointer(extra))
    cmd = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(cmd), ctypes.sizeof(cmd))

def mouse_in_bbox(cur_mouse, bbox):
    cur_x, cur_y = cur_mouse
    abs_bbox_center_x, abs_bbox_center_y, h, w = bbox
    if cur_x > abs_bbox_center_x-w/2-5 and cur_x < abs_bbox_center_x+w/2+5 and cur_y > abs_bbox_center_y-h/2-5 and cur_y < abs_bbox_center_y+h/2+5:
        return True
    return False

def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor

def test1(opt):
    pygame.init()
    screen_width = 500
    screen_height = 500
    sm_width = 256
    sm_height = 256

    buttons = {
        "Button.left": pynput.mouse.Button.left,
        "Button.right": pynput.mouse.Button.right
    }
    # model = torch.hub.load('ultralytics/yolov5', )

    weights = r"E:\PycharmProjects\autoGame\models\yolov5-master\runs\train\cfauto\weights\best.pt"
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    device = select_device('0')
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(256, s=stride)  # check image size

    # ckpt = torch.load(weights, map_location='cuda')
    # with open(str(opt.hyp), errors='ignore') as f:
    #     hyp = yaml.safe_load(f)  # load hyps dict
    # model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=1, anchors=hyp.get('anchors')).to('cuda')  # create
    # # exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
    # csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    # # csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    # model.load_state_dict(csd, strict=False)  # load

    _, _, Desk_width, Desk_height = win32gui.GetClientRect(win32gui.GetDesktopWindow())
    fuchsia = (255, 0, 128)
    dark_red = (139, 0, 0)
    dark_y = (139, 128, 0)

    # fuchia2 = (255, 0, 121)
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.NOFRAME)  # 创建一个名为screen的窗口
    pygame.display.set_caption("Transpose wind")  # 设置当前窗口标题
    # Create layered window
    hwnd = pygame.display.get_wm_info()["window"]

    # Make pygame window always on the top
    SetWindowPos = windll.user32.SetWindowPos
    SetWindowPos(pygame.display.get_wm_info()['window'], -1, (Desk_width-screen_width)//2, (Desk_height-screen_height)//2, 0, 0, 0x0001)

    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)
    # Set window transparency color 建纯色背景色, win32 使得该纯色为透明色
    win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*fuchsia), 0, win32con.LWA_COLORKEY)

    mouse = pynput.mouse.Controller()
    global waiting_click
    waiting_click = False
    screen.fill(fuchsia)
    # set_pos(Desk_width // 2, Desk_height // 2)
    # 开始游戏的主循环
    flag = True
    shotCounter = 0
    while True:

        for envent in pygame.event.get():  # 监听用户事件
            if envent.type == pygame.QUIT:  # 判断用户是否点击了关闭按钮
                sys.exit()  # 用户退出



        full_img = ImageGrab.grab()
        focus_area_left = int(Desk_width / 2 - sm_width / 2)
        focus_area_top = int(Desk_height / 2 - sm_height / 2)
        focus_area = np.array(full_img)[focus_area_top:focus_area_top + sm_height,
                     focus_area_left:focus_area_left + sm_width]

        focus_area_screen_left = screen_width / 2 - sm_width / 2
        focus_area_screen_top = screen_height / 2 - sm_height / 2


        # cv2.imshow("1", focus_area[:,:,::-1])

        focus_area = np2Tensor(focus_area/255.0).unsqueeze(0).cuda()
        pred = model(focus_area, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=10)

        for i, det in enumerate(pred):
            if len(det):
                pred = det.cpu().numpy()
                # for bbox in pred:
                xmin, ymin, xmax, ymax, conf, class_name = pred[0]
                if conf > 0.6:
                    bbox_w = xmax - xmin
                    bbox_h = ymax - ymin
                    center_x = xmin + bbox_w // 2
                    center_y = ymin + bbox_h // 2
                    screen.fill(fuchsia)
                    pygame.draw.rect(screen, dark_y, pygame.Rect(focus_area_screen_left+xmin, focus_area_screen_top+ymin,
                                                                 bbox_w, bbox_h), 1)
                    # target_x = []
                    # set_pos(focus_area_left + center_x, focus_area_top + center_y)
                    # MoveMouse(focus_area_left + center_x, focus_area_top + center_y)
                    # print(center_x)
                    # print(center_y)

                    cur_x, cur_y = pyautogui.position()
                    center_x_abs = center_x + focus_area_left
                    center_y_abs = center_y + focus_area_top
                    MoveMouse(int(center_x_abs - cur_x), int(center_y_abs - cur_y))
                    if mouse_in_bbox((cur_x, cur_y),
                                         (center_x + focus_area_left, center_y + focus_area_top, bbox_h, bbox_w)):
                        # if shotCounter >= 0:
                        #     pyautogui.click(clicks=2, interval=0.5)
                        #     shotCounter -= 1
                        # else:
                        #     shotCounter = 5
                        # shotCounter = 10
                        mouse.release(pynput.mouse.Button.left)
                        mouse.click(pynput.mouse.Button.left, 1)
                        mouse.release(pynput.mouse.Button.left)


                    print("\r %d" % shotCounter, end=" ", flush=True)


                        # flag = False
                        # waiting_click = True
                        # set_pos(center_x, center_y)
                        # time.sleep(5)

            pygame.draw.rect(screen, dark_red, pygame.Rect(focus_area_screen_left, focus_area_screen_top, sm_width,
                                                           sm_height), 1)

            pygame.display.update()
            # shotCounter = 10
            # if shotCounter > 0:
            #     shotCounter -= 1

            # time.sleep(1)

if __name__ == '__main__':
    # t1 = threading.Thread(target=start_mouse_listen)  # 创建用于监听鼠标的线程
    # t1.start()
    # MoveMouse(960, 540)
    # time.sleep(2)
    # MoveMouse(960, 500)
    # time.sleep(2)
    # MoveMouse(920, 500)
    set_pos(1920 // 2, 1080 // 2)
    opt = parse_opt()
    test1(opt)
    #
    # import torch
    #
    # # Model
    # model = torch.hub.load('ultralytics/yolov3', 'yolov3')  # or 'yolov3_spp', 'yolov3_tiny'
    #
    # # Image
    # img = 'https://ultralytics.com/images/zidane.jpg'
    #
    # # Inference
    # results = model(img)
    # results.print()  # or .show(), .save()