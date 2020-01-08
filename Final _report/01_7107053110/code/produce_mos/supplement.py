import os
import glob
import numpy as np
from skimage import color
from SLIC_alog import *


def get_path(input_dir):
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")
    input_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(input_dir, "*.png"))
    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")
    return input_paths


def trans_img(image):
    # L_image Range: [0, 1] --> [0, 255]
    img = image * 255
    # L_image type: float32 --> uint8
    img_out = img.astype(np.uint8)
    return img_out

def cat_trans_img(gray, ab):
    # L_image Range: [0, 1] --> [0, 100]
    L_channel = gray * 100
    L_channel = np.expand_dims(L_channel, axis=2)
    # AB_image Range: [-1, 1] --> [-128, 128]
    AB_channel = ab * 128
    lab = np.concatenate([L_channel, AB_channel], axis=2)
    # LAB image Range: [0, 100], [-128, 128], [-128, 128] --> RGB image Range: [0, 1]
    rgb = color.lab2rgb(lab)
    # RGB image Range: [0, 1] --> [0, 255]
    rgb_rejust = rgb * 255
    # RGB image type: float64 --> uint8
    rgb_out = rgb_rejust.astype(np.uint8)
    return rgb_out


def check_output(output_path):
    if output_path is None or not os.path.exists(output_path):
        os.makedirs(output_path)


