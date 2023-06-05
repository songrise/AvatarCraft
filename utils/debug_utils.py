# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/utils/debug_utils.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


def embed_breakpoint(debug_info='', terminate=True):
    print('\nyou are inside a break point')
    if debug_info:
        print(f'debug info: {debug_info}')
    print('')
    embedding = ('import numpy as np\n'
                 'import IPython\n'
                 'import matplotlib.pyplot as plt\n'
                 'IPython.embed()\n'
                 )
    if terminate:
        embedding += (
            'exit()'
        )

    return embedding

# -*- coding : utf-8 -*-
# @FileName  : debug_utils.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Sep 28, 2022
# @Github    : https://github.com/songrise
# @Description: Debug utils for stylization
import os
import torch
import numpy as np
import PIL.Image as Image
import cv2
import pickle
def dump_tensor(a, name, path=None):
    if path is None:
        path = "./"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, name), 'wb') as f:
        pickle.dump(a, f)
    print(f"dump tensor {name} to {path}")

def visualize_tensor(a, name, path):
    if not os.path.exists(path):
        os.makedirs(path)

    a = a.cpu().detach().numpy()
    a = np.transpose(a, (1, 2, 0))
    a = (a + 1) * 127.5
    a = a.astype(np.uint8)
    cv2.imwrite(os.path.join(path, name), a)

def log_exec(func):
    def wrapper(*args, **kwargs):
        print(f"[INFO] executing {func.__name__}")
        ret =  func(*args, **kwargs)
        print(f"[INFO] executed {func.__name__}")
        return ret
    return wrapper