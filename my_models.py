# Your model classes (A3, A4, A5) and helpers that later notebooks need (e.g. A2 feature functions), and one
# preprocessing function per model (normalization constants included), so P1_V, P1_B and P1_665 prepare inputs
# exactly as P1_A did. Save the training statistics they need (e.g. channel means and stds) to CKPT_DIR.
# This cell writes the file; the next cell imports it. Re-run both after every change.
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import cv2

def hsv_hist(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    out = []
    for ch, rng in enumerate([[0, 180], [0, 256], [0, 256]]):
        h = cv2.calcHist([hsv], [ch], None, [16], rng).ravel()
        out.append(h / h.sum())
    return np.concatenate(out).astype(np.float32)

_HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
def hog_feat(im):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return _HOG.compute(gray).ravel().astype(np.float32)

def _gabor_bank():
    bank = []
    for lambd, ks in [(4, 13), (8, 27)]:
        for k in range(4):
            kern = cv2.getGaborKernel((ks, ks), 0.56 * lambd, k * np.pi / 4, lambd, 0.5, 0, ktype=cv2.CV_32F)
            bank.append(kern - kern.mean())
    return bank
_GABOR = _gabor_bank()

def gabor_feat(im):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    out = []
    for kern in _GABOR:
        r = np.abs(cv2.filter2D(gray, cv2.CV_32F, kern))
        out += [r.mean(), r.var()]
    return np.array(out, dtype=np.float32)

FEAT_FNS = {'hsv': hsv_hist, 'hog': hog_feat, 'gabor': gabor_feat}

def standardize_blocks(blocks, stats):
    return {k: (blocks[k] - stats[k][0]) / stats[k][1] for k in blocks}

def concat_features(blocks_std):
    return np.concatenate([blocks_std[k] for k in ('hsv', 'hog', 'gabor')], axis=1)
