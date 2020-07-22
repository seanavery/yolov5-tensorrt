import cv2
import numpy as np
from classes import coco
import math

class Visualizer():
    def __init__(self):
        print('inside visualizer')
        self.num_classes = len(coco)

    def draw(self, img, pred):
        print('img', img.shape)
        print('pred', pred[0].shape)
        for p in pred[0]:
            p = p.astype(int)
            x1, y1, x2, y2, cf, cl = p[0], p[1], p[2], p[3], p[4], p[5]
            print('x1', x1)
            print('y1', y1)
            print('x2', x2)
            print('y2', y2)
            print('cf', cf)
            print('cl', cl, round(cl).item(), coco[cl])
            cl = coco[cl]
            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 205, 50), -1)
        return img
