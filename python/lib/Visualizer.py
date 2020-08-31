import cv2
import random
import colorsys
import numpy as np
import math
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from PIL import Image

from classes import coco

class Visualizer():
    def __init__(self):
        self.color_list = self.gen_colors(coco)
    
    def gen_colors(self, classes):
        """
            generate unique hues for each class and convert to bgr
            classes -- list -- class names (80 for coco dataset)
            -> list
        """
        hsvs = []
        for x in range(len(classes)):
            hsvs.append([float(x) / len(classes), 1., 0.7])
        random.seed(1234)
        random.shuffle(hsvs)
        rgbs = []
        for hsv in hsvs:
            h, s, v = hsv
            rgb = colorsys.hsv_to_rgb(h, s, v)
            rgbs.append(rgb)

        bgrs = []
        for rgb in rgbs:
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            bgrs.append(bgr)
        return bgrs
    
    def draw_object_grid(self, img, grids, conf_thres=0.08):
        """
            visualize object probabilty grid overlayed onto image

            img -- ndarray -- numpy array representing input image
            grids -- ndarray -- object probablity grid of shape (1, 3, x, y, 85)
            conf_thres -- float -- minimum objectness probability score
            -> None
            
        """
        for grid in grids:
            # set up image copies
            copy = img.copy()
            overlay = img.copy()
            
            # convert shit
            _, _, height, width, _ = grid.shape
            px_step = 640 // height
            px_stride = 640 // px_step

            # extract hot grid cells
            # take in (1, px_stride, py_stride, 1)
            # convert to (1, px_stride, py_stride, 3)
            # threshold by conf
            scales_x = np.arange(width) * px_step
            scales_y = np.arange(height) * px_step
            yv, xv = np.meshgrid(scales_x, scales_y)
            xy_grid = np.stack((yv, xv), axis=2)
            grid = grid.squeeze(axis=0)
            
            # take maximum of three anchor sizes
            grid = grid.max(axis=0)
            object_grid = np.concatenate((xy_grid, grid), axis=-1)
            
            # take threshold
            xc = object_grid[..., 2] > 0.1
            filtered = object_grid[xc]
            
            # draw rectangles
            for obj in filtered:
                x1y1 = (int(obj[0]), int(obj[1]))
                x2y2 =(int(obj[0]) + px_step, int(obj[1]) + px_step)
                cv2.rectangle(overlay, x1y1, x2y2, (0, 255, 0), -1)

            # draw lines
            line_color = [0, 255, 0]
            # draw x axis lines
            overlay[:, ::px_step, :] = line_color
            # draw y axis lines
            overlay[::px_step, :, :] = line_color

            cv2.addWeighted(overlay, 0.5, copy, 1 - 0.5, 0, copy)

            # make mat plt
            plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))
            plt.title('Grid Visualization \n stride: {}'.format(px_step))
            plt.xmin = 0
            plt.xmax = 640
            plt.ymin = 0
            plt.ymaxy = 640
            plt.show()
            continue
            maxes = grid.max(axis = 1)

            # cv2.imshow(window_name, copy)
            cv2.imwrite('box_grid_{}.jpg'.format(px_step), copy)
            # cv2.waitKey(10000)
    
    def draw_class_grid(self, img, grids, conf_thres=0.1):
        """
            visualize class probability grid 

            img -- ndarray -- input image
            grids -- ndarray -- class probabilities of shape (1, 3, x, y, 80)
            conf_thres -- float -- minimum threshold for class probability
        """
        for grid in grids:
            _, _, width, height, _ = grid.shape
            px_step = 640 // width
            window_name = 'classes {}'.format(height)
            cv2.namedWindow(window_name)
            copy = img.copy()
            for xi in range(width):
                for yi in range(height):
                    # calculate max class probability
                    # calculate max
                    mc = np.amax(grid[0][0][xi][yi][:])
                    mci = np.where(grid[0][0][xi][yi][:] == mc)
                    # mc = np.amax(grid[..., 0:])
                    
                    if mc > conf_thres:
                        cv2.rectangle(copy, (yi * px_step, xi * px_step), ((yi + 1) * px_step, (xi + 1) * px_step), self.color_list[int(mci[0])], -1)
                               
            cv2.imshow(window_name, copy)
            cv2.waitKey(1000) 
                       
        return None

    def draw_boxes(self, img, boxes):
        window_name = 'boxes'
        cv2.namedWindow(window_name)
        copy = img.copy()
        overlay = img.copy()
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.05, copy, 1 - 0.5, 0, copy)

        cv2.imshow(window_name, copy)
        cv2.waitKey(10000) 

    def draw_grid(self, img, output, i):
        x = 0
        while x < 640:
            cv2.line(c2, (0, x), (640, x), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            x += px_step
        y = 0
        while y < 640:
            cv2.line(c2, (y, 0), (y, 650), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            y += px_step

    def draw_results(self, img, boxes, confs, classes):
        window_name = 'final results'
        cv2.namedWindow(window_name)
        overlay = img.copy()
        final = img.copy()
        for box, conf, cls in zip(boxes, confs, classes):
            # draw rectangle
            x1, y1, x2, y2 = box
            conf = conf[0]
            cls_name = coco[cls]
            color = self.color_list[cls]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            # draw text
            cv2.putText(final, '%s %f' % (cls_name, conf), org=(x1, int(y1+10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255))
        cv2.addWeighted(overlay, 0.5, final, 1 - 0.5, 0, final)
        cv2.imshow(window_name, final)
        cv2.waitKey(20)
        return final
