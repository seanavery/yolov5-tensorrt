import cv2
import random
import colorsys
import numpy as np
import math
import sys

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
        
    def draw_object_grid(self, img, grids, conf_thres=0.1):
        """
            visualize object probabilty grid overlayed onto image

            img -- ndarray -- numpy array representing input image
            grids -- ndarray -- object probablity grid of shape (1, 3, x, y, 85)
            conf_thres -- float -- minimum objectness probability score
            -> None
            
        """
        for grid in grids:
            _, _, height, width, _ = grid.shape
            window_name = 'grid output {}'.format(height)
            cv2.namedWindow(window_name)

            px_step = 640 // height
            copy = img.copy()
            overlay = img.copy()
            x = 0
            while x < 640:
                cv2.line(copy, (0, x), (640, x), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                x += px_step
            y = 0
            while y < 640:
                cv2.line(copy, (y, 0), (y, 650), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                y += px_step
            
            for xi in range(width):
                for yi in range(height):
                    if grid[0][0][xi][yi][0] > conf_thres or grid[0][1][xi][yi][0] > conf_thres or grid[0][2][xi][yi][0] > conf_thres:
                        print('tile', xi, yi)
                        cv2.rectangle(overlay, (yi * px_step, xi * px_step), ((yi + 1) * px_step, (xi + 1) * px_step), (0, 255, 0), -1)

           
            cv2.addWeighted(overlay, 0.5, copy, 1 - 0.5, 0, copy)

            cv2.imshow(window_name, copy)
            cv2.waitKey(1000) 
    
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

        cv2.addWeighted(overlay, 0.5, copy, 1 - 0.5, 0, copy)
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
            cls = coco[cls]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            # draw text
            cv2.putText(final, '%s %f' % (cls, conf), org=(x1, int(y1+10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255))
        cv2.addWeighted(overlay, 0.5, final, 1 - 0.5, 0, final)
        cv2.imshow(window_name, final)
        cv2.waitKey(10000)
