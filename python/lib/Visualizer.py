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
            print('grid.shape', grid.shape)
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
                        print('mc', mc)
                        print('mci', mci) 
                        cv2.rectangle(copy, (xi * px_step, yi * px_step), ((xi + 1) * px_step, (yi + 1) * px_step), self.color_list[int(mci[0])], -1)
                               
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
        
    def draw_grid_tiles(self, img, shape=(20, 20)):
        print("yooo")

    def draw_grid(self, img, output, i):
        window_name = 'grid output {}'.format(i)
        cv2.namedWindow(window_name)
        # draw grid based on output scale (80, 40, 20)
        copy = img.copy()
        overlay = img.copy()
        c2 = img.copy()
        px_step = 640 // output.shape[2]
        print('px_step', px_step)
        x = 0
        while x < 640:
            cv2.line(c2, (0, x), (640, x), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            x += px_step
        y = 0
        while y < 640:
            cv2.line(c2, (y, 0), (y, 650), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            y += px_step

        # print(output[0][2][10][10])
        # print(output[0][2][15][10])

        object_probs = self.sigmoid_v(output[..., 4:5])

        xc = output[..., 4:5] > 0.4
        # object_probs = self.sigmoid_v(object_probs)
        # print('object_probs after', object_probs)

        flat = object_probs.flatten()
        print(flat.shape)

        average_object_prob = np.average(flat)
        print('average_object_prob', average_object_prob)

        mx = np.amax(flat)
        print('max object', mx)
        
        class_probs = self.sigmoid_v(output[..., 5:])
        box_scores = object_probs * class_probs
        
        # box_classes = np.argmax(box_scores, axis=-1)
        # box_class_scores = np.max(box_scores, axis=-1)
        # pos = np.where(box_class_scores > 0.01)
        # 
        # print('box_classes', box_classes.shape)
        # print('box_class_scores', box_class_scores.shape)
        # print('pos', pos)

        # classes = box_classes[pos]
        # scores = box_class_scores[pos]
        # print('classes', classes)
        # print('scores', scores)
        # print('class 16', coco[16])
        # print('class 1', coco[1])
        # print('class 10', coco[10])

        for xi in range(output.shape[2]):
            for yi in range(output.shape[2]):
                object_score = object_probs[0][i][xi][yi]
                #if class_probs[0][0][xi][yi][0] * object_score[0] > 0.0001:
                #    cv2.rectangle(overlay, (xi * px_step, yi * px_step), ((xi + 1) * px_step, (yi + 1) * px_step), (255, 0, 0), -1)

                if object_score[0] > 0.5: 
                    # check if it is a dog
                    mc = np.amax(class_probs[0][i][xi][yi], axis=-1)
                    print('mc', mc)
                    if mc > 0.2:
                        print('mc', mc)
                        mci = np.where(class_probs[0][2][xi][yi] == mc)
                        print('mci', mci, class_probs[0][2][xi][yi][0])
                        print('box', (xi * px_step, yi * px_step), ((xi + 1) * px_step, (yi + 1) * px_step))
                        cv2.rectangle(overlay, (xi * px_step, yi * px_step), ((xi + 1) * px_step, (yi + 1) * px_step), (0, 255, 0), -1)

        cv2.addWeighted(overlay, 0.5, c2, 1 - 0.5, 0, c2)
        cv2.imshow(window_name, c2)
        cv2.waitKey(10000)
    
    def sigmoid_v(self, array):
        return np.reciprocal(np.exp(-array) + 1.0)

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
