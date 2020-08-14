import cv2
from Processor import Processor
from Visualizer import Visualizer
import sys

def main():
    # setup processor and visualizer
    processor = Processor()
    visualizer = Visualizer()

    # fetch input
    # img = cv2.imread('./dogs.jpeg')
    img = cv2.imread('./sample_720p.jpg')

    # inference
    output = processor.detect(img) 
    img = cv2.resize(img, (640, 640))

    # object visualization
    object_grids = processor.extract_object_grids(output)
    visualizer.draw_object_grid(img, object_grids, 0.5)

    # class visualization
    class_grids = processor.extract_class_grids(output)
    visualizer.draw_class_grid(img, class_grids, 0.01)

    # bounding box visualization
    boxes = processor.extract_boxes(output)
    visualizer.draw_boxes(img, boxes)

    # final results
    boxes, confs, classes = processor.post_process(output)
    visualizer.draw_results(img, boxes, confs, classes)

if __name__ == '__main__':
    main()   
