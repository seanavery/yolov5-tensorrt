import cv2
import sys
import argparse

from Processor import Processor
from Visualizer import Visualizer

def cli():
    desc = 'Run TensorRT yolov5 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-model',  help='trt engine file located in ./models', required=False)
    parser.add_argument('-image', help='image file path', required=False)
    args = parser.parse_args()
    model = args.model or 'yolov5s-simple.trt'
    img = args.image or 'sample_720p.jpg'
    return { 'model': model, 'image': img }

def main():
    # parse arguments
    args = cli()

    # setup processor and visualizer
    processor = Processor(model=args['model'])
    visualizer = Visualizer()

    # fetch input
    print('image arg', args['image'])
    img = cv2.imread(args['image'])

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
