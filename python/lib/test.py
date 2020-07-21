import cv2
# from Processor import Processor
# from Yolov5s import Yolov5s as Processor
from Processor import Processor
processor = Processor()
img = cv2.imread('./dogs.jpeg')
frame = processor.detect(img)
