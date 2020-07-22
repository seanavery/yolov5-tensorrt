import cv2
# from Processor import Processor
# from Yolov5s import Yolov5s as Processor
from Processor import Processor
from Visualizer import Visualizer

window = cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)
processor = Processor()
visualizer = Visualizer()
img = cv2.imread('./dogs.jpeg')
pred = processor.detect(img)
img = cv2.resize(img, (640, 640))
img = visualizer.draw(img, pred)
cv2.imshow('camera', img)
while True:
    cv2.waitKey(3000)
