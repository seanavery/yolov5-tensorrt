# yolov5-tensorrt 

> port pytorch/onnx yolov5 model to run on a Jetson Nano

`ipynb` is for testing pytorch code and exporting onnx models inside Google Colab

`python` python code runs numpy/tensorrt implementation on Jetson Nano 


- [x] convert yolov5 onnx model to tensorrt
- [x] pre-process image 
- [x] run inference against input using tensorrt engine
- [x] post process output (forward pass)
- [ ] apply nms thresholding on candidate boxes
- [ ] visualize results
