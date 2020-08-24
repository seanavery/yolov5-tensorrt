# yolov5-tensorrt 

> port pytorch/onnx yolov5 model to run on a Jetson Nano

`ipynb` is for testing pytorch code and exporting onnx models using Google Colab

`python` code runs numpy/tensorrt implementation on Jetson Nano 

```
├── python
│   ├── lib
|       ├── demo.py
|       ├── Processor.py
|       ├── Visualizer.py
|       ├── classes.py
|       └── models
|           ├── yolov5s-simple-32.trt
|           ├── yolov5s-simple-16.trt
|           └── yolov5s-simple.onnx
│   └── export_tensorrt.py
```

- [x] convert yolov5 onnx model to tensorrt
- [x] pre-process image 
- [x] run inference against input using tensorrt engine
- [x] post process output (forward pass)
- [x] apply nms thresholding on candidate boxes
- [x] visualize results

___

## run demo

```
python3 demo.py -image=./path/to/image.jpg -model=./path/to/model.trt
```

___

## performance

```
for now, only testing initial inference performance
nms, and post processing are slow rn
```

| model  |  fp precision  | input size |  time (ms)   |
| ------------- | ------------- | ---------- | ---- |
| small-simple  |  32  |  640x640x3  | 15.46 |
| small-simple  |  16  |  640x640x3  | 9.47  |

___

## object probability

![](docs/object_grids.png)


