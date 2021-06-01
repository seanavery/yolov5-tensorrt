# yolov5-tensorrt 

> port pytorch/onnx yolov5 model to run on a Jetson Nano

![](docs/demo.gif)

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

## compile onnx model to trt

```
python3 export_tensorrt.py --help
usage: export_tensorrt.py [-h] [-m MODEL] [-fp FLOATINGPOINT] [-o OUTPUT]

compile Onnx model to TensorRT

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        onnx file location inside ./lib/models
  -fp FLOATINGPOINT, --floatingpoint FLOATINGPOINT
                        floating point precision. 16 or 32
  -o OUTPUT, --output OUTPUT
                        name of trt output file
```

## run demo

```
python3 demo.py --image=./path/to/image.jpg --model=./path/to/model.trt
```

___

## performance

```
for now, only testing initial inference performance
nms, and post processing are slow rn
```

| model  |  fp precision  | input size |  time (ms)   |
| ------------- | ------------- | ---------- | ---- |
| small-simple  |  32  |  640x640x3  | 221 ms |
| small-simple  |  16  |  640x640x3  | ?  |

___

## object probability

![](docs/object_grids.png)


