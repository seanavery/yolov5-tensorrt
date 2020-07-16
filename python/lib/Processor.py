import tensorrt as trt
import pycuda.driver as cuda

class Processor():
    def __init__(self):
        logger = trt.Logger(trt.Logger.INFO)
        model = 'models/yolov5s-simple.trt'

        with open(model, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            print('engine', engine)
            print('engine max batch size', engine.max_batch_size)
    
    def detect(self, img):
        return img
