import cv2 
import sys
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import math

class Processor():
    def __init__(self):

        print('setting up Yolov5s-simple.trt processor')

        # load tensorrt engine
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        TRTbin = 'models/yolov5s-simple-2.trt'
        with open(TRTbin, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        
        # allocate memory
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append({ 'host': host_mem, 'device': device_mem })
            else:
                print('output name', binding)
                outputs.append({ 'host': host_mem, 'device': device_mem })
            
        # save to class
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream

        # post processing config
        filters = (80 + 5) * 3
        self.output_shapes = [
                (1, filters, 80, 80),
                (1, filters, 40, 40),
                (1, filters, 20, 20)]
    
        self.anchors = [
            [(116,90), (156,198), (373,326)],
            [(30,61), (62,45), (59,119)],
            [(10,13), (16,30), (33,23)],

        ]

        self.nl = len(self.anchors)
        self.nc = 80 # classes
        self.no = self.nc + 5 # outputs per anchor
        self.na = len(self.anchors[0])


    def detect(self, img):
        shape_orig_WH = (img.shape[1], img.shape[0])
        resized = self.pre_process(img)
        outputs = self.inference(resized)
        processed_outputs = self.post_process(outputs, img)
        
        return img

    def pre_process(self, img):
        print('original image shape', img.shape)
        img = cv2.resize(img, (640, 640))
        img = img.transpose((2, 0, 1)).astype(np.float16)
        img /= 255.0
        return img

    def inference(self, img):
        # copy img to input memory
        self.inputs[0]['host'] = np.ascontiguousarray(img)

        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # run inference
        self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle)

        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)


        # synchronize stream
        self.stream.synchronize()
        
        return [out['host'] for out in self.outputs]

    def post_process(self, outputs, img):
        print('outputs[0]', outputs[0].shape)
        print('outputs[1]', outputs[1].shape)
        print('outputs[2]', outputs[2].shape)
        reshaped = []
        for output, shape in zip(outputs, self.output_shapes):
            reshaped.append(output.reshape(shape))

        transposed = []
        grids = []
        for output in reshaped:
            bs, _, ny, nx = output.shape
            print('bs', bs)
            print('ny', ny)
            print('nx', nx)
            grid = self.make_grid(nx, ny)
            print('grid', grid)
            print('grid shape', grid.shape)
            sys.exit()
            grids.append(grid)
            output = output.reshape(bs, self.na, self.no, ny, nx)
            transposed.append(output.transpose(0, 1, 3, 4, 2))

        print('transposed', transposed)
        
        for output in transposed:
            print('output shape', output.shape[2:4])
            normalized = self.sigmoid_v(output)
            
            box_xy = normalized[..., 0:2]
            print('box_xy', box_xy)

        sys.exit()

    # create meshgrid as seen in yolov5 pytorch implementation 
    def make_grid(self, nx, ny):
        nx_vec = np.arange(nx)
        ny_vec = np.arange(ny)
        print('nx_vec', nx_vec)
        print('ny_vec', ny_vec)
        yv, xv = np.meshgrid(ny_vec, nx_vec)
        grid = np.stack((xv, yv), axis=2)
        grid = grid.reshape(1, 1, ny, nx, 2)
        print('grid hre', grid.shape)
        return grid

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_v(self, array):
        return np.reciprocal(np.exp(-array) + 1.0)

