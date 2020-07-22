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

        self.strides = np.array([8., 16., 32.])
    
        anchors = np.array([
            [[116,90], [156,198], [373,326]],
            [[30,61], [62,45], [59,119]],
            [[10,13], [16,30], [33,23]],

        ])

        self.nl = len(anchors)
        self.nc = 80 # classes
        self.no = self.nc + 5 # outputs per anchor
        self.na = len(anchors[0])

        a = anchors.copy().astype(np.float32)
        a = a.reshape(self.nl, -1, 2)

        self.anchors = a.copy()
        self.anchor_grid = a.copy().reshape(self.nl, 1, -1, 1, 1, 2)

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
            grid = self.make_grid(nx, ny)
            grids.append(grid)
            output = output.reshape(bs, self.na, self.no, ny, nx)
            transposed.append(output.transpose(0, 1, 3, 4, 2))

        z = []
        for i, output in enumerate(transposed):
            print('output shape', output.shape)
            y = self.sigmoid_v(output)
            print('self.anchors shape', self.anchors.shape)
            # print('y', y)
            
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grids[i]) * self.strides[i] # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i] # wh

            z.append(y.reshape(1, -1, self.no))

        pred = np.concatenate(z, 1)
        pred = self.nms(pred)



    # create meshgrid as seen in yolov5 pytorch implementation 
    def make_grid(self, nx, ny):
        nx_vec = np.arange(nx)
        ny_vec = np.arange(ny)
        yv, xv = np.meshgrid(ny_vec, nx_vec)
        grid = np.stack((xv, yv), axis=2)
        grid = grid.reshape(1, 1, ny, nx, 2)
        return grid

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_v(self, array):
        return np.reciprocal(np.exp(-array) + 1.0)
    
    def nms(self, pred, conf_thres=0.8, iou_thres=0.6, classes=None):
        nc = pred[0].shape[1] - 5
        xc = pred[..., 4] > conf_thres
        print('nc', nc)
        print('xc', xc)

        # settings
        min_wh, max_wh = 2, 4096
        max_det = 10
        
        output = [None] * pred.shape[0]
        print('output', output)
        
        for xi, x in enumerate(pred):
            # only consider thresholded confidences
            x = x[xc[xi]]
            print('x', x.shape)
            
            # calcualte confidence (obj_conf * cls_conf)
            x[:, 5:] *= x[:, 4:5]
            box = self.xywh2xyxy(x[:, :4])
            print('box', box.shape)
            
        sys.exit()

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y


