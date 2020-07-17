import cv2 
import sys
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

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
                outputs.append({ 'host': host_mem, 'device': device_mem })
            
        # save to class
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream

        # post processing c onfig
        filters = (80 + 5) * 3
        self.output_shapes = [
                (1, filters, 80, 80),
                (1, filters, 40, 40),
                (1, filters, 20, 20)]

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
        reshaped_outputs = []
        for output, shape in zip(outputs, self.output_shapes):
            print("output, shape", output.shape, shape)
            reshaped_outputs.append(output.reshape(shape))

        print('reshaped_outputs', reshaped_outputs[0].shape, reshaped_outputs[1].shape, reshaped_outputs[2].shape)
        processed_outputs = []
        for output in reshaped_outputs:
            processed_outputs.append(self.reshape(output))

        boxes, categories, confidences = self.process_outputs(processed_outputs)
        
        sys.exit()
        self.non_max_supression(reshaped_outputs)

    def reshape(self, output):
        # convert from NCHW to NHWC
        output = np.transpose(output, [0, 2, 3 , 1])
        _, height, width, _ = output.shape
        print('height', height)
        print('width', width)
        print('output here', output.shape)

        return np.reshape(output, (height, width, 3, 85))
    

    def non_max_supression(self, outputs):
        # number of classes
        box = outputs[0]
        scores = outputs[1]
        indices = outputs[2]
        print('box_shape', box.shape)
        print('scores_shape', scores.shape)
        print('indices_shape', indices.shape)

        print('first box', box[0])
        print('second box', box[1])
        print('first score', scores[0])
        print('second score', scores[1])
        print('first index', indices[0])
        print('second index', indices[1])

        sys.exit()
        
        out_classes, out_scores, out_classes = [], [], []
        for idx in indices:
            print('idx', idx)
            out_classes.append(idx[1])
            out_scores.append(scores[tuple(idx)])
            idx_1 = (idx[0], idx[2])
            out_boxes.append(boxes[idx_1])

        print('out_classes', out_classes)
        print('out_scores', out_scores)
        print('out_classes', out_classes)

    
        sys.exit()
        

