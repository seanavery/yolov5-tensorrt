import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

class Processor():
    def __init__(self):
        logger = trt.Logger(trt.Logger.INFO)
        model = 'models/yolov5s-simple.trt'

        with open(model, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
       
        self.context = engine.create_execution_context()

        # allocate memory
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        print('stream', stream)
        for binding in engine:
            print('binding', binding)
            size = trt.volume(engine.get_binding_shape(binding)) # * \
                   # engine.max_batch_size
            print('size', size)
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
    
    def detect(self, img):
        return img
