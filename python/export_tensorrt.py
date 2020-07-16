import tensorrt as trt
import sys

"""
takes in onnx model
converts to tensorrt
"""

if __name__ == '__main__':
    model = 'lib/models/yolov5s-simple.onnx'
    logger = trt.Logger(trt.Logger.VERBOSE)
    EXPLICIT_BATCH = []
    if trt.__version__[0] >= '7':
        EXPLICIT_BATCH.append(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    with trt.Builder(logger) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, logger) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True

        with open(model, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit()
            
        # reshape input from 32 to 1
        print('network', network)
        shape = list(network.get_input(0).shape)
        print('shape', shape)

        engine = builder.build_cuda_engine(network)
        print('engine', engine)

        with open('yolov5s-simple-2.trt', 'wb') as f:
            f.write(engine.serialize())

