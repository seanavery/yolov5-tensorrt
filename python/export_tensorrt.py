import tensorrt as trt
import sys
import argparse

"""
takes in onnx model
converts to tensorrt
"""

def cli():
    desc = 'compile onnx model to TensorRT'
    parser = argparse.ArgumentParser(description=desc)
    print('parser', parser)
    parser.add_argument('-model', help='onnx file location inside ./lib/models')
    parser.add_argument('-fp16', help='floating point precision. 16 or 32')
    parser.add_argument('-output', help='name of trt output file')
    args = parser.parse_args()
    model = args.model or 'yolov5s-simple.onnx'
    fp = args.fp16 or False
    if fp:
        fp='16' 
    else:
        fp='32'
    output = args.output or 'yolov5s-simple-{}.trt'.format(fp)
    return {
        'model': model,
        'fp': fp,
        'output': output
    }

if __name__ == '__main__':
    args = cli()
    model = 'lib/models/{}'.format(args['model'])
    output = 'lib/models/{}'.format(args['output'])
    logger = trt.Logger(trt.Logger.VERBOSE)
    EXPLICIT_BATCH = []
    print('trt version', trt.__version__)
    if trt.__version__[0] >= '7':
        EXPLICIT_BATCH.append(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    with trt.Builder(logger) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, logger) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        if args['fp'] == '16':
            builder.fp16_mode = True

        with open(model, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            
        # reshape input from 32 to 1
        shape = list(network.get_input(0).shape)
        engine = builder.build_cuda_engine(network)
        with open(output, 'wb') as f:
            f.write(engine.serialize())
