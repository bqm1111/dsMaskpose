import argparse

import tensorrt as trt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, help="Path to PTH file")
    parser.add_argument("--trt_path", type=str, help="Path to TRT file")
    parser.add_argument("--max_batch_size", type=int, help="Max batch size")
    args = parser.parse_args()

    # logger to capture errors, warnings, and other information during the build and inference phases
    TRT_LOGGER = trt.Logger(trt.Logger.Severity.VERBOSE)

    # test built engine
    runtime = trt.Runtime(TRT_LOGGER)
    with open(args.trt_path, "rb") as f:
        serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)

    context = engine.create_execution_context()
    print(context.get_binding_name(0))
    print(context.get_binding_shape(0))
    print(context.get_binding_shape(1))
    print(context.get_binding_shape(2))
    print(context.get_binding_shape(3))
    # help(context)
    

    