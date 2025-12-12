#Pre-processing prepares a float32 model for quantization. Run the following command to pre-process model cartonize.onnx.
#python -m onnxruntime.quantization.preprocess --input ~/Dataset/Cartonize/Models/Cartonize.onnx --output ~/Dataset/Cartonize/Models/Cartonize-infer.onnx

import argparse
import numpy as np
import onnxruntime
import time
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

from onnx_data_reader import calib_data_reader

from defines import *
#from torchvision import transforms
#from models import Pix2PixHDGenerator
from utils2 import *


def benchmark(model_path):
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 4
    
    session = onnxruntime.InferenceSession(model_path, sess_options=so, providers=["OpenVINOExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print("Providers:", session.get_providers())

    total = 0.0 
    runs = 10
    input_data = np.zeros((1, 1, block_size, block_size), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def main():
    input_model_path = os.path.join(models_dir,"Cartonize-infer.onnx")      # run preprocess before
    output_model_path = os.path.join(models_dir,"Cartonize.uint8.onnx")
    calibration_dataset_path = sample_dir_A
    dr = calib_data_reader(calibration_dataset_path, input_model_path)

    # Calibrate and quantize model
    # Turn off model optimization during quantization
    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print("Calibrated and quantized model saved.")

    print("benchmarking fp32 model...")
    benchmark(input_model_path)

    print("benchmarking int8 model...")
    benchmark(output_model_path)


if __name__ == "__main__":
    main()
