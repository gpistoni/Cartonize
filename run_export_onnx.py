import torch
import os
import onnx
import onnxruntime
import numpy as np
from utils import getFilemodel, exportModelOnnx, getFileOnnx, quantizeModelOnnx
from onnxruntime.quantization import quantize_dynamic, QuantType,  QuantFormat, quantize_static
from onnxconverter_common import float16
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
from defines import *
from torchvision import transforms
from models import Pix2PixHDGenerator
from utils2 import *

############################################################################################################################################################
# PER ESPORTARE IMPOSTARE: use_checkpoint: False
############################################################################################################################################################

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_attuale = datetime.now().strftime("%Y%m%d")
  
    onnx_file_fp32 = os.path.join(models_dir,"Cartonize.onnx")
    onnx_file_uint8 = os.path.join(models_dir,"Cartonize.uint8.onnx")

    ############################################################################################
    #model = initialize_model(yaml_file)    
    G = Pix2PixHDGenerator(in_ch=1, out_ch=1, ngf=block_size).to(device)
    if os.path.exists(checkpoint_file):
        ckpt = load_checkpoint(checkpoint_file, device)
        if ckpt is not None:
            G.load_state_dict(ckpt['G_state'])
            #D.load_state_dict(ckpt['D_state'])
            #opt_G.load_state_dict(ckpt['opt_G_state'])
            #opt_D.load_state_dict(ckpt['opt_D_state'])
            #train_losses = ckpt.get('train_losses', [])
            #train_accuracies = ckpt.get('train_accuracies', [])
            #val_accuracies = ckpt.get('val_accuracies', [])
            #start_epoch = ckpt.get('epoch', 0) + 1
            print(f"Ripreso da checkpoint {checkpoint_file}")

        model = Pix2PixHDGenerator(in_ch=1, out_ch=1, ngf=block_size).to(device)      
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["G_state"])

        # After creating model1
        model1 = model  # Ensure this returns a valid model instance
        
        # Check if model1 is a valid instance of torch.nn.Module
        if isinstance(model1, torch.nn.Module):
            print("model1 is a valid instance of torch.nn.Module.")
        else:
            print("model1 is NOT a valid instance of torch.nn.Module.")

        # Imposta il modello in modalità di valutazione
        model1.eval()
        model1.to('cpu')

        # Definisci un input fittizio con le dimensioni corrette per il tuo modello
        dummy_input = torch.randn(1, 1, block_size, block_size).to('cpu')

        # test del modello
        output = model1(dummy_input)  # Esegui il modello

        torch.onnx.export(model1, dummy_input, onnx_file_fp32,
                      export_params=True,
                      opset_version=17,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])
        print(f"Modello esportato in formato ONNX: {onnx_file_fp32}")  


        

        ################################################################################
        #test FP32
        dummy_input = torch.randn(1, 1, block_size, block_size).to("cpu")

        model2 = onnx.load(onnx_file_fp32)
        onnx.checker.check_model(model2)

        session = onnxruntime.InferenceSession(onnx_file_fp32)
        outputs2 = session.run(None, {"input": dummy_input.numpy()})
        #print(outputs2)

        ################################################################################
        #convert INT8
        # Load the original ONNX model
        #model3 = onnx.load(onnx_file_fp32)
        #onnx.checker.check_model(model3)

        #Perform dynamic quantization
        #quantize_dynamic(onnx_file_fp32, onnx_file_uint8, weight_type = QuantType.QUInt8)
        #print(f"Quantized model saved to: {onnx_file_uint8}") 
        ################################################################################

        #test INT8
        #model4 = onnx.load(onnx_file_uint8)
        #onnx.checker.check_model(model4)

        #session = onnxruntime.InferenceSession(onnx_file_uint8)
        #outputs4 = session.run(None, {"input": dummy_input.numpy()})
        #print(outputs4)

        ################################################################################
        # Prune  # Assume `model` is your PyTorch model
        #for name, module in model.named_modules():
        #    if isinstance(module, torch.nn.Conv2d):  # Example for Linear layers
        #        prune.l1_unstructured(module, name='weight', amount=0.2)  # Prune 20% of weights

        # Export to ONNX
        #torch.onnx.export(model, dummy_input, onnx_file_fp32p)
        #print(f"Pruned model saved to: {onnx_file_fp32p}")



if __name__ == "__main__":
    main()