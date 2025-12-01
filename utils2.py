import os
import numpy as np
import torch
from PIL import Image
from datetime import datetime
from onnxruntime.quantization import quantize_dynamic, QuantType
import cv2

#########################################################################################################
def save_checkpoint(state, filename):
    """Salva lo stato del checkpoint su file (atomico via tmp -> rename)."""
    tmp = filename + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, filename)

def load_checkpoint(filename, device):
    """Carica checkpoint se esiste, restituisce dizionario o None."""
    if not os.path.exists(filename):
        return None
    checkpoint = torch.load(filename, map_location=device)
    return checkpoint

#########################################################################################################