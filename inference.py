import os
import argparse
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.utils import save_image
from models import Pix2PixHDGenerator
from defines import *

def load_image(path, size):
    img = Image.open(path).convert('L')
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    return transform(img).unsqueeze(0)  # 1x1xHxW

def save_gray_tensor(tensor, out_path):
    # tensor in [-1,1], shape 1x1xHxW -> convert to [0,1] and duplicate to 3ch for viewing
    out = (tensor.clamp(-1,1) + 1) * 0.5
    out_vis = out.repeat(1,3,1,1)
    save_image(out_vis, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='path to generator .pth')
    parser.add_argument('--input', required=True, help='input image or folder')
    parser.add_argument('--out_dir', default='inference_out', help='output folder')
    parser.add_argument('--size', type=int, default=512, help='short side resize (square)')
    parser.add_argument('--device', default='cuda', help='cpu or cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(trainOutput_dir, exist_ok=True)

    G = Pix2PixHDGenerator(in_ch=1, out_ch=1, ngf=block_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    # if checkpoint contains state_dict under key:
    if 'state_dict' in ckpt:
        G.load_state_dict(ckpt['state_dict'])
    else:
        G.load_state_dict(ckpt)
    G.eval()

    paths = []
    if os.path.isdir(args.input):
        exts = ('.png','.jpg','.jpeg')
        for f in sorted(os.listdir(args.input)):
            if f.lower().endswith(exts):
                paths.append(os.path.join(args.input, f))
    else:
        paths = [args.input]

    with torch.no_grad():
        for p in paths:
            img = load_image(p, args.size).to(device)
            fake, _ = G(img)
            name = os.path.basename(p)
            out_path = os.path.join(trainOutput_dir, f'cartoon_{name}')
