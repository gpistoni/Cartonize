import torch, torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3,1,1,bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3,1,1,bias=False),
            nn.InstanceNorm2d(dim)
        )
    def forward(self,x): return x + self.block(x)

class GeneratorStage(nn.Module):
    def __init__(self, in_ch, out_ch, ngf=64, n_blocks=6):
        super().__init__()
        layers = [nn.Conv2d(in_ch, ngf, 7,1,3,bias=False), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        cur = ngf
        for _ in range(2):
            layers += [nn.Conv2d(cur, cur*2, 3,2,1,bias=False), nn.InstanceNorm2d(cur*2), nn.ReLU(True)]
            cur *= 2
        for _ in range(n_blocks): layers += [ResBlock(cur)]
        for _ in range(2):
            layers += [nn.ConvTranspose2d(cur, cur//2, 3,2,1,output_padding=1,bias=False), nn.InstanceNorm2d(cur//2), nn.ReLU(True)]
            cur //= 2
        layers += [nn.Conv2d(cur, out_ch, 7,1,3), nn.Tanh()]
        self.model = nn.Sequential(*layers)
    def forward(self,x): return self.model(x)

class Pix2PixHDGenerator(nn.Module):
    def __init__(self, in_ch, out_ch, ngf):
        super().__init__()
        self.stage1 = GeneratorStage(in_ch, out_ch, ngf, n_blocks=4)
        self.stage2 = GeneratorStage(in_ch + out_ch, out_ch, ngf, n_blocks=6)
    def forward(self,x):
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        coarse = self.stage1(x_half)
        coarse_up = F.interpolate(coarse, size=x.shape[2:], mode='bilinear', align_corners=False)
        inp2 = torch.cat([x, coarse_up], dim=1)
        refined = self.stage2(inp2)
        return refined, coarse_up

class NLayerDiscriminator(nn.Module):
    def __init__(self, in_ch=2, ndf=64, n_layers=3):  # in_ch = concat(input, target) => 1+1=2
        super().__init__()
        kw=4; pad=1
        seq = [nn.Conv2d(in_ch, ndf, kw,2,pad), nn.LeakyReLU(0.2,True)]
        nf=ndf
        for n in range(1,n_layers):
            stride = 2 if n < n_layers-1 else 1
            seq += [nn.Conv2d(nf, min(nf*2,512), kw, stride, pad, bias=False),
                    nn.InstanceNorm2d(min(nf*2,512)), nn.LeakyReLU(0.2,True)]
            nf = min(nf*2,512)
        seq += [nn.Conv2d(nf, 1, kw,1,pad)]
        self.model = nn.Sequential(*seq)
    def forward(self,x): return self.model(x)
