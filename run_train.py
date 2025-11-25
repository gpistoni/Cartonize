import os
import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import make_grid, save_image
from models import Pix2PixHDGenerator, NLayerDiscriminator
from dataset import PairedGrayDataset
from losses import adversarial_loss, l1_loss, VGGLoss
import sys
import time
from defines import *

def save_side_by_side(real_A, fake_B, out_path, nrow=4):
    # real_A, fake_B: tensori [B,1,H,W], valori in [-1,1]
    # nrow: numero di coppie per riga
    # converti in [0,1]
    real = (real_A.clamp(-1,1) + 1) * 0.5
    fake = (fake_B.clamp(-1,1) + 1) * 0.5

    # duplicate canale per visualizzare come RGB
    real_rgb = real.repeat(1,3,1,1)
    fake_rgb = fake.repeat(1,3,1,1)

    # per ogni sample concatena orizzontalmente real|fake -> tensor [B,3,H,2W]
    pair_list = []
    for r, f in zip(real_rgb, fake_rgb):
        pair = torch.cat([r, f], dim=2)  # concat su width: CxHx(2W)
        pair_list.append(pair.unsqueeze(0))
    pairs = torch.cat(pair_list, dim=0)  # [B,3,H,2W]

    # crea una griglia e salva
    grid = make_grid(pairs, nrow=nrow, pad_value=1.0)  # pad_value = 1 (white)
    save_image(grid, out_path)

def save_side_by_side(real_A, real_B, fake_B, out_path, nrow=4):
    # real_A, real_B, fake_B: tensori [B,1,H,W], valori in [-1,1]
    # nrow: numero di triplette per riga

    # converti in [0,1]
    real_A = (real_A.clamp(-1,1) + 1) * 0.5
    real_B = (real_B.clamp(-1,1) + 1) * 0.5
    fake_B = (fake_B.clamp(-1,1) + 1) * 0.5

    # duplicare canale per visualizzare come RGB (C=3)
    real_A_rgb = real_A.repeat(1, 3, 1, 1)
    real_B_rgb = real_B.repeat(1, 3, 1, 1)
    fake_B_rgb = fake_B.repeat(1, 3, 1, 1)

    # per ogni sample concatena orizzontalmente real_A | real_B | fake_B -> [B,3,H,3W]
    triplet_list = []
    for a, b, f in zip(real_A_rgb, real_B_rgb, fake_B_rgb):
        triplet = torch.cat([a, b, f], dim=2)  # concat su width: C x H x (3W)
        triplet_list.append(triplet.unsqueeze(0))
    triplets = torch.cat(triplet_list, dim=0)  # [B,3,H,3W]

    # crea una griglia e salva; pad_value=1.0 per spazi bianchi
    grid = make_grid(triplets, nrow=nrow, pad_value=1.0)
    save_image(grid, out_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str,  default='/home/giulipis/Dataset/Cartonize' )
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(trainOutput_dir, exist_ok=True)

    dataset = PairedGrayDataset(args.dataroot, 'Train_Samples', block_size)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)

    G = Pix2PixHDGenerator(in_ch=1, out_ch=1, ngf=block_size).to(device)
    D = NLayerDiscriminator(in_ch=2).to(device)

    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5,0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5,0.999))

    vgg = VGGLoss(device=device).to(device)

    # Liste per tracciare perdita e accuratezza durante il training
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    # Altre variabili
    ldl = len(dataloader)

    for epoch in range(args.epochs):

        epoch_start_time = time.time()
        running_loss = 0.0                  # Accumulatore della loss per questa epoca (somma pesata)
        total_train = 0                     # Contatore dei sample visti finora 

        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(device); 
            real_B = real_B.to(device)

            # ----- Train D -----
            with torch.no_grad():
                fake_B, _ = G(real_A)
            real_AB = torch.cat([real_A, real_B], dim=1)
            fake_AB = torch.cat([real_A, fake_B.detach()], dim=1)

            pred_real = D(real_AB)
            pred_fake = D(fake_AB)
            loss_D = (adversarial_loss(pred_real, torch.ones_like(pred_real)) + adversarial_loss(pred_fake, torch.zeros_like(pred_fake))) * 0.5

            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # ----- Train G -----
            fake_B, coarse = G(real_A)
            fake_AB = torch.cat([real_A, fake_B], dim=1)
            pred_fake = D(fake_AB)
            loss_G_GAN = adversarial_loss(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = l1_loss(fake_B, real_B) * 100.0
            loss_G_VGG = vgg(fake_B, real_B) * 10.0

            loss_G = loss_G_GAN + loss_G_L1 + loss_G_VGG

            opt_G.zero_grad(); 
            loss_G.backward(); 
            opt_G.step()

            # ----- Logging -----
            running_loss += loss_G.item() * batch_size
            total_train += real_A.size(0)                # Aggiorna il totale dei sample visti
            elapsed = time.time() - epoch_start_time
            samples_per_sec = total_train / elapsed if elapsed > 0 else float('inf')
            sys.stdout.write(f"\rSample {i}/{ldl} | processed {total_train} samples | {samples_per_sec:.2f} samples/s")
            sys.stdout.flush()  
            
            #time.sleep(1)

            #----- Save Images -----
            #if i % 200 == 0:
            #    out = (fake_B + 1) * 0.5  # [B,1,H,W]
            #    out_vis = out.repeat(1,3,1,1)  # duplicate for viewing
            #    save_image(out_vis, os.path.join(trainOutput_dir, f'fake_epoch{epoch}_iter{i}.png'), nrow=4)
            #    out = (real_A + 1) * 0.5  # [B,1,H,W]
            #    out_vis = out.repeat(1,3,1,1)  # duplicate for viewing
            #    save_image(out_vis, os.path.join(trainOutput_dir, f'real_epoch{epoch}_iter{i}.png'), nrow=4)

             #----- Save Images -----
            if i % 200 == 0:
                # crea una griglia e salva
                with torch.no_grad():
                    fake_B, _ = G(real_A)   # real_A batch usato nel training
                    save_side_by_side(real_A.cpu(), real_B.cpu(), fake_B.cpu(),  os.path.join(trainOutput_dir, f'compare_epoch{epoch}_iter{i}.png'), nrow=4)


        # Calcola la loss media per sample nell'epoca
        train_loss = running_loss / total_train
        # Salva metriche per traccia/storico
        train_losses.append(train_loss)

        print(f'Epoch {epoch+1}/{args.epochs} done Train Loss: {train_loss:.4f} ')
