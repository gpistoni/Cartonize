from PIL import Image, ImageFilter, ImageOps
import numpy as np
import os
import cv2
import random
import shutil
from defines import *
############################################################################################################################################################
def check_couple(out_dir, out_folder1, out_folder2):
    i=0
    output_folderA = os.path.join(out_dir, out_folder1)
    output_folderB = os.path.join(out_dir, out_folder2)

    image_filenamesA = [x for x in os.listdir(output_folderA)] 
    image_filenamesB = [x for x in os.listdir(output_folderB)] 

    a_set = set(image_filenamesA)
    b_set = set(image_filenamesB)

    print(f"Total files: {len(image_filenamesA) + len(image_filenamesB)}")

    for fname in image_filenamesA:
        if fname not in b_set:
            os.remove(os.path.join(output_folderA, fname))
            i+=1
            print(f"Removed uncoupled:", os.path.join(output_folderA, fname))
    
    for fname in image_filenamesB:
        if fname not in a_set:
            os.remove(os.path.join(output_folderB, fname))
            i+=1
            print(f"Removed uncoupled:", os.path.join(output_folderB, fname))
    
    print(f"Removed uncoupled:{i}")

    
############################################################################################################################################################
def split_image(image_dir, out_dir, in_name, out_name, out_folder, tile_size, tile_step, resdown=1):
    
    # sorgente
    image_path = os.path.join(image_dir, in_name)
    output_folder = os.path.join(out_dir, out_folder)

    # Carica l'immagine
    image = Image.open(image_path)

    # Eventuale riduzione
    if (resdown!=1):
        i = image.size   # current size (height,width)
        i = (int)(i[0]/resdown), (int)(i[1]/resdown)  # new size
        image = image.resize(i, resample=Image.BICUBIC)

    img_width, img_height = image.size
     
    # aggiunge un bordo
    #image = ImageOps.expand(image, border=32, fill='black')
    
    # Assicurati che la cartella di output esista
    os.makedirs(output_folder, exist_ok=True)

    # Dividi l'immagine in sottosezioni
    i=0
    for top in range(0, img_height, tile_step):
        for left in range(0, img_width, tile_step):

            # Definisci il box della sottosezione
            box = (left, top, left + tile_size, top + tile_size)

            sub_image = image.crop(box).convert('L')    
            arr = np.array(sub_image, dtype=np.float64)  # evita overflow
            # varianza popolazione (ddof=0)
            var_pop = np.var(arr)

            if ( var_pop > 500 ):
                sub_imageMirrored = ImageOps.mirror(sub_image)
                sub_imageRotated180 = sub_image.rotate(180)

                # Salva la sottosezione           
                sub_image_name = out_name + f"_{top}_{left}.bmp"
                sub_image.save(os.path.join(output_folder, sub_image_name))
                i+=1

                sub_image_name = out_name + f"_{top}_{left}M.bmp"
                sub_imageMirrored.save(os.path.join(output_folder, sub_image_name))
                i+=1

                #sub_image_name = out_name + f"_{top}_{left}R.bmp"
                #sub_imageRotated180.save(os.path.join(output_folder, sub_image_name))
                #i+=1
        
    print(f"Immagine suddivisa: {image_path} in parti:{i}")

############################################################################################################################################################
tile_step_MB = (int)(block_size * 77 / 100)

image_filenames = [x for x in os.listdir(fullImage_dir)] 
image_filenames.sort()

if os.path.exists(sample_dir):
    shutil.rmtree(sample_dir)

os.makedirs(sample_dir)

bi = 0
for fname in image_filenames:
    if fname.startswith("MB"):  # sintetico vs1
        fnameAA = fname.replace("MB_", "MA_")
        split_image(fullImage_dir, sample_dir, fnameAA, "img" + str(bi), "A", block_size, tile_step_MB)
        split_image(fullImage_dir, sample_dir, fname, "img" + str(bi), "B", block_size, tile_step_MB) 
        bi += 1

check_couple(sample_dir, "A", "B") 

print(f"Numero file MB+MC: {bi}")
