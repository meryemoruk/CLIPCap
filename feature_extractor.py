import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Sizin Dosyalarınızdan Importlar ---
# Dosya yollarının Colab dizin yapısına uygun olduğundan emin olun
from model.model_encoder import Encoder
from data.SECOND_CC.SECONDCC import SECONDCCDataset

# --- AYARLAR ---
DATA_FOLDER = "/content/CLIPCap/data/SECOND_CC"  # SECOND veri setinin olduğu klasör
LIST_PATH = "/content/CLIPCap/data/SECOND_CC" # Train/Val listelerinin olduğu yer
SAVE_DIR = "/content/SECOND_CRADIO_Features" # Özelliklerin kaydedileceği yer (Drive önerilir)
SPLIT = "train" # 'train' veya 'val' veya 'test'
BATCH_SIZE = 32 # L4 GPU için 32 veya 64 uygundur

def extract_and_save():
    # 1. Kayıt Klasörünü Oluştur
    save_path_split = os.path.join(SAVE_DIR, SPLIT)
    os.makedirs(save_path_split, exist_ok=True)
    print(f"Features will be saved to: {save_path_split}")

    # 2. Modeli Yükle (Encoder)
    print("Loading Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(network="c-radio")
    encoder = encoder.to(device)
    encoder.eval() # Kesinlikle eval modunda olmalı

    # 3. Dataset ve DataLoader
    print("Loading Dataset...")
    dataset = SECONDCCDataset(
        data_folder=DATA_FOLDER, 
        list_path=LIST_PATH, 
        split=SPLIT, 
        token_folder=None, # Tokenlara ihtiyacımız yok, sadece görüntü işleyeceğiz
        vocab_file=None
    )
    
    # num_workers=2 veya 4 Colab'da veri yüklemeyi hızlandırır
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Starting extraction for {len(dataset)} images...")

    # 4. Döngü
    with torch.no_grad(): # Gradyan hesaplamayı kapat (Hafıza tasarrufu)
        for batch in tqdm(dataloader):
            # Dataset __getitem__ çıktısı: imgA, imgB, token_all, ..., name
            imgA_batch, imgB_batch, _, _, _, _, names = batch
            
            # Encoder, PIL Image listesi bekler (Processor kullanımı için)
            # Dataset numpy array döndürüyor, bunları PIL'e çevirmemiz lazım.
            pil_imgs_A = []
            pil_imgs_B = []
            
            for i in range(len(names)):
                # Tensor/Numpy -> PIL (RGB)
                # imgA_batch[i] shape: (256, 256, 3)
                img_a = Image.fromarray(imgA_batch[i].numpy().astype('uint8'))
                img_b = Image.fromarray(imgB_batch[i].numpy().astype('uint8'))
                pil_imgs_A.append(img_a)
                pil_imgs_B.append(img_b)

            # 5. C-RADIO Encoder'dan Geçir
            # Çıktı Shape: [Batch, 1280, 16, 16]
            featA, featB, _ = encoder(pil_imgs_A, pil_imgs_B)

            # 6. Diske Kaydet
            # Her resmi kendi ismiyle tek tek kaydediyoruz (Random access için en iyisi)
            for i, name in enumerate(names):
                # Veriyi CPU'ya alıp kaydedelim
                save_dict = {
                    "featA": featA[i].cpu().clone(),
                    "featB": featB[i].cpu().clone()
                }
                
                # Dosya yolu: .../train/00042.pt
                file_path = os.path.join(save_path_split, name + ".pt")
                torch.save(save_dict, file_path)

    print("Extraction Completed!")

if __name__ == "__main__":
    extract_and_save()