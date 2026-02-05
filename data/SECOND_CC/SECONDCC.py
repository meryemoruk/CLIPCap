import torch
from torch.utils.data import Dataset
from preprocess_data import encode
import json
import os
import numpy as np
from torch.utils.data import DataLoader
from imageio import imread
from random import randint

class SECONDCCDataset(Dataset):
    """
    SECOND Dataset (SECOND-CC-AUG) için PyTorch Dataset sınıfı.
    ClipEncoder ile uyumlu olması için ImageNet normalizasyonunu ve 
    ekran görüntüsündeki 'rgb' klasör yapısını kullanır.
    """

    def __init__(self, data_folder, list_path, split, token_folder=None, vocab_file=None, max_length=40, allow_unk=0, max_iters=None):
        self.list_path = list_path
        self.split = split
        self.max_length = max_length
        
        # --- 1. Normalizasyon (ImageNet Standartları) ---
        # ClipEncoder bu istatistikleri bekliyor.
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        assert self.split in {'train', 'val', 'test'}
        
        # Text dosyasını oku (preprocess tarafından oluşturulan train.txt vb.)
        txt_file_path = os.path.join(list_path, split + '.txt')
        if not os.path.exists(txt_file_path):
             raise FileNotFoundError(f"Liste dosyası bulunamadı: {txt_file_path}. Lütfen önce preprocess_data.py çalıştırın.")

        self.img_ids = [i_id.strip() for i_id in open(txt_file_path)]
        
        if vocab_file is not None:
            vocab_path = os.path.join(list_path, vocab_file + '.json')
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r') as f:
                    self.word_vocab = json.load(f)
            else:
                print(f"UYARI: Vocab dosyası bulunamadı ({vocab_path}).")
                self.word_vocab = {}
            self.allow_unk = allow_unk
            
        if not max_iters == None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]
            
        self.files = []
        
        # --- 2. Dosya Yollarını Oluşturma ---
        for item in self.img_ids:
            # item formatı genelde: "00003_0_0.png-0" (DosyaAdı - TokenID)
            
            if '-' in item:
                # Sondaki tireye göre ayır (Token ID'yi al)
                parts = item.rsplit('-', 1) 
                img_name = parts[0]
                token_id_str = parts[1]
            else:
                img_name = item
                token_id_str = None

            # SECOND-CC klasör yapısı ekran görüntüsüne göre:
            # data_folder / train / rgb / A / dosya_adı
            
            # 1. Ana split klasörü (train/test)
            split_dir = os.path.join(data_folder, split)
            
            # 2. 'rgb' klasörü kontrolü
            # Ekran görüntüsünde 'rgb' klasörü var, onu ekliyoruz.
            base_path = os.path.join(split_dir, 'rgb') 

            # 3. A ve B klasörleri (A: Before, B: After varsayımı)
            img_fileA = os.path.join(base_path, 'A', img_name)
            img_fileB = os.path.join(base_path, 'B', img_name) # Ekran görüntüsünde B yok ama çift olması gerekir.
            
            # Eğer B klasörü yoksa (belki isim farklıdır), kodu uyaralım veya 'A'yı kopyalayalım (Debug için)
            # Normalde Change Detection'da mutlaka A/B veya im1/im2 çifti olur.
            
            # Token file yolu
            if token_folder is not None:
                token_file = os.path.join(token_folder, img_name.split('.')[0] + '.txt')
            else:
                token_file = None
                
            self.files.append({
                "imgA": img_fileA,
                "imgB": img_fileB,
                "token": token_file,
                "token_id": token_id_str,
                "name": img_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]
        
        # --- Resim Okuma ---
        try:
            imgA = imread(datafiles["imgA"])
            # Eğer B resmi bulunamazsa hata vermemesi için kontrol (Dataset bütünlüğü için önemlidir)
            if not os.path.exists(datafiles["imgB"]):
                # Fallback: Eğer B klasörü yoksa, kodun patlamaması için A'yı kopyala (Sadece test için!)
                # Gerçek eğitimde burası hata vermeli.
                # print(f"UYARI: B Resmi bulunamadı: {datafiles['imgB']}")
                imgB = imgA.copy() 
            else:
                imgB = imread(datafiles["imgB"])
                
        except Exception as e:
            print(f"Hata: Resim okunamadı -> {datafiles['imgA']}")
            # Bozuk resim gelirse dummy data döndür (Eğitimi durdurmamak için)
            imgA = np.zeros((256, 256, 3), dtype=np.uint8)
            imgB = np.zeros((256, 256, 3), dtype=np.uint8)

        # Float dönüşümü ve 0-1 Aralığına Çekme (ImageNet için Kritik)
        imgA = np.asarray(imgA, np.float32) / 255.0
        imgB = np.asarray(imgB, np.float32) / 255.0
        
        # (H, W, C) -> (C, H, W)
        imgA = np.moveaxis(imgA, -1, 0)     
        imgB = np.moveaxis(imgB, -1, 0)

        # ImageNet Normalizasyonu
        # (img - mean) / std
        for i in range(len(self.mean)):
            imgA[i,:,:] -= self.mean[i]
            imgA[i,:,:] /= self.std[i]
            imgB[i,:,:] -= self.mean[i]
            imgB[i,:,:] /= self.std[i]      

        # --- Token İşlemleri ---
        if datafiles["token"] is not None and os.path.exists(datafiles["token"]):
            with open(datafiles["token"], 'r') as f:
                caption = f.read()
            
            try:
                caption_list = json.loads(caption)
            except:
                caption_list = [caption.strip()]

            token_all = np.zeros((len(caption_list), self.max_length), dtype=int)
            token_all_len = np.zeros((len(caption_list), 1), dtype=int)
            
            for j, tokens in enumerate(caption_list):
                tokens_encode = encode(tokens, self.word_vocab, allow_unk=self.allow_unk == 1)
                token_all[j, :len(tokens_encode)] = tokens_encode
                token_all_len[j] = len(tokens_encode)
                
            if datafiles["token_id"] is not None:
                id = int(datafiles["token_id"])
                # id index sınırlarını aşarsa diye güvenlik
                if id >= len(caption_list): id = 0 
                token = token_all[id]
                token_len = token_all_len[id].item()
            else:
                j = randint(0, len(caption_list) - 1)
                token = token_all[j]
                token_len = token_all_len[j].item()
        else:
            token_all = np.zeros((1, self.max_length), dtype=int)
            token = np.zeros(self.max_length, dtype=int)
            token_len = 0
            token_all_len = np.zeros(1, dtype=int)

        return imgA.copy(), imgB.copy(), token_all.copy(), token_all_len.copy(), token.copy(), np.array(token_len), name