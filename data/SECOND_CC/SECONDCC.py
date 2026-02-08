import torch
from torch.utils.data import Dataset
from preprocess_data import encode
import json
import os
import numpy as np
from torch.utils.data import DataLoader
from imageio import imread
from random import randint
import cv2  # OpenCV kütüphanesi eklendi (Resize için)

class SECONDCCDataset(Dataset):
    """
    SECOND Dataset (SECOND-CC-AUG) için PyTorch Dataset sınıfı.
    ClipEncoder ile uyumlu olması için ImageNet normalizasyonunu kullanır.
    Hata düzeltmeleri:
    1. Tüm resimleri zorla 256x256 boyutuna getirir.
    2. token_all çıktısını sabit boyutta (5 caption) tutar.
    """

    def __init__(self, data_folder, list_path, split, token_folder=None, vocab_file=None, max_length=40, allow_unk=0, max_iters=None):
        self.list_path = list_path
        self.split = split
        self.max_length = max_length
        
        # ClipEncoder için ImageNet İstatistikleri
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        assert self.split in {'train', 'val', 'test'}
        
        txt_file_path = os.path.join(list_path, split + '.txt')
        if not os.path.exists(txt_file_path):
             raise FileNotFoundError(f"Liste dosyası bulunamadı: {txt_file_path}")

        self.img_ids = [i_id.strip() for i_id in open(txt_file_path)]
        
        if vocab_file is not None:
            vocab_path = os.path.join(list_path, vocab_file + '.json')
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r') as f:
                    self.word_vocab = json.load(f)
            else:
                self.word_vocab = {}
            self.allow_unk = allow_unk
            
        if not max_iters == None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]
            
        self.files = []
        
        for item in self.img_ids:
            if '-' in item:
                parts = item.rsplit('-', 1) 
                img_name = parts[0]
                token_id_str = parts[1]
            else:
                img_name = item
                token_id_str = None

            split_dir = os.path.join(data_folder, split)
            # Klasör yapısı: .../train/rgb/A/00042.png
            base_path = os.path.join(split_dir, 'rgb') 

            img_fileA = os.path.join(base_path, 'A', img_name)
            img_fileB = os.path.join(base_path, 'B', img_name)
            
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
        
        # --- 1. Resim Okuma ve Resize ---
        try:
            imgA = imread(datafiles["imgA"])
            if not os.path.exists(datafiles["imgB"]):
                imgB = imgA.copy() 
            else:
                imgB = imread(datafiles["imgB"])
        except Exception as e:
            # Hata durumunda siyah resim döndür
            imgA = np.zeros((256, 256, 3), dtype=np.uint8)
            imgB = np.zeros((256, 256, 3), dtype=np.uint8)  

        # --- 2. Token İşlemleri ve Sabitleme ---
        # [ÇÖZÜM 2] Caption sayısını sabitleme (Batch yapabilmek için)
        MAX_CAPTION_COUNT = 5  # Her resim için maksimum 5 cümle varsayıyoruz
        
        # Boş şablonları oluştur (Hepsi 0)
        token_all = np.zeros((MAX_CAPTION_COUNT, self.max_length), dtype=int)
        token_all_len = np.zeros((MAX_CAPTION_COUNT, 1), dtype=int)
        
        if datafiles["token"] is not None and os.path.exists(datafiles["token"]):
            with open(datafiles["token"], 'r') as f:
                caption = f.read()
            try:
                caption_list = json.loads(caption)
            except:
                caption_list = [caption.strip()]

            # Var olan captionları şablona doldur
            # Eğer 5'ten az ise kalanlar 0 (NULL) kalır, sorun olmaz.
            # Eğer 5'ten çok ise ilk 5'i alınır.
            limit = min(len(caption_list), MAX_CAPTION_COUNT)
            
            for j in range(limit):
                tokens_encode = encode(caption_list[j], self.word_vocab, allow_unk=self.allow_unk == 1)
                # Max length kontrolü (Eğer kelime sayısı 40'ı geçerse kırp)
                if len(tokens_encode) > self.max_length:
                     tokens_encode = tokens_encode[:self.max_length]
                     
                token_all[j, :len(tokens_encode)] = tokens_encode
                token_all_len[j] = len(tokens_encode)
                
            if datafiles["token_id"] is not None:
                id = int(datafiles["token_id"])
                if id >= limit: id = 0 
                token = token_all[id]
                token_len = token_all_len[id].item()
            else:
                # Rastgele bir caption seç (Eğitim sırasında token_all yerine token kullanılır)
                # Sadece dolu olanlardan seç (limit'e kadar)
                if limit > 0:
                    j = randint(0, limit - 1)
                    token = token_all[j]
                    token_len = token_all_len[j].item()
                else:
                    token = np.zeros(self.max_length, dtype=int)
                    token_len = 0
        else:
            # Token dosyası yoksa hepsi sıfır kalır
            token = np.zeros(self.max_length, dtype=int)
            token_len = 0

        return imgA.copy(), imgB.copy(), token_all.copy(), token_all_len.copy(), token.copy(), np.array(token_len), name