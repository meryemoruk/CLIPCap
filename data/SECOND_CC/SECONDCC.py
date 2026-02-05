import torch
from torch.utils.data import Dataset
from preprocess_data import encode
import json
import os
import numpy as np
from torch.utils.data import DataLoader
from imageio import imread
from random import *

class SECONDCCDataset(Dataset):
    """
    SECOND Dataset için PyTorch Dataset sınıfı.
    ClipEncoder ile uyumlu olması için ImageNet normalizasyonu kullanır.
    """

    def __init__(self, data_folder, list_path, split, token_folder=None, vocab_file=None, max_length=40, allow_unk=0, max_iters=None):
        self.list_path = list_path
        self.split = split
        self.max_length = max_length
        
        # --- ÖNEMLİ DEĞİŞİKLİK 1: Normalizasyon ---
        # ClipEncoder'ınız "denormalize" işlemi yaparken ImageNet istatistiklerini (self.source_std) kullanıyor.
        # Bu yüzden veriyi veri yükleyici (DataLoader) aşamasında ImageNet standartlarına göre hazırlamalıyız.
        # ImageNet Mean ve Std (0-1 aralığı için)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        assert self.split in {'train', 'val', 'test'}
        
        # List dosyasından isimleri oku (uzantıları temizle)
        self.img_ids = [i_id.strip() for i_id in open(os.path.join(list_path + split + '.txt'))]
        
        if vocab_file is not None:
            with open(os.path.join(list_path + vocab_file + '.json'), 'r') as f:
                self.word_vocab = json.load(f)
            self.allow_unk = allow_unk
            
        if not max_iters == None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]
            
        self.files = []
        
        # --- ÖNEMLİ DEĞİŞİKLİK 2: Dosya Yolları (SECOND Yapısı) ---
        # SECOND veri seti genelde 'A' ve 'B' klasörlerini kullanır.
        # Dosya isimleri genelde '00001.png' şeklindedir.
        
        for name in self.img_ids:
            # İsim formatı kontrolü (gerekirse .png ekle veya split et)
            # Eğer txt dosyasında sadece "12345" yazıyorsa:
            img_name = name if name.endswith('.png') else name + '.png'
            
            # SECOND klasör yapısı varsayımı: data_folder/train/A/xxxxx.png
            if split == 'train':
                 # Train klasör yapısı (Datasetinizi buraya göre düzenleyin)
                 # Örn: .../SECOND/train/A/001.png
                 base_path = os.path.join(data_folder, split, 'rgb')
            else:
                 base_path = os.path.join(data_folder, split, 'rgb')

            img_fileA = os.path.join(base_path, 'A', img_name)
            img_fileB = os.path.join(base_path, 'B', img_name)
            
            # Token işlemleri (LEVIR mantığı ile aynı)
            token_id = None # SECOND'da genelde tek caption olur ama yapı korunabilir
            if token_folder is not None:
                # Token dosya ismi resim ismiyle aynı varsayılır
                token_file = os.path.join(token_folder, img_name.split('.')[0] + '.txt')
            else:
                token_file = None
                
            self.files.append({
                "imgA": img_fileA,
                "imgB": img_fileB,
                "token": token_file,
                "token_id": token_id,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]
        
        try:
            imgA = imread(datafiles["imgA"])
            imgB = imread(datafiles["imgB"])
        except Exception as e:
            print(f"Hata: Resim okunamadı -> {datafiles['imgA']}")
            raise e

        # Float dönüşümü
        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)
        
        # --- ÖNEMLİ DEĞİŞİKLİK 3: 0-1 Aralığına Çekme ---
        # ImageNet istatistikleri 0-1 aralığındaki tensörler içindir.
        # imread 0-255 döndürür, bu yüzden 255'e bölüyoruz.
        imgA /= 255.0
        imgB /= 255.0

        # (H, W, C) -> (C, H, W)
        imgA = np.moveaxis(imgA, -1, 0)     
        imgB = np.moveaxis(imgB, -1, 0)

        # ImageNet Normalizasyonu
        for i in range(len(self.mean)):
            imgA[i,:,:] -= self.mean[i]
            imgA[i,:,:] /= self.std[i]
            imgB[i,:,:] -= self.mean[i]
            imgB[i,:,:] /= self.std[i]      

        # --- Token İşlemleri (Aynı Kalıyor) ---
        if datafiles["token"] is not None and os.path.exists(datafiles["token"]):
            with open(datafiles["token"], 'r') as f:
                caption = f.read()
            
            # JSON formatında mı yoksa düz metin mi kontrolü
            try:
                caption_list = json.loads(caption)
            except:
                # Eğer json değilse (SECOND bazen düz txt olabilir), listeye çevir
                caption_list = [caption.strip()]

            token_all = np.zeros((len(caption_list), self.max_length), dtype=int)
            token_all_len = np.zeros((len(caption_list), 1), dtype=int)
            
            for j, tokens in enumerate(caption_list):
                tokens_encode = encode(tokens, self.word_vocab, allow_unk=self.allow_unk == 1)
                token_all[j, :len(tokens_encode)] = tokens_encode
                token_all_len[j] = len(tokens_encode)
                
            if datafiles["token_id"] is not None:
                id = int(datafiles["token_id"])
                token = token_all[id]
                token_len = token_all_len[id].item()
            else:
                j = randint(0, len(caption_list) - 1)
                token = token_all[j]
                token_len = token_all_len[j].item()
        else:
            # Token dosyası yoksa dummy data
            token_all = np.zeros((1, self.max_length), dtype=int)
            token = np.zeros(self.max_length, dtype=int)
            token_len = 0
            token_all_len = np.zeros(1, dtype=int)

        return imgA.copy(), imgB.copy(), token_all.copy(), token_all_len.copy(), token.copy(), np.array(token_len), name