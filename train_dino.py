import time
import os
import numpy as np
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import torch.optim as optim
import argparse
import json
#import torchvision.transforms as transforms
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from data.SECOND_CC.SECONDCC import SECONDCCDataset
from data.Dubai_CC.DubaiCC import DubaiCCDataset
from model.model_encoder import Encoder, AttentiveEncoder, DinoEncoder, DinoMLP, DinoProjector, RSCLIPTextEncoder, ClipLoss
from model.model_decoder import DecoderTransformer
from utils import *
import clip

def compute_recall(image_features, text_features, k_vals=[1, 5, 10]):
    """
    Image-to-Text ve Text-to-Image Recall değerlerini hesaplar.
    
    Args:
        image_features: [N, D] tensor (Normalize edilmiş)
        text_features:  [N, D] tensor (Normalize edilmiş)
        k_vals: Hangi k değerlerine bakılacağı (Top-1, Top-5, Top-10)
    """
    # 1. Benzerlik Matrisi (Similarity Matrix)
    # N adet resim ile N adet metnin hepsini birbiriyle çarparız.
    # Sonuç: [N, N] matrisi
    # logits[i, j]: i. resim ile j. metin arasındaki benzerlik skoru.
    logits = image_features @ text_features.t()
    
    num_samples = logits.shape[0]
    
    # Ground Truth: Köşegenler (i. resim, i. metin ile eşleşmeli)
    ground_truth = torch.arange(num_samples, device=logits.device)
    
    # --- Image-to-Text Retrieval (I2T) ---
    # Her resim için en benzer metinleri bul
    # values, indices = logits.topk(max(k_vals), dim=1)
    # Daha hızlı olması için argsort kullanabiliriz (büyükten küçüğe)
    scores_i2t = logits
    
    i2t_results = {}
    for k in k_vals:
        # Top-k indices: [N, k]
        _, topk_indices = scores_i2t.topk(k, dim=1)
        
        # Doğru cevap (ground_truth) bu topk_indices içinde var mı?
        # ground_truth.view(-1, 1): [N, 1] yaparız ki yayın (broadcast) yapabilsin
        correct = topk_indices.eq(ground_truth.view(-1, 1)).any(dim=1)
        
        # Doğru olanların yüzdesi
        recall = correct.float().mean().item() * 100
        i2t_results[f"R@{k}"] = recall

    # --- Text-to-Image Retrieval (T2I) ---
    # Her metin için en benzer resmi bul (Transpoz alarak)
    scores_t2i = logits.t()
    
    t2i_results = {}
    for k in k_vals:
        _, topk_indices = scores_t2i.topk(k, dim=1)
        correct = topk_indices.eq(ground_truth.view(-1, 1)).any(dim=1)
        recall = correct.float().mean().item() * 100
        t2i_results[f"R@{k}"] = recall
        
    return i2t_results, t2i_results

def count_parameters(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name}: {total_params:,} toplam parametre | {trainable_params:,} eğitilebilir parametre")
    return trainable_params

def main(args):
    """
    Training and validation.
    """

    args.data_folder = './data/' + args.data_name + '/images/'
    args.token_folder = './data/' + args.data_name + '/tokens/'
    args.list_path = './data/' + args.data_name + '/'

    earlyStop = 0

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)  
    if os.path.exists(args.savepath)==False:
        os.makedirs(args.savepath)

    start_epoch = 0
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    # Initialize / load checkpoint
    if args.checkpoint is None:      
        encoder = DinoEncoder() 
        dino_mlp = DinoMLP(dropout=args.dropout)  
        dino_projection = DinoProjector()
        text_encoder = RSCLIPTextEncoder()
    
    # Move to GPU, if available
    encoder = encoder.cuda()
    dino_mlp = dino_mlp.cuda()
    dino_projection = dino_projection.cuda()
    text_encoder = text_encoder.cuda()
    # Loss function
    criterion = ClipLoss().cuda()
    optimizer = torch.optim.AdamW(list(dino_mlp.parameters()) + list(dino_projection.parameters()) + [criterion.logit_scale], lr=args.lr)

    # ------- PARAMETRE -------
    print("-" * 50)
    print("MODEL PARAMETRE İSTATİSTİKLERİ")
    print("-" * 50)
    
    total_trainable = 0

    total_trainable += count_parameters(encoder, "Encoder (Backbone)")
    total_trainable += count_parameters(dino_mlp, "mlp")
    total_trainable += count_parameters(dino_projection, "projection")
    total_trainable += count_parameters(text_encoder, "text encoder")

    print("-" * 50)
    print(f"TOPLAM EĞİTİLEBİLİR PARAMETRE: {total_trainable:,}")
    print("-" * 50)
    # ------- PARAMETRE -------

    # Custom dataloaders
    if args.data_name == 'LEVIR_CC':
        train_loader = data.DataLoader(
            LEVIRCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = data.DataLoader(
            LEVIRCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    elif args.data_name == 'Dubai_CC':
        train_loader = data.DataLoader(
            DubaiCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = data.DataLoader(
            DubaiCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.data_name == 'SECOND_CC':
        train_loader = data.DataLoader(
            SECONDCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = data.DataLoader(
            SECONDCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    l_resizeA = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    l_resizeB = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    index_i = 0

    # Epochs
    for epoch in range(start_epoch, args.num_epochs):        
        # Batches
        for id, (imgA, imgB, _, _, token, token_len, raw_captions) in enumerate(train_loader):
            #if id == 20:
            #    break
            start_time = time.time()
            encoder.eval()
            dino_mlp.train()
            dino_projection.train()
            text_encoder.eval()

            if optimizer is not None:
                optimizer.zero_grad()

            # Move to GPU, if available
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            if args.data_name == 'Dubai_CC':
                imgA = l_resizeA(imgA)
                imgB = l_resizeB(imgB)
            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()
            
            # Forward prop.
            img_feat_A = encoder(imgA)
            img_feat_B = encoder(imgB)

            img_feat_A = dino_mlp(img_feat_A)
            img_feat_B = dino_mlp(img_feat_B)

            img_feat_A = dino_projection(img_feat_A)
            img_feat_B = dino_projection(img_feat_B)

            diff_feat = img_feat_B - img_feat_A

            # Normalize et (Cosine Similarity için şart!)
            diff_feat = diff_feat / diff_feat.norm(dim=-1, keepdim=True)
            
            # 5. Text Encoding (Tokenize işlemi burada yapılır)
            # Veri setinden gelen ID'leri değil, ham metni CLIP ile tokenize ediyoruz.
            with torch.no_grad():
                # raw_captions bir tuple veya list gelir, listeye çevirin
                if isinstance(raw_captions, tuple): raw_captions = list(raw_captions)
                
                # CLIP Tokenizer
                text_inputs = clip.tokenize(raw_captions, truncate=True).cuda()
                
                # Text Encoder
                text_embeds = text_encoder(text_inputs)
                # Normalize et
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # Calculate loss
            loss = criterion(diff_feat, text_embeds)
    
            # Back prop.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- VALIDATION LOOP ---
        encoder.eval()
        dino_mlp.eval()
        dino_projection.eval()
        text_encoder.eval()

        val_start_time = time.time()
        
        # Tüm datasetin vektörlerini saklayacak listeler
        all_image_feats = []
        all_text_feats = []
        
        with torch.no_grad():
            for ind, (imgA, imgB, _, _, _, _, raw_captions) in enumerate(val_loader):
                
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                
                # --- GÖRSEL İŞLEMLER ---
                fA = dino_projection(dino_mlp(encoder(imgA)).mean(dim=(1,2)))
                fB = dino_projection(dino_mlp(encoder(imgB)).mean(dim=(1,2)))
                
                diff_feat = fB - fA
                diff_feat = diff_feat / diff_feat.norm(dim=-1, keepdim=True)
                
                # Listeye ekle (CPU'ya alarak RAM tasarrufu yapabiliriz, 
                # ama VRAM yetiyorsa GPU'da kalsın hesap hızlı olur)
                all_image_feats.append(diff_feat)
                
                # --- METİN İŞLEMLERİ ---
                if isinstance(raw_captions, tuple): raw_captions = list(raw_captions)
                text_inputs = clip.tokenize(raw_captions, truncate=True).cuda()
                text_embeds = text_encoder(text_inputs)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                all_text_feats.append(text_embeds)

            # --- DÖNGÜ SONU: METRİK HESABI ---
            
            # Listeleri Tensor'a çevir (Concatenate)
            # Sonuç: [Total_Val_Size, 512]
            all_image_feats = torch.cat(all_image_feats, dim=0)
            all_text_feats = torch.cat(all_text_feats, dim=0)
            
            print(f"Validation Feature Size: {all_image_feats.shape}")
            
            # Recall Hesapla
            i2t_res, t2i_res = compute_recall(all_image_feats, all_text_feats)
            
            # Skorları Yazdır
            print("--- Image to Text Retrieval ---")
            print(f"R@1: {i2t_res['R@1']:.2f} | R@5: {i2t_res['R@5']:.2f} | R@10: {i2t_res['R@10']:.2f}")
            
            print("--- Text to Image Retrieval ---")
            print(f"R@1: {t2i_res['R@1']:.2f} | R@5: {t2i_res['R@5']:.2f} | R@10: {t2i_res['R@10']:.2f}")

            # Takip edilecek ana metrik (Genelde R@1 veya R@5 toplamı seçilir)
            current_score = i2t_res['R@1'] + i2t_res['R@5'] + t2i_res['R@1'] + t2i_res['R@5']
            
            # --- MODEL KAYDETME (SCORE BAZLI) ---
            if current_score > best_score:
                print(f'New Best Model! (Score: {best_score:.2f} -> {current_score:.2f})')
                best_score = current_score
                earlyStop = 0
                
                state = {
                    'epoch': epoch,
                    'mlp_state_dict': dino_mlp.state_dict(),
                    'projection_state_dict': dino_projection.state_dict(),
                    'best_score': best_score
                }
                torch.save(state, os.path.join(args.savepath, f"{args.data_name}_Best_Recall.pth"))
            else:
                earlyStop += 1
                print(f"No improvement since {earlyStop} epochs.")
    
        if(earlyStop == args.early_stop):
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')

    # Data parameters
    parser.add_argument('--data_folder', default='./data/LEVIR_CC/images',help='folder with data files')
    parser.add_argument('--list_path', default='./data/LEVIR_CC/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_CC/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_CC",help='base name shared by data files.')

    #parser.add_argument('--data_folder', default='/root/Data/Dubai_CC/DubaiCC500impair/datasetDubaiCCPublic/imgs_tiles/RGB/',help='folder with data files')
    #parser.add_argument('--list_path', default='./data/Dubai_CC/', help='path of the data lists')
    #parser.add_argument('--token_folder', default='./data/Dubai_CC/tokens/', help='folder with token files')
    #parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    #parser.add_argument('--max_length', type=int, default=27, help='path of the data lists')
    #parser.add_argument('--allow_unk', type=int, default=0, help='if unknown token is allowed')
    #parser.add_argument('--data_name', default="Dubai_CC",help='base name shared by data files.')

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--print_freq',type=int, default=50, help='print training/validation stats every __ batches')
    # Training parameters
    parser.add_argument('--train_batchsize', type=int, default=64, help='batch_size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=8, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop')
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="./models_checkpoint/")
    # Model parameters
    args = parser.parse_args()
    main(args)