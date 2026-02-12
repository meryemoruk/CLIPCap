#burası recall'u arttırmak için attention pooling eklemeden önceki dino dosyası. recall 13 civarı gelmişti.

import time
import os
import numpy as np
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import torch.optim as optim
import argparse
import json
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from data.SECOND_CC.SECONDCC import SECONDCCDataset
from data.Dubai_CC.DubaiCC import DubaiCCDataset
from model.model_encoder import Encoder, AttentiveEncoder, ClipMLP, ClipProjector, RSCLIPTextEncoder, ClipLoss, ClipEncoder
from model.model_decoder import DecoderTransformer
from model.model_encoder import DifferenceAttention
from utils import *
import clip

def compute_recall(image_features, text_features, k_vals=[1, 5, 10]):
    logits = image_features @ text_features.t()
    num_samples = logits.shape[0]
    ground_truth = torch.arange(num_samples, device=logits.device)
    
    scores_i2t = logits
    i2t_results = {}
    for k in k_vals:
        _, topk_indices = scores_i2t.topk(k, dim=1)
        correct = topk_indices.eq(ground_truth.view(-1, 1)).any(dim=1)
        recall = correct.float().mean().item() * 100
        i2t_results[f"R@{k}"] = recall

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main(args):
    args.data_folder = './data/' + args.data_name + '/images/'
    args.token_folder = './data/' + args.data_name + '/tokens/'
    args.list_path = './data/' + args.data_name + '/'

    earlyStop = 0

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)  
    if os.path.exists(args.savepath)==False:
        os.makedirs(args.savepath)

    start_epoch = 0
    best_score = 0 # Loss yerine Recall Score takibi

    # Initialize / load checkpoint
    if args.checkpoint is None:      
        encoder = ClipEncoder() 
        clip_mlp = ClipMLP(dropout=args.dropout)  
        clip_projector = ClipProjector()
        text_encoder = RSCLIPTextEncoder()
    
    # Move to GPU
    encoder = encoder.cuda()
    clip_mlp = clip_mlp.cuda()
    clip_projector = clip_projector.cuda()
    text_encoder = text_encoder.cuda()
    
    criterion = ClipLoss().cuda()
    diff_attn = DifferenceAttention(dim=768, heads=8).cuda()
    optimizer = torch.optim.AdamW(list(clip_mlp.parameters()) + list(clip_projector.parameters()) +list(diff_attn.parameters()) + [criterion.logit_scale], lr=args.lr)

    # PARAMETRE İSTATİSTİKLERİ
    print("-" * 50)
    total_trainable = 0
    total_trainable += count_parameters(encoder, "Encoder (Backbone)")
    total_trainable += count_parameters(clip_mlp, "MLP")
    total_trainable += count_parameters(clip_projector, "Projection")
    total_trainable += count_parameters(text_encoder, "Text Encoder")
    print("-" * 50)
    print(f"TOPLAM EĞİTİLEBİLİR PARAMETRE: {total_trainable:,}")
    print("-" * 50)

    # Dataloaders
    if args.data_name == 'LEVIR_CC':
        DatasetClass = LEVIRCCDataset
    elif args.data_name == 'Dubai_CC':
        DatasetClass = DubaiCCDataset
    elif args.data_name == 'SECOND_CC':
        DatasetClass = SECONDCCDataset
    
    train_loader = data.DataLoader(
        DatasetClass(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
        batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = data.DataLoader(
        DatasetClass(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
        batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # --- SCHEDULER DÜZELTME ---
    # ReduceLROnPlateau: Skor (mode='max') artmazsa LR düşürür.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    l_resizeA = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    l_resizeB = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)

    # Epochs
    for epoch in range(start_epoch, args.num_epochs):        
        encoder.eval()
        clip_mlp.train()
        clip_projector.train()
        text_encoder.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        
        end = time.time()

        # Batches - raw_captions EKLENDİ
        for id, (imgA, imgB, _, _, token, token_len, raw_captions) in enumerate(train_loader):
            
            optimizer.zero_grad()

            imgA = imgA.cuda()
            imgB = imgB.cuda()
            if args.data_name == 'Dubai_CC':
                imgA = l_resizeA(imgA)
                imgB = l_resizeB(imgB)
            
            # Forward prop.
            img_feat_A = encoder(imgA)
            img_feat_B = encoder(imgB)

            img_feat_A = clip_mlp(img_feat_A)
            img_feat_B = clip_mlp(img_feat_B)

            b, h, w, c = img_feat_A.shape
            seq_A = img_feat_A.reshape(b, h*w, c) # [B, 256, 768]
            seq_B = img_feat_B.reshape(b, h*w, c)

            diff_seq = diff_attn(seq_A, seq_B)
            diff_vec = diff_seq.mean(dim=1)

            diff_feat = clip_projector(diff_vec)
            diff_feat = diff_feat / diff_feat.norm(dim=-1, keepdim=True)    

            # # Pooling (Eğer projection içinde yoksa burada yapılmalı)
            # # DINO [B, 16, 16, 768] -> Mean -> [B, 768]
            # vec_A = img_feat_A.mean(dim=(1, 2))
            # vec_B = img_feat_B.mean(dim=(1, 2))

            # proj_A = clip_projector(vec_A)
            # proj_B = clip_projector(vec_B)

            # diff_feat = proj_B - proj_A
            # diff_feat = diff_feat / diff_feat.norm(dim=-1, keepdim=True)
            
            with torch.no_grad():
                if isinstance(raw_captions, tuple): raw_captions = list(raw_captions)
                text_inputs = clip.tokenize(raw_captions, truncate=True).cuda()
                # .float() EKLENDİ (RuntimeError Çözümü)
                text_embeds = text_encoder(text_inputs).float()
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            loss = criterion(diff_feat, text_embeds)
            
            loss.backward()
            optimizer.step()

            # İstatistikleri güncelle
            losses.update(loss.item(), imgA.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            # --- BİLGİSEL PRINT (TRAINING) ---
            if id % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'LR {lr:.6f}'.format(
                       epoch, id, len(train_loader), batch_time=batch_time,
                       loss=losses, lr=optimizer.param_groups[0]['lr']))

        # --- VALIDATION LOOP ---
        print(f"\nEpoch {epoch} Validation Başlıyor...")
        encoder.eval()
        clip_mlp.eval()
        clip_projector.eval()
        text_encoder.eval()
        
        all_image_feats = []
        all_text_feats = []
        
        with torch.no_grad():
            for ind, (imgA, imgB, _, _, _, _, raw_captions) in enumerate(val_loader):
                
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                
                # 1. Encoder'dan geçir
                feat_A = clip_mlp(encoder(imgA)) # [B, 16, 16, 768]
                feat_B = clip_mlp(encoder(imgB)) # [B, 16, 16, 768]
                
                # 2. Sequence formatına çevir [B, 256, 768]
                b, h, w, c = feat_A.shape
                seq_A = feat_A.reshape(b, h*w, c)
                seq_B = feat_B.reshape(b, h*w, c)
                
                # 3. Difference Attention Uygula (Validation'da da bunu kullanmalı!)
                diff_seq = diff_attn(seq_A, seq_B)
                
                # 4. Ortalama al ve Project et
                diff_vec = diff_seq.mean(dim=1) # [B, 768]
                diff_feat = clip_projector(diff_vec)
                
                # 5. Normalize et
                diff_feat = diff_feat / diff_feat.norm(dim=-1, keepdim=True)
                all_image_feats.append(diff_feat)
                
                if isinstance(raw_captions, tuple): raw_captions = list(raw_captions)
                text_inputs = clip.tokenize(raw_captions, truncate=True).cuda()
                text_embeds = text_encoder(text_inputs).float() 
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                all_text_feats.append(text_embeds)

            all_image_feats = torch.cat(all_image_feats, dim=0)
            all_text_feats = torch.cat(all_text_feats, dim=0)
            
            i2t_res, t2i_res = compute_recall(all_image_feats, all_text_feats)
            
            print("--- Image to Text Retrieval ---")
            print(f"R@1: {i2t_res['R@1']:.2f} | R@5: {i2t_res['R@5']:.2f} | R@10: {i2t_res['R@10']:.2f}")
            print("--- Text to Image Retrieval ---")
            print(f"R@1: {t2i_res['R@1']:.2f} | R@5: {t2i_res['R@5']:.2f} | R@10: {t2i_res['R@10']:.2f}")

            # Toplam Skor (Recall Sum)
            current_score = i2t_res['R@1'] + i2t_res['R@5'] + t2i_res['R@1'] + t2i_res['R@5']
            
            # SCHEDULER STEP (Validation Sonucuna Göre)
            scheduler.step() # İçini boş bırak, epoch'u kendi saysın.
            
            # MODEL KAYDETME
            if current_score > best_score:
                print(f'New Best Model! (Score: {best_score:.2f} -> {current_score:.2f})')
                best_score = current_score
                earlyStop = 0
                
                state = {
                    'epoch': epoch,
                    'mlp_state_dict': clip_mlp.state_dict(),
                    'projection_state_dict': clip_projector.state_dict(),
                    'best_score': best_score,
                    'optimizer': optimizer.state_dict()
                }
                save_name = os.path.join(args.savepath, f"{args.data_name}_Best_Recall.pth")
                torch.save(state, save_name)
                print(f"Model kaydedildi: {save_name}")
            else:
                earlyStop += 1
                print(f"No improvement since {earlyStop} epochs.")
    
        if(earlyStop == args.early_stop):
            print("Early stopping triggered. Training finished.")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')
    parser.add_argument('--data_folder', default='./data/LEVIR_CC/images',help='folder with data files')
    parser.add_argument('--list_path', default='./data/LEVIR_CC/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_CC/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_CC",help='base name shared by data files.')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--print_freq',type=int, default=20, help='print training stats every __ batches')
    parser.add_argument('--train_batchsize', type=int, default=64, help='batch_size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--workers', type=int, default=4, help='num workers')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop patience')
    parser.add_argument('--val_batchsize', type=int, default=64, help='batch_size for validation')
    parser.add_argument('--savepath', default="./models_checkpoint/")
    args = parser.parse_args()
    main(args)
