import time
import os
import numpy as np
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json
import torch.nn.functional as F
import types

from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from data.SECOND_CC.SECONDCC import SECONDCCDataset
from data.Dubai_CC.DubaiCC import DubaiCCDataset
from model.model_encoder import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils import *

def count_parameters(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name}: {total_params:,} toplam parametre | {trainable_params:,} eğitilebilir parametre")
    return trainable_params

# --- SCST İÇİN GEREKLİ YENİ FONKSİYONLAR ---
def sample_scst_wrapper(self, x1, x2, sample_max_len=40):
    """
    DecoderTransformer için SCST Sampling Metodu (Monkey-Patch).
    Greedy yerine Multinomial sampling yapar ve log_prob'ları döndürür.
    """
    # 1. Feature Hazırlığı (Forward metodundan alındı)
    x_sam = self.cos(x1, x2)
    x = torch.cat([x1, x2], dim = 1) + x_sam.unsqueeze(1) 
    x = self.LN(self.Conv1(x))

    batch, channel = x.size(0), x.size(1)
    x = x.view(batch, channel, -1).permute(2, 0, 1) # (hw, batch, feature_dim)

    # 2. Hazırlık
    device = x.device
    tgt = torch.zeros(batch, sample_max_len).to(torch.int64).to(device)
    tgt[:, 0] = self.word_vocab['<START>']
    
    seqs = []
    log_probs = []
    
    # Mevcut token (Batch,)
    curr_token = tgt[:, 0] 
    
    # Maske (Triangular)
    mask = torch.triu(torch.ones(sample_max_len, sample_max_len) * float('-inf'), diagonal=1).to(device)
    
    # Loop
    for step in range(sample_max_len - 1):
        # Mevcut target (Batch, Step+1)
        curr_tgt = tgt[:, :step+1]
        
        # Padding mask
        tgt_pad_mask = (curr_tgt == self.word_vocab['<NULL>'])
        
        # Embedding
        word_emb = self.vocab_embedding(curr_tgt) # (Batch, Len, Dim)
        word_emb = word_emb.transpose(1, 0) # (Len, Batch, Dim)
        word_emb = self.position_encoding(word_emb)
        
        # Transformer Forward
        # Maskeyi o anki uzunluğa göre kes
        curr_mask = mask[:step+1, :step+1]
        
        pred = self.transformer(word_emb, x, tgt_mask=curr_mask, tgt_key_padding_mask=tgt_pad_mask)
        
        # Son tokenin çıktısını al
        last_output = pred[-1, :, :] # (Batch, Dim)
        last_output = self.dropout(last_output)
        
        # Logits
        logits = self.wdc(last_output) # (Batch, Vocab)
        
        # --- MULTINOMIAL SAMPLING ---
        probs = F.softmax(logits, dim=-1)
        token = probs.multinomial(1).squeeze(1) # (Batch,)
        
        # Log Prob'u kaydet (Gradient için gerekli)
        log_prob = F.log_softmax(logits, dim=-1)
        token_log_prob = log_prob.gather(1, token.unsqueeze(1)).squeeze(1) # (Batch,)
        
        # Kaydet
        seqs.append(token)
        log_probs.append(token_log_prob)
        
        # Bir sonraki inputu hazırla
        tgt[:, step+1] = token
        
        # Eğer tüm batch <END> ürettiyse erken durdurulabilir (Opsiyonel, burada basitlik için geçiyoruz)

    # Listeleri Tensor yap
    seqs = torch.stack(seqs, dim=1) # (Batch, Len)
    log_probs = torch.stack(log_probs, dim=1) # (Batch, Len)
    
    return seqs, log_probs

def get_self_critical_reward(model, sample_seqs, greedy_seqs, gt_tokens, word_vocab):
    """
    SCST için Reward Hesaplama (CIDEr kullanır).
    """
    batch_size = sample_seqs.size(0)
    res = {}
    gts = {}
    
    # Index -> Kelime dönüşümü
    vocab_inv = {v: k for k, v in word_vocab.items()}
    
    # Helper: Tensor -> String List
    def decode_to_str(seq_tensor):
        result = []
        for i in range(seq_tensor.size(0)):
            words = []
            for idx in seq_tensor[i]:
                w_idx = idx.item()
                if w_idx == word_vocab['<END>']: break
                if w_idx not in [word_vocab['<START>'], word_vocab['<NULL>']]:
                    words.append(vocab_inv.get(w_idx, 'UNK'))
            result.append(words) # Liste döndür (utils.py formatına uygun olması için)
        return result

    # 1. Sampled Captions (Hypothesis)
    sample_res = decode_to_str(sample_seqs)
    
    # 2. Greedy Captions (Baseline)
    greedy_res = decode_to_str(greedy_seqs)
    
    # 3. Ground Truths
    # gt_tokens list of lists formatında geliyor zaten
    gt_res = []
    for i in range(len(gt_tokens)):
        # <START>, <END>, <NULL> temizle
        clean_gt = [w for w in gt_tokens[i] if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
        # Token ID -> Word String dönüşümü (Eğer gt_tokens ID ise)
        # get_eval_score ID listesi kabul ediyorsa stringe çevirmeye gerek yok.
        # utils.get_eval_score implementation'ına bağlı. Genelde ID listesi kabul eder.
        gt_res.append([clean_gt]) # Referanslar liste içinde liste ister

    # CIDEr Skorlarını Hesapla
    # Not: get_eval_score fonksiyonunuzun yapısına göre burası değişebilir.
    # Genellikle get_eval_score(ref, hypo) şeklinde çalışır.
    
    # Sample için Skor
    # Her bir örnek için ayrı ayrı skor hesaplamamız lazım ama get_eval_score genelde tüm corpus için çalışır.
    # SCST için her cümle için ayrı skora ihtiyacımız var. 
    # Hızlı olması için batch halinde verip array dönmesi gerekir ama standart get_eval_score ortalama döner.
    # BURADA BASİT BİR FOR LOOP KULLANACAĞIZ (YAVAŞ OLABİLİR AMA GÜVENLİ)
    
    rewards = np.zeros(batch_size)
    baselines = np.zeros(batch_size)
    
    for i in range(batch_size):
        # Tekil hesaplama (CIDEr'ı tercih et)
        # Eğer utils.py CIDEr scoru tekil döndürmüyorsa tüm metrikleri hesaplar.
        
        # Referanslar (ID listesi)
        ref = [gt_res[i][0]] # [[1, 4, 5...]]
        
        # Sample (ID listesi)
        hyp_sample = sample_res[i]
        score_s = get_eval_score([ref], [hyp_sample]) # utils.py fonksiyonu
        rewards[i] = score_s['CIDEr']
        
        # Greedy (ID listesi)
        hyp_greedy = greedy_res[i]
        score_g = get_eval_score([ref], [hyp_greedy])
        baselines[i] = score_g['CIDEr']
        
    return torch.from_numpy(rewards).float().cuda(), torch.from_numpy(baselines).float().cuda()
# ------------------------------------------

import matplotlib.pyplot as plt

def visualize_results(img1_tensor, img2_tensor, mask_tensor, output_path="result.png"):
    if img1_tensor.dim() == 4: img1_use = img1_tensor[0]
    else: img1_use = img1_tensor
    if img2_tensor.dim() == 4: img2_use = img2_tensor[0]
    else: img2_use = img2_tensor
    if mask_tensor.dim() == 4: mask_use = mask_tensor[0]
    else: mask_use = mask_tensor
    target_h, target_w = img1_use.shape[1], img1_use.shape[2]
    mask_resized = F.interpolate(mask_use.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False)
    img1_np = img1_use.detach().permute(1, 2, 0).cpu().numpy()
    img2_np = img2_use.detach().permute(1, 2, 0).cpu().numpy()
    mask_np = mask_resized.squeeze().detach().cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img1_np = np.clip((img1_np * std) + mean, 0, 1)
    img2_np = np.clip((img2_np * std) + mean, 0, 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(img1_np); plt.title("Before"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(img2_np); plt.title("After"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(mask_np, cmap='jet', vmin=0, vmax=1); plt.title("Mask"); plt.axis('off')
    plt.tight_layout(); plt.savefig(output_path, bbox_inches='tight'); plt.close()

def main(args):
    args.data_folder = './data/' + args.data_name + '/images/'
    args.token_folder = './data/' + args.data_name + '/tokens/'
    args.list_path = './data/' + args.data_name + '/'

    earlyStop = 0
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)  
    if os.path.exists(args.savepath)==False:
        os.makedirs(args.savepath)
    
    best_epoch  = 0
    best_avg    = 0
    start_epoch = 0
    
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)

    # Initialize / load checkpoint
          
    encoder = Encoder(args.network)   
    encoder.fine_tune(args.fine_tune_encoder)     
    encoder_optimizer = torch.optim.Adam(params=encoder.parameters(), lr=args.encoder_lr) if args.fine_tune_encoder else None
    
    encoder_trans = AttentiveEncoder(n_layers =args.n_layers, feature_size=[args.feat_size, args.feat_size, args.encoder_dim], 
                                        heads=args.n_heads, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, dropout=args.dropout, network=args.network)
    encoder_trans_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_trans.parameters()), lr=args.encoder_lr)
    
    decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_vocab), max_lengths=args.max_length, word_vocab=word_vocab, n_head=args.n_heads,
                                n_layers= args.decoder_n_layers, dropout=args.dropout)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=args.decoder_lr)
   
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        best_avg = checkpoint.get('avg_score', 0) # Eski checkpointlerde yoksa 0
        decoder = checkpoint['decoder_dict']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder_trans = checkpoint['encoder_trans_dict']
        encoder_trans_optimizer = checkpoint['encoder_trans_optimizer']
        encoder = checkpoint['encoder_dict']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if args.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.encoder_lr)

    # --- MONKEY PATCHING ---
    # Decoder'a sample_scst_wrapper metodunu ekliyoruz
    decoder.sample_scst = types.MethodType(sample_scst_wrapper, decoder)
    # -----------------------

    # Move to GPU
    encoder = encoder.cuda()
    encoder_trans = encoder_trans.cuda()
    decoder = decoder.cuda()
    
    criterion_ce = torch.nn.CrossEntropyLoss().cuda()

    # Parametre sayımı
    count_parameters(encoder, "Encoder")
    count_parameters(encoder_trans, "AttentiveEncoder")
    count_parameters(decoder, "Decoder")

    # Dataloader seçimi
    DatasetClass = None
    if args.data_name == 'LEVIR_CC': DatasetClass = LEVIRCCDataset
    elif args.data_name == 'Dubai_CC': DatasetClass = DubaiCCDataset
    elif args.data_name == 'SECOND_CC': DatasetClass = SECONDCCDataset
    
    train_loader = data.DataLoader(
        DatasetClass(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
        batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = data.DataLoader(
        DatasetClass(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
        batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # Schedulers
    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=5, gamma=0.5) if args.fine_tune_encoder else None
    encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_trans_optimizer, step_size=5, gamma=0.5)
    decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=5, gamma=0.5)
    
    l_resize = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    
    hist = np.zeros((args.num_epochs * len(train_loader), 3))
    index_i = 0
    
    print(f"Eğitim başlıyor... SCST Modu Başlangıç Epoch: {args.scst_start_epoch}")

    for epoch in range(start_epoch, args.num_epochs):        
        # SCST Modu Kontrolü
        scst_mode = (epoch >= args.scst_start_epoch)
        if scst_mode:
            print(f"--- EPOCH {epoch}: SCST (Self-Critical) LOSS AKTİF ---")
        else:
            print(f"--- EPOCH {epoch}: Cross-Entropy (XE) LOSS AKTİF ---")

        for id, (imgA, imgB, _, _, token, token_len, _) in enumerate(train_loader):
            start_time = time.time()
            
            # Mod ayarları
            # SCST için de train modunda kalmalı (Dropout vb. için)
            decoder.train()
            encoder.eval() 
            encoder_trans.train()
            
            decoder_optimizer.zero_grad()
            encoder_trans_optimizer.zero_grad()
            if encoder_optimizer is not None: encoder_optimizer.zero_grad()

            imgA = imgA.cuda()
            imgB = imgB.cuda()
            if args.data_name == 'Dubai_CC':
                imgA = l_resize(imgA)
                imgB = l_resize(imgB)
                
            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()
            
            # Encoder Forward
            feat1, feat2, mask = encoder(imgA, imgB)
            feat1, feat2 = encoder_trans(feat1, feat2, mask)
            
            loss = 0
            
            if not scst_mode:
                # --- CROSS ENTROPY LOSS (Klasik Eğitim) ---
                scores, caps_sorted, decode_lengths, sort_ind = decoder(feat1, feat2, token, token_len)
                targets = caps_sorted[:, 1:]
                scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
                loss = criterion_ce(scores_packed, targets_packed)
                acc = accuracy(scores_packed, targets_packed, 5)
            else:
                # --- SCST LOSS (RL Eğitim) ---
                
                # 1. Greedy Search (Baseline) - Gradientsiz
                with torch.no_grad():
                    # decoder.sample fonksiyonu greedy (k=1) döndürür
                    greedy_seqs = decoder.sample(feat1, feat2, k=1) 
                    if isinstance(greedy_seqs, list): # sample fonksiyonu list dönerse tensor yap
                        greedy_seqs = torch.LongTensor(greedy_seqs).cuda()
                        if greedy_seqs.dim() == 1: greedy_seqs = greedy_seqs.unsqueeze(0)

                # 2. Monte-Carlo Sampling (Policy) - Gradientli
                # Yeni eklenen sample_scst fonksiyonunu kullanıyoruz
                sample_seqs, sample_log_probs = decoder.sample_scst(feat1, feat2, sample_max_len=args.max_length)
                
                # 3. Reward Hesaplama
                # token.tolist() ground truth referanslarıdır
                rewards, baselines = get_self_critical_reward(decoder, sample_seqs, greedy_seqs, token.tolist(), word_vocab)
                
                # 4. Loss Hesaplama
                # loss = - (reward - baseline) * log_prob
                # log_prob shape: (Batch, Len)
                # reward shape: (Batch,) -> (Batch, 1) yapmalıyız
                
                advantage = rewards - baselines
                
                # Maskeleme: <END> token sonrası veya padding için log_prob'ları 0 yapmalıyız
                # Basitçe sample_seqs üzerinden maske oluşturabiliriz
                pad_mask = (sample_seqs != word_vocab['<NULL>']).float()
                
                # Ortalama loss
                loss = - (advantage.unsqueeze(1) * sample_log_probs * pad_mask).sum() / batch_size
                
                acc = 0 # SCST'de accuracy anlamsızdır, reward takip edilir

            # Back prop
            loss.backward()
            
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(encoder_trans.parameters(), args.grad_clip)
                if encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(encoder.parameters(), args.grad_clip)

            decoder_optimizer.step()
            encoder_trans_optimizer.step()
            if encoder_optimizer is not None: encoder_optimizer.step()

            hist[index_i,0] = time.time() - start_time       
            hist[index_i,1] = loss.item() 
            hist[index_i,2] = acc
            index_i += 1   
            
            if index_i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time: {3:.3f}\t'
                    'Loss: {4:.4f}\t'
                    'Acc/Reward: {5:.3f}'.format(epoch, index_i %len(train_loader), len(train_loader),
                                            np.mean(hist[index_i-args.print_freq:index_i-1,0])*args.print_freq,
                                            np.mean(hist[index_i-args.print_freq:index_i-1,1]),
                                            np.mean(hist[index_i-args.print_freq:index_i-1,2])))

        # --- VALIDATION ---
        decoder.eval()
        encoder_trans.eval()
        if encoder is not None: encoder.eval()

        val_start_time = time.time()
        references = list()
        hypotheses = list()
        
        with torch.no_grad():
            for ind, (imgA, imgB, token_all, _, _, _, _) in enumerate(val_loader):
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                if args.data_name == 'Dubai_CC':
                    imgA = l_resize(imgA)
                    imgB = l_resize(imgB)
                token_all = token_all.squeeze(0).cuda()
                
                if encoder is not None:
                    feat1, feat2, mask = encoder(imgA, imgB)
                feat1, feat2 = encoder_trans(feat1, feat2, mask)
                seq = decoder.sample(feat1, feat2, k=1)

                img_token = token_all.tolist()
                img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}], img_token))
                references.append(img_tokens)

                pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                hypotheses.append(pred_seq)

        val_time = time.time() - val_start_time
        score_dict = get_eval_score(references, hypotheses)
        Bleu_4 = score_dict['Bleu_4']
        Cider = score_dict['CIDEr']
        Meteor = score_dict['METEOR']
        Rouge = score_dict['ROUGE_L']
        Avg = (Bleu_4 + Meteor + Rouge + Cider) / 4
        
        print('Validation:\n' 'Time: {0:.3f}\t' 'AVG: {1:.4f}\t' 'CIDEr: {2:.4f}\t' 'Bleu-4: {3:.4f}'.format(val_time, Avg, Cider, Bleu_4))
        
        # Scheduler Step
        decoder_lr_scheduler.step()
        encoder_trans_lr_scheduler.step()
        if encoder_lr_scheduler is not None: encoder_lr_scheduler.step()
        
        # Save Best
        if Avg > best_avg:
            best_avg = Avg
            print(f'New Best Model (AVG: {best_avg:.4f}) Saved!')
            state = {
                'epoch': epoch,
                'avg_score': best_avg,
                'bleu-4': Bleu_4,
                'encoder_dict': encoder.state_dict(), 
                'encoder_trans_dict': encoder_trans.state_dict(),   
                'decoder_dict': decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer,
                'encoder_trans_optimizer': encoder_trans_optimizer,
                'decoder_optimizer': decoder_optimizer
            }                     
            model_name = f"{args.data_name}_b{args.train_batchsize}_{args.network}_SCST.pth"
            torch.save(state, os.path.join(args.savepath, model_name))
            earlyStop = 0
        else:
            earlyStop += 1
            print(f"No improvement since {earlyStop} epochs.")
        
        if earlyStop == args.early_stop:
            print("Early Stopping.")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCST Training')

    # Args
    parser.add_argument('--data_folder', default='./data/LEVIR_CC/images')
    parser.add_argument('--list_path', default='./data/LEVIR_CC/')
    parser.add_argument('--token_folder', default='./data/LEVIR_CC/tokens/')
    parser.add_argument('--vocab_file', default='vocab')
    parser.add_argument('--max_length', type=int, default=41)
    parser.add_argument('--allow_unk', type=int, default=1)
    parser.add_argument('--data_name', default="LEVIR_CC")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--checkpoint', default=None, help='Path to pre-trained XE checkpoint (REQUIRED for SCST)')
    parser.add_argument('--print_freq',type=int, default=50)
    
    # Training Params
    parser.add_argument('--fine_tune_encoder', type=bool, default=True)    
    parser.add_argument('--train_batchsize', type=int, default=32) # SCST'de batch size önemli
    parser.add_argument('--network', default='resnet101')
    parser.add_argument('--encoder_dim',default=1024)
    parser.add_argument('--feat_size', default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    
    # LR ayarları (SCST için daha düşük LR önerilir)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--decoder_lr', type=float, default=5e-5)
    
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--val_batchsize', type=int, default=1)
    parser.add_argument('--savepath', default="./models_checkpoint/")
    
    # Transformer Params
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--attention_dim', type=int, default=512)
    parser.add_argument('--feature_dim', type=int, default=512)
    
    # --- SCST PARAMETRELERİ ---
    parser.add_argument('--scst_start_epoch', type=int, default=0, 
                        help='SCST kaybının başlayacağı epoch. Eğer pre-trained model yüklüyorsanız 0 yapın, sıfırdan eğitiyorsanız 20+ yapın.')

    args = parser.parse_args()
    main(args)