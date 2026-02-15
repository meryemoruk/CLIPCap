import time
import os
import numpy as np
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json
import torch.nn.functional as F
from eval_func.cider.cider import Cider


from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from data.SECOND_CC.SECONDCC import SECONDCCDataset
from data.Dubai_CC.DubaiCC import DubaiCCDataset
from model.model_encoder import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer_scst
from utils import *

def count_parameters(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name}: {total_params:,} toplam parametre | {trainable_params:,} eğitilebilir parametre")
    return trainable_params

# --- NOT: Wrapper fonksiyonları (sample_scst_wrapper, sample_greedy_wrapper) SİLİNDİ ---
# Çünkü artık DecoderTransformer sınıfının içinde yer alıyorlar.

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
    gt_res = []
    for i in range(len(gt_tokens)):
        # <START>, <END>, <NULL> temizle
        clean_gt = [w for w in gt_tokens[i] if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
        gt_res.append([clean_gt]) 

    rewards = np.zeros(batch_size)
    baselines = np.zeros(batch_size)
    
    for i in range(batch_size):
        # Tekil hesaplama
        ref = [gt_res[i][0]]

        # Sample
        hyp_sample = sample_res[i]
        rewards[i] = Cider.compute_score(ref, hyp_sample)
        
        # Greedy
        hyp_greedy = greedy_res[i]
        baselines[i] = Cider.compute_score(ref, hyp_greedy)
        
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
    trainable_params = filter(lambda p: p.requires_grad, encoder.parameters())

    encoder_optimizer = torch.optim.Adam(params=trainable_params, lr=args.encoder_lr)
    encoder_trans = AttentiveEncoder(n_layers =args.n_layers, feature_size=[args.feat_size, args.feat_size, args.encoder_dim], 
                                        heads=args.n_heads, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, dropout=args.dropout, network=args.network)
    
    encoder_trans_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, encoder_trans.parameters()),
                                        lr=args.encoder_lr, weight_decay=1e-4)
    decoder = DecoderTransformer_scst(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_vocab), max_lengths=args.max_length, word_vocab=word_vocab, n_head=args.n_heads,
                                n_layers= args.decoder_n_layers, dropout=args.dropout)
    
    decoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, decoder.parameters()), 
                                            lr=args.decoder_lr, weight_decay=1e-4)
   
    if args.checkpoint is not None:
        print(f"Checkpoint yükleniyor: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        best_avg = checkpoint.get('avg_score', 0)
        
        # Ağırlıkları var olan modellere yükle
        decoder.load_state_dict(checkpoint['decoder_dict'])
        encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'])
        encoder.load_state_dict(checkpoint['encoder_dict'])
        
        if 'decoder_optimizer' in checkpoint:
            try:
                decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'].state_dict())
            except AttributeError:
                decoder_optimizer = checkpoint['decoder_optimizer']

        if 'encoder_trans_optimizer' in checkpoint:
            try:
                encoder_trans_optimizer.load_state_dict(checkpoint['encoder_trans_optimizer'].state_dict())
            except AttributeError:
                encoder_trans_optimizer = checkpoint['encoder_trans_optimizer']

        if 'encoder_optimizer' in checkpoint:
            try:
                encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'].state_dict())
            except AttributeError:
                encoder_optimizer = checkpoint['encoder_optimizer']
        if args.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.encoder_lr)

    # --- MONKEY PATCHING SİLİNDİ ---
    # decoder.sample_scst ve decoder.sample_greedy artık sınıfın kendi metodlarıdır.
    # -------------------------------

    # Move to GPU
    encoder = encoder.cuda()
    encoder_trans = encoder_trans.cuda()
    decoder = decoder.cuda()
    
    criterion_ce = torch.nn.CrossEntropyLoss().cuda()

    def move_optimizer_to_gpu(optimizer):
        if optimizer is not None:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    move_optimizer_to_gpu(decoder_optimizer)
    move_optimizer_to_gpu(encoder_trans_optimizer)
    move_optimizer_to_gpu(encoder_optimizer)

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
    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=8, gamma=0.5) if args.fine_tune_encoder else None
    encoder_trans_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(encoder_trans_optimizer, T_max=args.num_epochs, eta_min=1e-6)
    decoder_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=args.num_epochs, eta_min=1e-6)
    
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
                
                # 1. Greedy Search (Baseline)
                # Modelin kendi metodunu çağırıyoruz
                greedy_seqs = decoder.sample_greedy(feat1, feat2, sample_max_len=args.max_length)

                # 2. Monte-Carlo Sampling (Policy)
                # Modelin kendi metodunu çağırıyoruz
                sample_seqs, sample_log_probs = decoder.sample_scst(feat1, feat2, sample_max_len=args.max_length)
                
                # 3. Reward Hesaplama
                rewards, baselines = get_self_critical_reward(decoder, sample_seqs, greedy_seqs, token.tolist(), word_vocab)
                
                # 4. Loss Hesaplama
                advantage = rewards - baselines
                
                # Maskeleme
                pad_mask = (sample_seqs != word_vocab['<NULL>']).float()
                
                # Ortalama loss
                loss = - (advantage.unsqueeze(1) * sample_log_probs * pad_mask).sum() / args.train_batchsize
                
                acc = 0

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
    parser.add_argument('--print_freq',type=int, default=10)
    
    # Training Params
    parser.add_argument('--fine_tune_encoder', type=bool, default=True)    
    parser.add_argument('--train_batchsize', type=int, default=32) 
    parser.add_argument('--network', default='resnet101')
    parser.add_argument('--encoder_dim',default=1024)
    parser.add_argument('--feat_size', default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    
    # LR ayarları
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
                        help='SCST kaybının başlayacağı epoch.')

    args = parser.parse_args()
    main(args)