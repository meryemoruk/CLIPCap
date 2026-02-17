import time
import os
import numpy as np
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json
#import torchvision.transforms as transforms
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from data.SECOND_CC.SECONDCC import SECONDCCDataset
from model.model_encoder import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils import *

def count_parameters(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name}: {total_params:,} toplam parametre | {trainable_params:,} eğitilebilir parametre")
    return trainable_params

import matplotlib.pyplot as plt
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

def visualize_results(img1_tensor, img2_tensor, mask_tensor, output_path="result.png"):
    """
    Görüntüleri görselleştirir ve kaydeder.
    Otomatik 'Denormalization' yapar (ImageNet standartlarına göre).
    """
    
    # --- 1. Batch Boyutunu Yönetme ---
    if img1_tensor.dim() == 4: img1_use = img1_tensor[0]
    else: img1_use = img1_tensor

    if img2_tensor.dim() == 4: img2_use = img2_tensor[0]
    else: img2_use = img2_tensor

    if mask_tensor.dim() == 4: mask_use = mask_tensor[0]
    else: mask_use = mask_tensor

    # --- 2. Maskeyi Büyütme (Upsample) ---
    target_h, target_w = img1_use.shape[1], img1_use.shape[2]
    mask_resized = F.interpolate(mask_use.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    # --- 3. Tensor -> Numpy ve Renk Kanalı Düzenleme (H, W, C) ---
    img1_np = img1_use.detach().permute(1, 2, 0).cpu().numpy()
    img2_np = img2_use.detach().permute(1, 2, 0).cpu().numpy()
    mask_np = mask_resized.squeeze().detach().cpu().numpy()

    # --- 4. DENORMALIZATION (ÖNEMLİ ADIM) ---
    # ImageNet Mean ve Std değerleri (CLIP ve DINO bunları kullanır)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Formül: original = (normalized * std) + mean
    img1_np = (img1_np * std) + mean
    img2_np = (img2_np * std) + mean

    # Değerleri [0, 1] aralığına sıkıştır (Clip)
    # Bu işlem float hatalarını temizler ve 'Clipping' uyarısını kesin çözer.
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)

    # --- 5. Çizim ---
    plt.figure(figsize=(15, 5))

    # Before Image
    plt.subplot(1, 3, 1)
    plt.imshow(img1_np)
    plt.title("Önce (Before)")
    plt.axis('off')

    # After Image
    plt.subplot(1, 3, 2)
    plt.imshow(img2_np)
    plt.title("Sonra (After)")
    plt.axis('off')

    # Difference Mask (Heatmap)
    plt.subplot(1, 3, 3)
    plt.imshow(mask_np, cmap='jet', vmin=0, vmax=1) 
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Değişim Maskesi")
    plt.axis('off')

    plt.tight_layout()
    
    # Kaydetme
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Görsel kaydedildi (Düzeltilmiş): {output_path}")

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
    best_epoch  = 0
    best_bleu4  = 0  # BLEU-4 score right now
    best_rouge  = 0
    best_cider  = 0
    best_meteor = 0
    best_avg    = 0

    start_epoch = 0
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    # Initialize / load checkpoint
    if args.checkpoint is None:      
        encoder = Encoder(args.network)   
        encoder.fine_tune(args.fine_tune_encoder)     
        # 1. Sadece gradyan isteyen (requires_grad=True) parametreleri filtrele
        trainable_params = filter(lambda p: p.requires_grad, encoder.parameters())

        # 2. Eğer eğitilecek parametre varsa (Adapter gibi), optimizer'ı oluştur.
        # fine_tune_encoder False olsa bile Adapter eğitilmeli.
        encoder_optimizer = torch.optim.Adam(params=trainable_params, lr=args.encoder_lr)
        encoder_trans = AttentiveEncoder(n_layers =args.n_layers, feature_size=[args.feat_size, args.feat_size, args.encoder_dim], 
                                            heads=args.n_heads, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, dropout=args.dropout, network=args.network)
        # encoder_trans_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_trans.parameters()),
        #                                     lr=args.encoder_lr)
        encoder_trans_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, encoder_trans.parameters()),
                                            lr=args.encoder_lr, weight_decay=1e-4)
        decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_vocab), max_lengths=args.max_length, word_vocab=word_vocab, n_head=args.n_heads,
                                    n_layers= args.decoder_n_layers, dropout=args.dropout)
        # decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
        #                                     lr=args.decoder_lr)
        decoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, decoder.parameters()), 
                                              lr=args.decoder_lr, weight_decay=1e-4)
    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder_trans = checkpoint['encoder_trans']
        encoder_trans_optimizer = checkpoint['encoder_trans_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if args.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                            lr=args.encoder_lr)
    # Move to GPU, if available
    encoder = encoder.cuda()
    encoder_trans = encoder_trans.cuda()
    decoder = decoder.cuda()
    # Loss function
    # Loss fonksiyonunu tanımla
    ranking_loss_fn = BiDirectionalRankingLoss(margin=0.2).cuda()
    lambda_r = 0.2 # Makalede önerilen ağırlık [cite: 362]
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).cuda()

    # ------- PARAMETRE -------
    print("-" * 50)
    print("MODEL PARAMETRE İSTATİSTİKLERİ")
    print("-" * 50)
    
    p_encoder = count_parameters(encoder, "Encoder (Backbone)")
    p_trans = count_parameters(encoder_trans, "Attentive Encoder")
    p_decoder = count_parameters(decoder, "Decoder Transformer")
    
    total_trainable = p_encoder + p_trans + p_decoder
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
    elif args.data_name == 'SECOND_CC':
        train_loader = data.DataLoader(
            SECONDCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = data.DataLoader(
            SECONDCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            json_file='./data/SECOND_CC/SECOND_CC.json', batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=8, gamma=0.5) if args.fine_tune_encoder else None
    # encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_trans_optimizer, step_size=5, gamma=0.5)
    encoder_trans_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(encoder_trans_optimizer, T_max=args.num_epochs, eta_min=1e-6)
    # decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=8, gamma=0.5)
    decoder_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=args.num_epochs, eta_min=1e-6)

    index_i = 0
    hist = np.zeros((args.num_epochs * len(train_loader), 3))
    # Epochs
    mask_example_count = 9
    for epoch in range(start_epoch, args.num_epochs):        
        # Batches
        for id, (imgA, imgB, _, _, token, token_len, _, categories) in enumerate(train_loader):
            #if id == 20:
            #    break
            start_time = time.time()
            decoder.train()  # train mode (dropout and batchnorm is used)
            encoder.train()
            encoder_trans.train()
            decoder_optimizer.zero_grad()
            encoder_trans_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()

            # Move to GPU, if available
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            
            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()
            # Forward prop.
            feat1, feat2, mask = encoder(imgA, imgB)

            # --- MASK SAVING ---
            if(mask_example_count != 0):
                visualize_results(imgA, imgB, mask, "./"+str(mask_example_count)+".png")
                mask_example_count -= 1
            # --- MASK SAVING ---

            feat1, feat2 = encoder_trans(feat1, feat2, mask)
            scores, caps_sorted, decode_lengths, sort_ind, d12_vec, d21_vec = decoder(feat1, feat2, token, token_len)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            # 1. Caption Loss (Mevcut)

            categories = categories.cuda()

            # 2. Decoder içinde 'sort_ind' ile sıralama yapıldığı için, kategorileri de aynı sıraya sok
            # (Çünkü d12_vec ve d21_vec sıralandı)
            categories = categories[sort_ind]
            loss_ce = criterion(scores, targets)

            # 2. Ranking Loss (Yeni) [cite: 360-362]
            # d12_vec ve d21_vec zaten sort_ind ile sıralanmış gelecektir (decoder içinde handle edilirse).
            # Eğer decoder içinde sıralanmadıysa burada sort_ind ile sıralayın veya sıralanmamış haliyle kullanın (ikisi de aynı batch sırasında olduğu sürece sorun yok).
            # Decoder'dan dönen d12_vec'in caption_lengths ile sıralanmış olduğunu varsayarsak:
            loss_rank = ranking_loss_fn(d12_vec, d21_vec, categories)
            # Toplam Loss [cite: 360]
            loss = loss_ce + lambda_r * loss_rank
            # Back prop.
            loss.backward()
            
            # Clip gradients
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(encoder_trans.parameters(), args.grad_clip)
                # Filter parameters to only those that have gradients
                encoder_grads = [p for p in encoder.parameters() if p.grad is not None]
                # Only clip if there are gradients to clip
                if encoder_grads:
                    torch.nn.utils.clip_grad_value_(encoder_grads, args.grad_clip)

            # Update weights                      
            decoder_optimizer.step()
            encoder_trans_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Keep track of metrics     
            hist[index_i,0] = time.time() - start_time #batch_time        
            hist[index_i,1] = loss.item() #train_loss
            hist[index_i,2] = accuracy(scores, targets, 5) #top5
            index_i += 1   
            # Print status
            if index_i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time: {3:.3f}\t'
                    'Loss: {4:.4f}\t'
                    'Top-5 Accuracy: {5:.3f}'.format(epoch, index_i %len(train_loader), len(train_loader),
                                            np.mean(hist[index_i-args.print_freq:index_i-1,0])*args.print_freq,
                                            np.mean(hist[index_i-args.print_freq:index_i-1,1]),
                                            np.mean(hist[index_i-args.print_freq:index_i-1,2])))
        # One epoch's validation
        decoder.eval()  # eval mode (no dropout or batchnorm)
        encoder_trans.eval()
        if encoder is not None:
            encoder.eval()

        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]

        val_start_time = time.time()
        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)
        change_references = list()
        change_hypotheses = list()
        nochange_references = list()
        nochange_hypotheses = list()
        change_acc=0
        nochange_acc=0
        
        with torch.no_grad():
            # Batches
            for ind, (imgA, imgB, token_all, token_all_len, _, _, _) in enumerate(val_loader):
                # Move to GPU, if available
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                
                token_all = token_all.squeeze(0).cuda()

                # Forward prop.
                feat1, feat2, mask = encoder(imgA, imgB)

                feat1, feat2 = encoder_trans(feat1, feat2, mask)

                seq = decoder.sample_beam(feat1, feat2, args.beam_size)

                img_token = token_all.tolist()
                img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],
                        img_token))  # remove <start> and pads
                references.append(img_tokens)

                pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                hypotheses.append(pred_seq)
                assert len(references) == len(hypotheses)

                pred_caption = ""
                ref_caption = ""
                for i in pred_seq:
                    pred_caption += (list(word_vocab.keys())[i]) + " "
                for j in img_tokens[0]:
                    ref_caption += (list(word_vocab.keys())[j]) + " "
                
                if ind == 0 or ind == 120:
                    print("Prediction:")
                    print(pred_caption)
                    print("Referance:")
                    print(ref_caption)

                if ref_caption in nochange_list:
                    nochange_references.append(img_tokens)
                    nochange_hypotheses.append(pred_seq)
                    if pred_caption in nochange_list:
                        nochange_acc = nochange_acc+1
                else:
                    change_references.append(img_tokens)
                    change_hypotheses.append(pred_seq)
                    if pred_caption not in nochange_list:
                        change_acc = change_acc+1

            # Calculate evaluation scores
            print('len(nochange_references):', len(nochange_references))
            print('len(change_references):', len(change_references))
            val_time = time.time() - val_start_time
            # Calculate evaluation scores
            score_dict = get_eval_score(references, hypotheses)
            Bleu_1 = score_dict['Bleu_1']
            Bleu_2 = score_dict['Bleu_2']
            Bleu_3 = score_dict['Bleu_3']
            Bleu_4 = score_dict['Bleu_4']
            Meteor = score_dict['METEOR']
            Rouge = score_dict['ROUGE_L']
            Cider = score_dict['CIDEr']
            Avg = (Bleu_4 + Meteor + Rouge + Cider) / 4
            print('Validation:\n' 'Time: {0:.3f}\t' 'AVG: {1:.4f}\t' 'BLEU-4: {2:.4f}\t' 'Cider: {3:.4f}\t' 'BLEU-1: {4:.4f}\t' 
                'BLEU-2: {5:.4f}\t' 'BLEU-3: {6:.4f}\t' 'Meteor: {7:.4f}\t' 'Rouge: {8:.4f}\t'
                .format(val_time, Avg, Bleu_4, Cider, Bleu_1, Bleu_2, Bleu_3, Meteor, Rouge))
        
        #Adjust learning rate
        decoder_lr_scheduler.step()
        #print(decoder_optimizer.param_groups[0]['lr'])
        encoder_trans_lr_scheduler.step()
        if encoder_lr_scheduler is not None:
            encoder_lr_scheduler.step()
            #print(encoder_optimizer.param_groups[0]['lr'])
        # Check if there was an improvement        
        if  Avg > best_avg:
            best_avg = Avg
            best_bleu4 = Bleu_4
            best_cider = Cider
            best_rouge = Rouge
            best_meteor = Meteor
            best_epoch = epoch
            print('New Best:\n' 'AVG: {0:.4f}\t' 'BLEU-4: {1:.4f}\t' 'Cider: {2:.4f}\t' 'Meteor: {3:.4f}\t' 'Rouge: {4:.4f}\t'
                .format(best_avg, best_bleu4, best_cider, best_meteor, best_rouge))
            earlyStop = 0

            #save_checkpoint
            print('Save Model')  
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
            model_name = str(args.data_name)+'_batchsize_'+str(args.train_batchsize)+'_'+str(args.network)+'Bleu_4_'+str(round(10000*best_bleu4))+'.pth'
            torch.save(state, os.path.join(args.savepath, model_name))
        else:
            earlyStop += 1
            print(f"No improvement since {earlyStop} epochs.\n")

            print('Old Best:\n' 'Epoch: {0:d}\t' 'AVG: {1:.4f}\t' 'BLEU-4: {2:.4f}\t' 'Cider: {3:.4f}\t' 'Meteor: {4:.4f}\t' 'Rouge: {5:.4f}\t'
                .format(best_epoch, best_avg, best_bleu4, best_cider, best_meteor, best_rouge))
    
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
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')    
    parser.add_argument('--train_batchsize', type=int, default=64, help='batch_size for training')
    parser.add_argument('--network', default='resnet101', help='define the encoder to extract features')
    parser.add_argument('--encoder_dim',default=1024, help='the dimension of extracted features using different network')
    parser.add_argument('--feat_size', default=16, help='define the output size of encoder to extract features')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=8, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop')
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--beam_size', type=int, default=3, help='beam size for validation')
    parser.add_argument('--savepath', default="./models_checkpoint/")
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--attention_dim', type=int, default=512)
    parser.add_argument('--feature_dim', type=int, default=512)
    args = parser.parse_args()
    main(args)