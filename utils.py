import os
import torch
from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor



def save_checkpoint(args, data_name, epoch, encoder, encoder_feat, decoder, encoder_optimizer,
                encoder_feat_optimizer, decoder_optimizer, best_bleu4):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'best_bleu-4': best_bleu4,
             'encoder': encoder,
             'encoder_feat': encoder_feat,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'encoder_feat_optimizer': encoder_feat_optimizer,
             'decoder_optimizer': decoder_optimizer,
             }
    #filename = 'checkpoint_' + data_name + '_' + args.network + '.pth.tar'
    path = args.savepath #'./models_checkpoint/mymodel/3-times/'
    if os.path.exists(path)==False:
        os.makedirs(path)
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    torch.save(state, os.path.join(path, 'BEST_' + data_name))

    # torch.save(state, os.path.join(path, 'checkpoint_' + data_name +'_epoch_'+str(epoch) + '.pth.tar'))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]
    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        #print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
                
def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

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

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiDirectionalRankingLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(BiDirectionalRankingLoss, self).__init__()
        self.margin = margin

    def cosine_sim(self, x1, x2):
        # Cosine similarity: (Batch, Dim) -> (Batch)
        return F.cosine_similarity(x1, x2, dim=1)

    def forward(self, d12, d21, categories):
        """
        d12: (Batch, Dim)
        d21: (Batch, Dim)
        categories: (Batch)
        """
        # 1. Vektörleri Normalize Et (L2 Norm)
        # Cosine Similarity hesaplayabilmek için vektörlerin uzunluğunu 1'e sabitliyoruz.
        # Bu sayede sadece dot product (çarpım) yaparak cosine similarity bulabiliriz.
        d12_norm = F.normalize(d12, p=2, dim=1)
        d21_norm = F.normalize(d21, p=2, dim=1)
        
        # 2. Tam Benzerlik Matrisi (Cosine Similarity Matrix)
        # (Batch, Dim) x (Dim, Batch) -> (Batch, Batch)
        # scores[i][j]: i. görüntünün değişimi ile j. görüntünün değişimi arasındaki benzerlik.
        scores = torch.mm(d12_norm, d21_norm.t()) 
        
        # 3. Pozitif Skorlar (Köşegen Elemanlar)
        # i. görüntü kendisiyle (i) eşleşmeli. Bu bizim hedefimiz.
        pos_scores = scores.diag() 
        
        # 4. Kategori Maskesi (Negatif Seçimi İçin)
        # categories: [CatA, CatB, CatA, CatC]
        # Maske: Kategorisi FARKLI olanlar 1 (True), AYNI olanlar 0 (False).
        # diff_cat_mask[i][j] == 1 ise i ve j farklı kategoridedir (geçerli negatif).
        diff_cat_mask = (categories.unsqueeze(0) != categories.unsqueeze(1)).float()
        
        # 5. Maskeleme (Hard Negative Mining)
        # Bizim amacımız: Farklı kategoride olup (diff_cat_mask=1), 
        # benzerlik skoru en yüksek (modelin en çok karıştırdığı) örneği bulmak.
        
        # Skoru maskeliyoruz: Aynı kategoride olanları (mask=0) elemek için
        # skorlarını -sonsuz yapıyoruz. Böylece max işleminde seçilmezler.
        masked_scores = scores.clone()
        masked_scores[diff_cat_mask == 0] = -1e9 
        
        # 6. En Zor Negatifi Seç (Max Similarity)
        # Her satır (örnek) için geçerli negatifler arasındaki en yüksek skoru bul.
        hard_neg_scores, _ = torch.max(masked_scores, dim=1)
        
        # Güvenlik Kontrolü: Eğer batch içindeki herkes aynı kategorideyse,
        # geçerli hiç negatif yoktur. Bu durumda loss patlamasın diye kontrol ediyoruz.
        # valid_negatives: En az bir tane farklı kategorili eşleşmesi olanlar.
        valid_negatives = (diff_cat_mask.sum(dim=1) > 0).float()
        
        # 7. Loss Hesabı: max(0, margin - pos + neg)
        # margin: pozitif ile negatif arasında olmasını istediğimiz fark (örn: 0.2)
        loss = torch.clamp(self.margin - pos_scores + hard_neg_scores, min=0.0)
        
        # Sadece geçerli negatifi olan örneklerin ortalamasını al
        final_loss = (loss * valid_negatives).sum() / (valid_negatives.sum() + 1e-6)
        
        return final_loss