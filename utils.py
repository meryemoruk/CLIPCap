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

    def forward(self, d12, d21):
        """
        d12: (Batch, Dim) - Before-to-After değişim vektörü
        d21: (Batch, Dim) - After-to-Before değişim vektörü
        """
        batch_size = d12.size(0)
        
        # Pozitif benzerlik (Kendi eşleşmesi)
        # s(d12, d21)
        pos_sim = self.cosine_sim(d12, d21) # (Batch)
        
        loss = 0
        # Negatif örnekleme: Batch içindeki diğer örnekleri negatif olarak kullanıyoruz.
        # Makale "hard negatives" veya batch içi negatiflerden bahseder.
        # Basitlik ve verimlilik için batch içindeki en yakın negatifi (hardest negative) seçebiliriz
        # veya tüm negatiflerin ortalamasını alabiliriz. Burada standart triplet mantığıyla
        # batch'i karıştırarak (roll) negatif üretiyoruz.
        
        # Negatif örnekler (Batch'i kaydırarak elde ediyoruz)
        d21_neg = torch.roll(d21, shifts=1, dims=0)
        d12_neg = torch.roll(d12, shifts=1, dims=0)
        
        # s(d12, d21-)
        neg_sim_1 = self.cosine_sim(d12, d21_neg)
        # s(d21, d12-)
        neg_sim_2 = self.cosine_sim(d21, d12_neg)
        
        # Denklem 11: L_r = max(0, margin - s(pos) + s(neg))
        # Not: Makalede s(.) cosine distance denmiş ama formül yapısı (gamma - s + s_neg) 
        # ve standart triplet loss mantığına göre s(.) similarity olmalıdır.
        # Eğer s similarity ise, pos yüksek, neg düşük olmalı.
        # Loss = max(0, margin - (sim_pos - sim_neg)) = max(0, margin - sim_pos + sim_neg)
        
        loss1 = torch.clamp(self.margin - pos_sim + neg_sim_1, min=0.0)
        loss2 = torch.clamp(self.margin - pos_sim + neg_sim_2, min=0.0)
        
        total_loss = torch.mean(loss1 + loss2)
        return total_loss