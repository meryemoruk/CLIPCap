import torch
from torch import nn
import torchvision.models as models
from einops import rearrange
import clip
import os
import torch.nn.functional as F

# --- Cradio ---
import numpy as np
from transformers import AutoModel, CLIPImageProcessor
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, network):
        super(Encoder, self).__init__()
        self.network = network
        if self.network=='alexnet': #256,7,7
            cnn = models.alexnet(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg11': #512,1/32H,1/32W
            cnn = models.vgg11(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg16': #512,1/32H,1/32W
            cnn = models.vgg16(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg19':#512,1/32H,1/32W
            cnn = models.vgg19(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='inception': #2048,6,6
            cnn = models.inception_v3(pretrained=True, aux_logits=False)  
            modules = list(cnn.children())[:-3]
        elif self.network=='resnet18': #512,1/32H,1/32W
            cnn = models.resnet18(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet34': #512,1/32H,1/32W
            cnn = models.resnet34(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet50': #2048,1/32H,1/32W
            cnn = models.resnet50(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet101':  #2048,1/32H,1/32W
            cnn = models.resnet101(pretrained=True)  
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet152': #512,1/32H,1/32W
            cnn = models.resnet152(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext50_32x4d': #2048,1/32H,1/32W
            cnn = models.resnext50_32x4d(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext101_32x8d':#2048,1/256H,1/256W
            cnn = models.resnext101_32x8d(pretrained=True)  
            modules = list(cnn.children())[:-1]
        elif self.network=='densenet121': #no AdaptiveAvgPool2d #1024,1/32H,1/32W
            cnn = models.densenet121(pretrained=True) 
            modules = list(cnn.children())[:-1] 
        elif self.network=='densenet169': #1664,1/32H,1/32W
            cnn = models.densenet169(pretrained=True)  
            modules = list(cnn.children())[:-1]
        elif self.network=='densenet201': #1920,1/32H,1/32W
            cnn = models.densenet201(pretrained=True)  
            modules = list(cnn.children())[:-1]
        elif self.network=='regnet_x_400mf': #400,1/32H,1/32W
            cnn = models.regnet_x_400mf(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='regnet_x_8gf': #1920,1/32H,1/32W
            cnn = models.regnet_x_8gf(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='regnet_x_16gf': #2048,1/32H,1/32W
            cnn = models.regnet_x_16gf(pretrained=True) 
            modules = list(cnn.children())[:-2]
        
        model_name = "nvidia/C-RADIOv4-H" 
        # Transformers kütüphanesi ile modeli çekiyoruz (zoo.py mantığı)
        # trust_remote_code=True kritik, çünkü özel bir mimari kullanıyor.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
        self.processor = CLIPImageProcessor.from_pretrained(model_name)


    def forward(self, imageA, imageB):
        """
        Forward propagation.
        Inputs (imageA, imageB) should be PIL Images or list of PIL images.
        If they are already Tensors, you should bypass self.processor!
        """
        
        # 1. İşlemci (Processor) ile veriyi hazırla ve GPU'ya taşı
        # Not: imageA ve imageB burada PIL Image listesi olmalıdır. 
        # Eğer DataLoader'dan Tensor geliyorsa processor kullanılmamalı, direkt resize/normalize yapılmalı.
        inputs_a = self.processor(images=imageA, return_tensors="pt", do_resize=True)
        inputs_b = self.processor(images=imageB, return_tensors="pt", do_resize=True)
        
        pixel_values_a = inputs_a.pixel_values.to(self.device)
        pixel_values_b = inputs_b.pixel_values.to(self.device)

        with torch.no_grad():
            # 2. Modelden geçir (Tuple döner: summary, spatial)
            _, feat1_spatial = self.model(pixel_values_a)  # [Batch, N_tokens, Channels]
            _, feat2_spatial = self.model(pixel_values_b)  # [Batch, N_tokens, Channels]

        # 3. Şekil Düzenleme (NLC -> NCHW) - KRİTİK ADIM
        # Model çıktısı: [Batch, Sequence_Length, Dim] -> Örn: [B, 256, 1280]
        # Hedef çıktı:   [Batch, Dim, Height, Width]   -> Örn: [B, 1280, 16, 16]
        
        feat1_spatial = self._reshape_to_spatial(feat1_spatial)
        feat2_spatial = self._reshape_to_spatial(feat2_spatial)

        return feat1_spatial, feat2_spatial, None

    def _reshape_to_spatial(self, x):
        """
        [B, N, C] formatını [B, C, H, W] formatına çevirir.
        """
        B, N, C = x.shape
        H = W = int(N ** 0.5) # Karekök alarak Grid boyutunu bul (örn: 14x14 veya 16x16)
        
        # 1. [B, N, C] -> [B, H, W, C]
        x = x.reshape(B, H, W, C)
        
        # 2. [B, H, W, C] -> [B, C, H, W] (Permute)
        x = x.permute(0, 3, 1, 2)
        return x

    def fine_tune(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.model.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.model.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAtt(nn.Module):
    def __init__(self, dim_q, dim_kv, attention_dim, heads = 8, dropout = 0.):
        super(MultiHeadAtt, self).__init__()
        project_out = not (heads == 1 and attention_dim == dim_kv)
        self.heads = heads
        self.scale = (attention_dim // self.heads) ** -0.5

        self.to_q = nn.Linear(dim_q, attention_dim, bias = False)
        self.to_k = nn.Linear(dim_kv, attention_dim, bias = False)
        self.to_v = nn.Linear(dim_kv, attention_dim, bias = False)       
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(attention_dim, dim_q),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2, x3, mask=None):
        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_v(x3)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # --- MASKE ---
        if mask is not None:
            # Maske şekli (Batch, 1, 1, Seq_Len) veya (Batch, 1, Seq_Len, Seq_Len) olmalı.
            dots = dots.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)#(b,n,dim)

class Transformer(nn.Module):
    def __init__(self, dim_q, dim_kv, heads, attention_dim, hidden_dim, dropout = 0., norm_first = False):
        super(Transformer, self).__init__()
        self.norm_first = norm_first
        self.att = MultiHeadAtt(dim_q, dim_kv, attention_dim, heads = heads, dropout = dropout)
        self.feedforward = FeedForward(dim_q, hidden_dim, dropout = dropout)
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)

       

    def forward(self, x1, x2, x3, mask=None):
        if self.norm_first:
            x = self.att(self.norm1(x1), self.norm1(x2), self.norm1(x3), mask) + x1
            x = self.feedforward(self.norm2(x)) + x
        else:
            x = self.norm1(self.att(x1, x2, x3, mask) + x1)
            x = self.norm2(self.feedforward(x) + x)

        return x

class AttentiveEncoder(nn.Module):
    """
    One visual transformer block
    """
    def __init__(self, n_layers, feature_size, heads, hidden_dim, attention_dim = 512, dropout = 0., network = "resnet101"):
        super(AttentiveEncoder, self).__init__()
        h_feat, w_feat, channels = feature_size
        self.network = network
        
        self.h_embedding = nn.Embedding(h_feat, int(channels/2))
        self.w_embedding = nn.Embedding(w_feat, int(channels/2))
        self.selftrans = nn.ModuleList([])
        for i in range(n_layers):                 
            self.selftrans.append(nn.ModuleList([
                Transformer(channels, channels, heads, attention_dim, hidden_dim, dropout, norm_first=False),
                Transformer(channels*2, channels*2, heads, attention_dim, hidden_dim, dropout, norm_first=False),
            ]))

        # self.cross_attr1 = Transformer(channels, channels, heads, attention_dim, hidden_dim, dropout, norm_first=False)
        # self.cross_attr2 = Transformer(channels, channels, heads, attention_dim, hidden_dim, dropout, norm_first=False)
        # self.last_norm1 = nn.LayerNorm(channels)
        # self.last_norm2 = nn.LayerNorm(channels)

         
        
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img1, img2, mask=None):
        batch, c, h, w = img1.shape

        # --- KRİTİK DÜZELTME: Maske Boyutlandırma ---
        if mask is not None:
            # 1. Maskeyi, mevcut özellik haritası boyutuna (h, w) getir (Örn: 7x7)
            # DINO (16x16) -> CLIP (7x7)
            if mask.shape[-2:] != (h, w):
                mask = F.interpolate(mask, size=(h, w), mode='nearest')
            
            # 2. Transformer'ın anlayacağı formata (Flatten) çevir
            # (Batch, 1, h, w) -> (Batch, 1, 1, h*w) -> (Batch, 1, 1, 49)
            # Bu format, (Batch, Heads, 49, 49) matrisiyle işlem yapmaya uygundur.
            mask = mask.view(batch, 1, 1, h * w)
        # --------------------------------------------

        pos_h = torch.arange(h).to(img1.device)
        pos_w = torch.arange(w).to(img1.device)
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)

        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                       embed_h.unsqueeze(1).repeat(1, w, 1)], 
                                       dim = -1)                            
        pos_embedding = pos_embedding.permute(2,0,1).unsqueeze(0).repeat(batch, 1, 1, 1)
        img1 = img1 + pos_embedding
        img2 = img2 + pos_embedding
        img1 = img1.view(batch, c, -1).transpose(-1, 1)#batch, hw, c
        img2 = img2.view(batch, c, -1).transpose(-1, 1)
        img_sa1, img_sa2 = img1, img2

        for (l, m) in self.selftrans:           
            img_sa1 = l(img_sa1, img_sa1, img_sa1, mask) + img_sa1
            img_sa2 = l(img_sa2, img_sa2, img_sa2, mask) + img_sa2
            # img_ca1 = self.cross_attr1(img_sa1, img_sa2, img_sa2)
            # img_ca2 = self.cross_attr1(img_sa2, img_sa1, img_sa1)
            img = torch.cat([img_sa1, img_sa2], dim = -1)
            # mask_double = torch.cat([mask, mask], dim = -1)
            img = m(img, img, img, mask)
            img_sa1 = img[:,:,:c] + img1 #+ img_ca1
            img_sa2 = img[:,:,c:] + img2 #+ img_ca2
            # img_sa1 = self.last_norm1(img_sa1)
            # img_sa2 = self.last_norm2(img_sa2)

        img1 = img_sa1.reshape(batch, h, w, c).transpose(-1, 1)
        img2 = img_sa2.reshape(batch, h, w, c).transpose(-1, 1)

        return img1, img2