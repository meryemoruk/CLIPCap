import torch
from torch import nn
import torchvision.models as models
from einops import rearrange
import clip
import math
import torch.nn.functional as F

# --- maske ---
from torchvision import transforms




class DinoMaskGenerator(nn.Module):
    def __init__(self, model_type='dinov2_vits14'):
        super().__init__()
        print(f"Loading {model_type}...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_type)
        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # DINOv2 için gerekli normalizasyon değerleri
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, img):
        # 2. Normalize
        img = (img - self.mean) / self.std
        return img

    def forward(self, img1, img2):
        with torch.no_grad():
            # (Batch, 3, H, W)
            p_img1 = self.preprocess(img1)
            p_img2 = self.preprocess(img2)
            
            # Özellik Çıkarımı
            feat1 = self.backbone.forward_features(p_img1)["x_norm_patchtokens"]
            feat2 = self.backbone.forward_features(p_img2)["x_norm_patchtokens"]
            
            # Benzerlik ve Maske Hesabı (Cosine Similarity)
            similarity = F.cosine_similarity(feat1, feat2, dim=-1)
            mask = 1.0 - similarity # Fark ne kadar büyükse değer 1'e o kadar yakın olur
            
            # Kare forma dönüştürme (Grid)
            B, N = mask.shape
            H_grid = int(N**0.5) # Örn: 256 -> 16
            mask = mask.view(B, 1, H_grid, H_grid)
            
            return mask

class FeatureCNN(nn.Module):
    def __init__(self, input_dim):
        super(FeatureCNN, self).__init__()
                
        # Spatial Modeling (ResNet Bloğu benzeri yapı)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_dim)
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: (Batch, Channels, H, W) -> Örn: (B, 1024, 16, 16)
            
        # 2. Residual Connection ile Spatial İşlem
        identity = x
        out = self.spatial_conv(x)
        out += identity  # Skip connection
        out = self.relu(out)
        
        return out

class ClipEncoder(nn.Module):
    def __init__(self, path = "/content/CLIPCap/RemoteCLIP-ViT-L-14.pt"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model, preprocess = clip.load("ViT-L/14", device=self.device, jit=False)
        except AttributeError:
            pass 

        # CLIP (OpenAI) İstatistikleri
        self.register_buffer('target_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('target_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        jit_path = path
        print(jit_path)

        if (True):
            # HATA ÇÖZÜMÜ: torch.jit.load YERİNE torch.load kullanıyoruz
            checkpoint = torch.load(jit_path, map_location=self.device)
            
            # Checkpoint yapısını kontrol et (Bazen direkt dict, bazen 'state_dict' key'i içinde olur)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Temizlik: Eğer anahtarlarda 'module.' varsa (DataParallel artığı) temizle
            clean_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace("module.", "")
                clean_state_dict[new_k] = v
            
            # Ağırlıkları Yükle (strict=False, versiyon farkı varsa patlamaması için)
            # DOĞRU: Sadece fonksiyonu çağır, atama yapma
            self.model.load_state_dict(clean_state_dict, strict=False)

        else:
            print("UYARI: JIT dosyası bulunamadı, orijinal ağırlıklar kullanılıyor.")

        # --- 3. DERİN HOOK MEKANİZMASI (ÇÖZÜM BURADA) ---
        self.features = {}

        def get_activation_deep(name):
            def hook(model, input, output):
                self.features[name] = output.detach().clone()
            return hook

        self.model.visual.transformer.resblocks[-1].register_forward_hook(get_activation_deep('last_block'))

    def forward(self, img):
        
        # 2. Normalize (Raw -> CLIP)
        img = (img - self.target_mean) / self.target_std

        self.feature = {}

        with torch.no_grad():
            _ = self.model.encode_image(img)
            
        # --- 5. Veriyi İşleme (Manuel Normalizasyon) ---
        # Transformer çıktısı: [Seq_Len, Batch, Dim] -> Genelde [50, 1, 768]
        raw_output = self.features['last_block']
        
        # Şekil Düzenleme: [Seq, Batch, Dim] -> [Batch, Seq, Dim]
        
        raw_output = raw_output.permute(1, 0, 2) # [1, 50, 768]
        
        # KRİTİK ADIM: Hook'u ln_post öncesine attığımız için,
        # LayerNorm işlemini şimdi elle yapmalıyız. Yoksa veriler normalize olmaz.
        final_features = self.model.visual.ln_post(raw_output)

        patch_tokens = final_features[:, 1:, :] 
            
        batch_size = patch_tokens.shape[0]     # 1
        num_patches = patch_tokens.shape[1]    # 49 olmalı
        embed_dim = patch_tokens.shape[2]      # 768
        
        grid_size = int(num_patches ** 0.5)    # 7

        return patch_tokens.reshape(batch_size, grid_size, grid_size, embed_dim).permute(0, 3, 1, 2)

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
        elif self.network=='clip':
            clip = ClipEncoder()

        if('clip' in self.network):
            self.model = clip
        else:
            self.model = nn.Sequential(*modules)
            # Resize image to fixed size to allow input images of variable size
            # self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
            self.fine_tune()

        for param in self.model.parameters():
            param.requires_grad = False

        # --- MASKE ---
        self.dino = DinoMaskGenerator()

        self.featureCNN = FeatureCNN(1024)

    def forward(self, imageA, imageB):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """

        # 1. Resize (224'ün katları)
        imageA = F.interpolate(imageA, size=(224, 224), mode='bicubic', align_corners=False)
        imageB = F.interpolate(imageB, size=(224, 224), mode='bicubic', align_corners=False)

        mask =  None
        with torch.no_grad():
            mask = self.dino(imageA, imageB)

            # mask = F.interpolate(mask, size=(224, 224), mode='bicubic')

            # mask_feat = self.encoder(mask)

            # maskedA = imageA * mask
            # maskedB = imageB * mask

            # feat1 = self.model(imageA)  # (batch_size, 2048, image_size/32, image_size/32)
            # feat2 = self.model(imageB)

            featA = self.model(imageA)  # (batch_size, 2048, image_size/32, image_size/32)
            featB = self.model(imageB)

            featA = self.featureCNN(featA.float())
            featB = self.featureCNN(featB.float())

            mask_spatial = F.interpolate(mask, size=featA.shape[2:], mode='bicubic')

            featA = featA * mask_spatial
            featB = featB * mask_spatial

        # maskedfeat1 = torch.cat([feat1, maskedfeat1], dim=1)
        # maskedfeat2 = torch.cat([feat2, maskedfeat2], dim=1)

        return featA, featB, mask

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


class FusionConvBlock(nn.Module):
    def __init__(self, dim):
        super(FusionConvBlock, self).__init__()
        
        # Giriş boyutu concat olduğu için (dim * 2) olacak.
        # Hedef boyut (dim) olacak.
        
        # 1. Adım: Boyut düşürme ve özellik karıştırma
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # 2. Adım: Mekansal (Spatial) işleme (3x3)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        
        # Non-linearity
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: [Batch, 2*Dim, Height, Width]
        
        # Önce boyutu düşür
        x_reduced = self.reduce_conv(x)
        
        # Residual bağlantı için kopyasını tut
        identity = x_reduced
        
        # Mekansal işleme
        out = self.spatial_conv(x_reduced)
        
        # Residual bağlantı (Skip Connection)
        out = out + identity
        out = self.final_relu(out)
        
        return out

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, mask=None):
        """
        tgt: Image B (Current)
        memory: Image A (Reference)
        """
        
        # 2. Cross Attention (Image B, Image A'ya bakar: "Ne değişti?")
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.cross_attn(query=tgt2, key=memory, value=memory)
        tgt = tgt + self.dropout(tgt2)

        # 3. Feed Forward
        tgt2 = self.norm2(tgt)
        tgt = tgt + self.dropout(self.ffn(tgt2))
        
        return tgt

class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Özellik haritasını 1 kanala (önem derecesine) indir
        self.conv = nn.Conv1d(dim, 1, kernel_size=1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_original, x_fused):
        # x_fused: [B, N, D] -> [B, D, N] (Conv1d için)
        map_in = x_fused.permute(0, 2, 1)
        
        # Dikkat haritası: [B, 1, N]
        attention_map = self.sigmoid(self.conv(map_in))
        
        # [B, 1, N] -> [B, N, 1]
        attention_map = attention_map.permute(0, 2, 1)
        
        # Orijinal veriyi önem derecesine göre ölçekle
        return x_original * attention_map

class FiLMBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        # img_fused (condition) bilgisinden gamma ve beta üret
        # dim -> dim * 2 (hem gamma hem beta için)
        self.cond_proj = nn.Linear(dim, dim * 2)
        
        # İsteğe bağlı: Gamma ve Beta'nın hemen devreye girmemesi için 
        # ağırlıkları 0'a yakın başlatmak iyi bir pratiktir.
        self._init_weights()

    def _init_weights(self):
        # Linear katmanın ağırlıklarını sıfıra yakın başlatıyoruz
        nn.init.constant_(self.cond_proj.weight, 0)
        nn.init.constant_(self.cond_proj.bias, 0)

    def forward(self, x_content, x_condition):
        """
        x_content: img_sa1 veya img_sa2 (Değiştirilecek özellik)
        x_condition: img_fused (Değişiklik bilgisi)
        """
        
        # 1. Gamma ve Beta'yı üret
        # x_condition shape: [Batch, N, Dim]
        params = self.cond_proj(x_condition)
        
        # Chunk ile ikiye böl: Gamma ve Beta
        gamma, beta = torch.chunk(params, 2, dim=-1)
        
        # 2. Residual FiLM İşlemi (Formül: (1 + gamma) * x + beta)
        # Eğer gamma ve beta 0 ise, sonuç direkt x_content olur (Residual etkisi).
        out = (1 + gamma) * x_content + beta
        
        return out

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
                FusionConvBlock(channels),
                FiLMBlock(channels)
                # Transformer(channels, channels, heads, attention_dim, hidden_dim, dropout, norm_first=False),
            ]))

        # self.cross_attr1 = Transformer(channels, channels, heads, attention_dim, hidden_dim, dropout, norm_first=False)
        # self.cross_attr2 = Transformer(channels, channels, heads, attention_dim, hidden_dim, dropout, norm_first=False)
        # self.last_norm1 = nn.LayerNorm(channels)
        # self.last_norm2 = nn.LayerNorm(channels)

        # self.maskedsizetonormal = nn.Sequential(
        #     nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(inplace=True) 
        # )

        # self.maskedsizetonormal = self.maskedsizetonormal.half()

        
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img1, img2, mask=None):
        

        # # --- KRİTİK DÜZELTME: Maske Boyutlandırma ---
        # if mask is not None:
        #     # 1. Maskeyi, mevcut özellik haritası boyutuna (h, w) getir (Örn: 7x7)
        #     # DINO (16x16) -> CLIP (7x7)
        #     if mask.shape[-2:] != (h, w):
        #         mask = F.interpolate(mask, size=(h, w), mode='nearest')
            
        #     # 2. Transformer'ın anlayacağı formata (Flatten) çevir
        #     # (Batch, 1, h, w) -> (Batch, 1, 1, h*w) -> (Batch, 1, 1, 49)
        #     # Bu format, (Batch, Heads, 49, 49) matrisiyle işlem yapmaya uygundur.
        #     mask = mask.view(batch, 1, 1, h * w)
        # # --------------------------------------------

        # img1 = self.maskedsizetonormal(img1)
        # img2 = self.maskedsizetonormal(img2)

        batch, c, h, w = img1.shape

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

        for (resblock, film) in self.selftrans:           
            img_concat = torch.cat([img_sa1, img_sa2], dim = -1)
            
            H = W = int(img_concat.shape[1] ** 0.5) 
            # [B, N, 2D] -> [B, 2D, N] -> [B, 2D, H, W]
            img_spatial = img_concat.permute(0, 2, 1).view(-1, img_concat.shape[-1], H, W)

            img_fused = resblock(img_spatial)

            img_fused = img_fused.flatten(2).permute(0, 2, 1)

            img_sa1 = film(img_sa1, img_fused)
            img_sa2 = film(img_sa2, img_fused)


        img1 = img_sa1.reshape(batch, h, w, c).transpose(-1, 1)
        img2 = img_sa2.reshape(batch, h, w, c).transpose(-1, 1)

        return img1, img2
    



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Sinusoidal Positional Encoding (CLIP özelliklerinde daha stabildir)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class ChangeAwareEncoder(nn.Module):
    def __init__(self, feature_dim=1024, heads=8, n_layers=2, hidden_dim=2048, dropout=0.1):
        super().__init__()
        
        # Giriş projeksiyonu (CLIP 1024 -> 512 gibi küçültmek hesaplama yükünü azaltır)
        self.project = nn.Linear(feature_dim, 512) 
        
        # Sadece Encoder Layer kullanın (Self-Attention)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=512, nhead=heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # Son çıkış düzeltme
        self.output_head = nn.Linear(512, feature_dim)

    def forward(self, featA, featB, mask=None):
        # (Batch, C, H, W) -> (Batch, Seq, C)
        featA = featA.flatten(2).transpose(1, 2)
        featB = featB.flatten(2).transpose(1, 2)
        
        # Boyut düşürme (Opsiyonel ama önerilir)
        featA = self.project(featA)
        featB = self.project(featB)

        # 1. Fark Vektörü (Explicit Difference)
        # Modelin işini kolaylaştırın: Farkı elle verin.
        diff = featB - featA 
        
        # 2. Hepsini Birleştir: [Eski, Yeni, Fark]
        # Bu sayede model hem bağlamı hem de değişimi görür.
        combined = torch.cat([featA, featB, diff], dim=1) # Seq boyutu 3 katına çıkar (3*49)
        
        # 3. Maskeyi (Varsa) sadece Diff kısmına uygula veya tamamen kaldır
        # Şimdilik maskeyi kaldırıyorum, çünkü DINO maskesi captioning'i bozabilir.
        
        # 4. Transformer Encoder (Self Attention)
        for layer in self.layers:
            combined = layer(combined)
            
        # 5. Sadece "Fark" veya "Yeni" kısmını değil, tüm bağlamı decoder'a gönder
        # Ancak Decoder boyutu sabit bekliyorsa, Average Pooling yapabiliriz veya
        # sadece 'diff' kısmına karşılık gelen çıkışı alabiliriz.
        
        # En iyi yöntem: Decoder'a hepsini vermek.
        # Eğer decoder token sayısı sorunu yoksa 'combined' döndür.
        # Eğer varsa, tekrar feature boyutuna indirge.
        
        out = self.output_head(combined)
        
        return out