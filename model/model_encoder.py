#burası recall'u arttırmak için attention pooling eklemeden önceki encoder dosyası. recall 13 civarı gelmişti.


import torch
from torch import nn
import torchvision.models as models
from einops import rearrange
import clip
import os
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

class ClipMLP(nn.Module):
    def __init__(self, input_dim=768, output_dim=768, expansion_factor=4, dropout=0.1):
        """
        DINOv2 özellikleri için Hizalama (Alignment) Katmanı.
        
        Args:
            input_dim (int): DINOv2 çıktı boyutu (Base: 768, Small: 384, Large: 1024)
            output_dim (int): Modelin devamında kullanılacak boyut. 
                              - Stage 2'ye aktarırken genelde input_dim ile aynı (768) tutulur.
                              - CLIP Loss hesaplarken CLIP boyutuna (512) indirilir.
            expansion_factor (int): Gizli katman genişliği (Genelde 4 katı).
            dropout (float): Ezberlemeyi önlemek için dropout oranı.
        """
        super().__init__()
        
        hidden_dim = int(input_dim * expansion_factor)
        
        self.net = nn.Sequential(
            # 1. Genişletme (Linear)
            nn.Linear(input_dim, hidden_dim),
            
            # 2. Normalizasyon (Eğitim kararlılığı için kritik)
            nn.LayerNorm(hidden_dim),
            
            # 3. Aktivasyon (ReLU yerine modern GELU)
            nn.GELU(),
            
            # 4. Dropout
            nn.Dropout(dropout),
            
            # 5. İzdüşüm / Sıkıştırma (Linear)
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: [Batch, Patches, Dim] -> [B, 256, 768]
        return self.net(x)

class ClipProjector(nn.Module):
    def __init__(self, input_dim=768, output_dim=512):
        """
        DINOv2 özelliklerini CLIP kaybı (loss) hesaplamak için hazırlar.
        
        Args:
            input_dim (int): DINOv2 özellik boyutu (Base model: 768)
            output_dim (int): CLIP metin embedding boyutu (Genelde 512)
        """
        super().__init__()
        
        # Sadece boyutu değiştiren Linear katman (Projector Head)
        self.projector = nn.Linear(input_dim, output_dim)
        
        # Ağırlık başlatma (Xavier - Eğitim kararlılığı için)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projector.weight)
        if self.projector.bias is not None:
            nn.init.constant_(self.projector.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [Batch, H, W, C] formatında DINO özellikleri. Örn: [B, 16, 16, 768]
               veya [Batch, N, C] formatında. Örn: [B, 256, 768]
        Returns:
            projected: [Batch, 512] boyunda CLIP uyumlu vektör.
        """
        

        # 2. Projection (Loss için boyut indirgeme)
        # [Batch, 768] -> [Batch, 512]
        projected = self.projector(x)
        
        return projected

class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # CLIP'in orijinal başlangıç değeri (np.log(1 / 0.07))
        # Bu parametre eğitilebilir olmalı (requires_grad=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, image_features, text_features):
        """
        image_features: [Batch, 512] (Normalize edilmiş olmalı)
        text_features:  [Batch, 512] (Normalize edilmiş olmalı)
        """
        
        # 1. Normalizasyon Kontrolü (Güvenlik için tekrar yapabiliriz)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 2. Logit Scale'i al (Exp yaparak pozitif olmasını garantile)
        # Maksimum değer 100 olacak şekilde clip'lemek eğitimi stabilize eder.
        logit_scale = self.logit_scale.exp().clamp(max=100)

        # 3. Benzerlik Matrisi Hesapla (Cosine Similarity)
        # [B, 512] @ [512, B] -> [B, B]
        logits_per_image = logit_scale * image_features @ text_features.t()
        
        # Transpozu: Text -> Image logits
        logits_per_text = logits_per_image.t()

        # 4. Etiketler (Ground Truth)
        # 0, 1, 2, ..., Batch_Size-1 (Köşegenler doğrudur)
        batch_size = image_features.shape[0]
        device = image_features.device
        labels = torch.arange(batch_size, dtype=torch.long, device=device)

        # 5. İki Yönlü Cross Entropy Loss
        loss_i = F.cross_entropy(logits_per_image, labels) # Resim için doğru metni bul
        loss_t = F.cross_entropy(logits_per_text, labels)  # Metin için doğru resmi bul

        # 6. Simetrik Loss (Ortalama)
        total_loss = (loss_i + loss_t) / 2
        
        return total_loss

class RSCLIPTextEncoder(nn.Module):
    def __init__(self, model_name="ViT-B/32", checkpoint_path="RemoteCLIP-ViT-B-32.pt", device="cuda"):
        super().__init__()
        print(f"Loading RemoteCLIP Text Encoder ({model_name})...")
        
        # 1. Standart CLIP Mimarisi Yükle (Mimari iskeletini oluşturur)
        model, _ = clip.load(model_name, device=device, jit=False)
        
        # 2. RemoteCLIP Ağırlıklarını Yükle
        if checkpoint_path:
            print(f"Loading weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Eğer checkpoint 'state_dict' anahtarı içindeyse oradan al, yoksa direkt al
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 'module.' öneklerini temizle (DataParallel artığı olabilir)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
                
            # Ağırlıkları modele giydir (Strict=False, visual encoder farklılıklarını yoksaymak için)
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"Weights loaded: {msg}")

        # 3. Sadece Text Encoder'ı al, Visual kısmı sil
        self.text_encoder = model
        
        # 4. Dondur (Freeze)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        self.device = device

    def forward(self, text_tokens):
        with torch.no_grad():
            text_features = self.text_encoder.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

class ClipEncoder(nn.Module):
    def __init__(self, path = "/content/CLIPCap/RemoteCLIP-ViT-B-32.pt"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model, preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
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

        img = F.interpolate(img, size=(224, 224), mode='bicubic', align_corners=False)

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

    def __init__(self, network, args):
        super(Encoder, self).__init__()
        self.network = network
        
        if self.network=='clip':
            clip = ClipEncoder()
            
        # train_dino.py'de dropout=0.1 varsayılan olarak kullanılıyor.
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_mlp = ClipMLP(dropout=args.dropout).to(device)

        # 2. Checkpoint Dosyasını Yükleyin
        checkpoint_path = "./models_checkpoint/SECOND_CC_Best_Recall.pth" # Kendi dosya yolunuzu yazın

        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 3. Ağırlıkları Yükleyin (Strict=True önerilir, birebir eşleşme için)
            try:
                self.clip_mlp.load_state_dict(checkpoint['mlp_state_dict'])
                print("✅ DinoMLP başarıyla yüklendi.")
            except KeyError:
                print("❌ HATA: Checkpoint içinde 'mlp_state_dict' bulunamadı.")
                
        else:
            print(f"❌ Dosya bulunamadı: {checkpoint_path}")

        # 4. Modelleri Eval Moduna Alın (Test/Inference için şart)
        self.clip_mlp.eval()

        if('clip' in self.network):
            self.model = clip
            for param in self.clip_mlp.parameters():
                param.requires_grad = False
            self.clip_mlp.eval()
            self.dino = DinoMaskGenerator()

       
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

            feat1 = self.model(imageA)  # (batch_size, 2048, image_size/32, image_size/32)
            feat2 = self.model(imageB)

            feat1 = self.clip_mlp(feat1)  # (batch_size, 2048, image_size/32, image_size/32)
            feat2 = self.clip_mlp(feat2)

            feat1 = feat1.permute(0, 3, 1, 2)
            feat2 = feat2.permute(0, 3, 1, 2)

            mask_spatial = F.interpolate(mask, size=feat1.shape[2:], mode='bicubic')

            feat1 = feat1 * mask_spatial
            feat2 = feat2 * mask_spatial

        return feat1, feat2, mask

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

class DifferenceAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, img_feat_A, img_feat_B):
        # img_feat: [Batch, Sequence_Len, Dim] (Örn: B, 256, 768)
        
        # Query: After Image (Değişimi aradığımız yer)
        # Key/Value: Before Image (Referans)
        
        # Before görüntüsüne göre "farklılaşan" özellikleri bul
        attn_out, _ = self.cross_attn(query=img_feat_B, key=img_feat_A, value=img_feat_A)
        
        # After görüntüsünden, Before ile eşleşen kısımları çıkar (Soft Subtraction)
        difference = img_feat_B - attn_out 
        
        return self.layer_norm(difference)

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

    def forward(self, img1, img2):
        batch, c, h, w = img1.shape
        pos_h = torch.arange(h).cuda()
        pos_w = torch.arange(w).cuda()
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
            img_sa1 = l(img_sa1, img_sa1, img_sa1) + img_sa1
            img_sa2 = l(img_sa2, img_sa2, img_sa2) + img_sa2
            img = torch.cat([img_sa1, img_sa2], dim = -1)
            img = m(img, img, img)
            img_sa1 = img[:,:,:c] + img1
            img_sa2 = img[:,:,c:] + img2

        img1 = img_sa1.reshape(batch, h, w, c).transpose(-1, 1)
        img2 = img_sa2.reshape(batch, h, w, c).transpose(-1, 1)

        return img1, img2
