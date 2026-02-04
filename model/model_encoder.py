import torch
from torch import nn
import torchvision.models as models
from einops import rearrange
import clip
import os
import torch.nn.functional as F

class ClipEncoder(nn.Module):
    def __init__(self, path = "/content/CLIPCap/RemoteCLIP-ViT-B-32.pt"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model, preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        except AttributeError:
            pass 

        # ResNet (ImageNet) İstatistikleri
        self.register_buffer('source_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('source_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # CLIP (OpenAI) İstatistikleri
        self.register_buffer('target_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('target_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        jit_path = path

        if os.path.exists(jit_path):
            print(f"JIT Model yükleniyor: {jit_path}")
            model_jit = torch.jit.load(jit_path, map_location=self.device).eval()
            jit_state_dict = model_jit.state_dict()
            
            # Fazlalık anahtarları temizle
            keys_to_ignore = []
            clean_state_dict = {k: v for k, v in jit_state_dict.items() if k not in keys_to_ignore}
            
            # Ağırlıkları yükle
            self.model.load_state_dict(clean_state_dict, strict=True)
            print("Ağırlıklar başarıyla transfer edildi!")
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

        # 1. Denormalize (ResNet -> Raw [0,1])
        img = img * self.source_std + self.source_mean
        
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

    def forward(self, imageA, imageB):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        feat1 = self.model(imageA)  # (batch_size, 2048, image_size/32, image_size/32)
        feat2 = self.model(imageB)

        return feat1, feat2

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

    def forward(self, x1, x2, x3):
        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_k(x3)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

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

    def forward(self, x1, x2, x3):
        if self.norm_first:
            x = self.att(self.norm1(x1), self.norm1(x2), self.norm1(x3)) + x1
            x = self.feedforward(self.norm2(x)) + x
        else:
            x = self.norm1(self.att(x1, x2, x3) + x1)
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