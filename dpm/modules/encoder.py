from transformers import Swinv2Model,ViTModel,ViTConfig,Swinv2Config
from torch import nn
import torch
from torchvision.models import resnet50,resnet101
class ViTencoder(nn.Module):
    def __init__(self, hidden_size = 768,num_hidden_layers = 12,num_attention_heads = 12,intermediate_size = 3072,
              image_size = 224,patch_size = 16, num_channels = 3,type='patch'):
        super().__init__()
        self.config=ViTConfig(hidden_size=hidden_size,num_hidden_layers=num_hidden_layers,num_attention_heads =num_attention_heads,
                            intermediate_size=intermediate_size,image_size=image_size,patch_size=patch_size
                            )
        self.model=ViTModel(self.config,add_pooling_layer=True)
        self.encode_type=type
    def forward(self,x):
        if self.encode_type=='patch':
            return self.model(x).last_hidden_state
        elif self.encode_type=='all':
            pooler_output = self.model(x).pooler_output.unsqueeze(1)  # 维度变为 [batch_size, 1, hidden_size]
            return pooler_output
    
        
class Swinv2encoder(nn.Module):
    def __init__(self, 
                image_size=224,
                patch_size=4,
                num_channels=3,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4.0,
                qkv_bias=True,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                drop_path_rate=0.1,
                hidden_act="gelu",
                use_absolute_embeddings=False,
                initializer_range=0.02,
                layer_norm_eps=1e-5,
                encoder_stride=32,
                type='patch'):
        super().__init__()
        self.config=Swinv2Config(

                image_size=image_size,
                patch_size=patch_size, 
                num_channels=num_channels,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                drop_path_rate=drop_path_rate,
                hidden_act=hidden_act,
                use_absolute_embeddings=use_absolute_embeddings,
                initializer_range=initializer_range,
                layer_norm_eps=layer_norm_eps,
                encoder_stride=encoder_stride
        )
        self.model=Swinv2Model(self.config,add_pooling_layer = True)
        self.encode_type=type
    def forward(self,x):
        if self.encode_type=='patch':
            return self.model(x).last_hidden_state
        elif self.encode_type=='all':
            pooler_output = self.model(x).pooler_output.unsqueeze(1)  # 维度变为 [batch_size, 1, hidden_size]
            return pooler_output
        
class VGG19(nn.Module):
    def __init__(self, embed_dim=512):
        super(VGG19, self).__init__()
        # VGG-19 layers
        self.features = nn.Sequential(
            # Convolutional layers block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional layers block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional layers block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional layers block 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional layers block 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=embed_dim, kernel_size=1)

        )
    def forward(self, x):
        x = self.features(x)
        b=x.shape[0]
        c=x.shape[1]
        x = x.transpose(1,3).contiguous().reshape(b,-1,c)
        return x


class ResNet101Encoder(nn.Module):
    def __init__(self,embed_dim=512):
        super().__init__()

        # 预训练的 ResNet-101 特征提取模型
        self.feature_extractor = resnet101(pretrained=True)
        layers = list(self.feature_extractor.children())[:-2]
        self.feature_extractor = nn.Sequential(*layers)
        # 用于调整特征向量大小的线性层
        self.resize_features = nn.Conv2d(2048,embed_dim, 1)  # 1x1 卷积

    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)  # 特征图维度为 (B, 2048, H, W)
        
        # 调整特征向量大小
        resized_features = self.resize_features(features)  # 输出维度 (B, 512, H, W)

        # 重新排列维度以匹配所需输出格式 (B, H*W, 512)
        bs, c, h, w = resized_features.shape
        output = resized_features.view(bs, c, h*w).permute(0, 2, 1)

        return output