from transformers import Swinv2Model,ViTModel,ViTConfig,Swinv2Config
from torch import nn

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
                image_size,
                patch_size, 
                num_channels,
                embed_dim,
                depths,
                num_heads,
                window_size,
                mlp_ratio,
                qkv_bias,
                hidden_dropout_prob,
                attention_probs_dropout_prob,
                drop_path_rate,
                hidden_act,
                use_absolute_embeddings,
                initializer_range,
                layer_norm_eps,
                encoder_stride,
        )
        self.model=Swinv2Model(self.config,add_pooling_layer = True)
        self.encode_type=type
    def forward(self,x):
        if self.encode_type=='patch':
            return self.model(x).last_hidden_state
        elif self.encode_type=='all':
            pooler_output = self.model(x).pooler_output.unsqueeze(1)  # 维度变为 [batch_size, 1, hidden_size]
            return pooler_output