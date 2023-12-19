from transformers import Swinv2Model,ViTModel,ViTConfig
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
    
        
    
    