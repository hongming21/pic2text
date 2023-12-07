from transformers import Swinv2Model,ViTModel
from torch import nn

class ViTencoder(nn.Module):
    def __init__(self, hidden_size = 768,num_hidden_layers = 12,num_attention_heads = 12,intermediate_size = 3072,
              image_size = 224,
               patch_size = 16,num_channels = 3):
        super.__init__()
        self.model=ViTModel(hidden_size=hidden_size,num_hidden_layers=num_hidden_layers,
                            intermediate_size=intermediate_size,image_size=image_size,patch_size=patch_size
                            )
        
    def forward(self,x):
        return self.model(x)
        
    
    