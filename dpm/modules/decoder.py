from torch.nn import TransformerDecoder,Transformer
from torch import nn
class Decoder_only(nn.Module):
    def __init__(self,d_model,num_head,num_layer):
        super.__init__()
        self.decoder_layer=nn.TransformerDecoderLayer(d_model=d_model, nhead=num_head)
        self.num_layer=num_layer
        self.decode=TransformerDecoder(self.decoder_layer,self.num_layer)
    def forward(self,x): #x:[S,N,E]
        return self.decode(x)

class Encoder_Decoder(nn.Module):
    def __init__(self,d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048,dropout=0.1):
        super.__init__()
        self.transformer=Transformer(d_model=d_model,num_decoder_layers=num_encoder_layers,nhead=nhead,
                                     num_encoder_layers=num_decoder_layers,dim_feedforward=dim_feedforward,
                                     dropout=dropout
                                     )
    def forward(self,x):
        return self.transformer(x)
        
     
                 