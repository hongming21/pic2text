from torch.nn import TransformerDecoder,Transformer
from torch import nn
import torch
import math
class Decoder_only(nn.Module):
    def __init__(self, d_model, num_head, num_layer):
        super().__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_head)
        self.decode = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layer)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, gt, x):
        x = x.permute(1, 0, 2) #[S,N,E]
        tgt_mask = self.generate_square_subsequent_mask(gt.size(0))
        output=self.decode(gt, x, tgt_mask=tgt_mask)
        return output.permute(1,0,2)

class Encoder_Decoder(nn.Module):
    def __init__(self,d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048,dropout=0.1):
        super.__init__()
        self.transformer=Transformer(d_model=d_model,num_decoder_layers=num_encoder_layers,nhead=nhead,
                                     num_encoder_layers=num_decoder_layers,dim_feedforward=dim_feedforward,
                                     dropout=dropout
                                     )
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self,gt,x):
        x=x.permute(1,0,2)
        tgt_mask = self.generate_square_subsequent_mask(gt.size(0))
        output=self.transformer(src=x,tgt=gt,tgt_mask=tgt_mask)
        return output.permute(1,0,2)
        
     
                 