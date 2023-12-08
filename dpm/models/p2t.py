from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import lightning.pytorch as pl
from dpm.utils import instantiate_from_config
from dpm.evaluation import compute_meteor_score,compute_rouge_score
class Pic2TextModel(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 lossconfig,
                 ecconfig,
                 dcconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 input_key="image",
                 gt_key='description_vectors'
                 ):
        super.__init__()
        self.lr=learning_rate
        self.loss=instantiate_from_config(lossconfig)
        self.encoder=instantiate_from_config(ecconfig)
        self.decoder=instantiate_from_config(dcconfig)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.input_key=input_key
        self.gt_key=gt_key
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    def get_data(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            return x
        elif len(x.shape) ==4:
            return x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
    
    def forward(self, input,target):
        i=self.get_data(input,self.input_key)
        gt=self.get_data(input,self.gt_key)
        hidden=self.encoder(i)
        output=self.decoder(target,hidden)
        
        return output
    
    def configure_optimizers(self):
        lr = self.lr
        opt = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))

        return opt
    def training_step(self, batch,batch_idx) -> STEP_OUTPUT:
        inputs = self.get_data(batch, self.input_key)
        gt=self.get_data(batch,self.gt_key)

        output=self(gt[:,:-1,:],inputs)
        loss=self.loss(output[:,1:,:],gt[:,:-1,:]) #teacher forcing
        self.log('train/loss',loss,on_step=True,on_epoch=True,prog_bar=True)
    
    def validation_step(self, batch, batch_indx) :
        inputs = self.get_data(batch, self.input_key)
        gt=self.get_data(batch,self.gt_key)
        <eos>
        output=self(inputs,<eos>)
        loss=self.loss(output,gt)
        self.log('val/loss',loss,on_step=True,on_epoch=True)
        rouge=compute_rouge_score(gt,output)
        meteur=compute_meteor_score(gt,output)
        self.log('val/rouge-l',rouge,on_epoch=True)
        self.log('val/meteor',meteur,on_epoch=True)
        
        