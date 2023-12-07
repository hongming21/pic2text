import torch
import lightning.pytorch as pl


class Pic2TextModel(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 lossconfig,
                 ecconfig,
                 dcconfig,
                 embody_type='patch',
                 ckpt_path=None,
                 ignore_keys=[],
                 input_key="image",
                 gt_key='description_vectors'
                 ):
        super.__init__()
        self.lr=learning_rate
        self.loss=
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        
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
        