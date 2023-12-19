from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from dpm.utils import instantiate_from_config
from dpm.evaluation import compute_meteor_score,compute_rouge_score
class Pic2TextModel(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 lossconfig,
                 ecconfig,
                 dcconfig,
                 vocabulary_fig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 input_key="image",
                 gt_key='description_vectors',
                 gt_text='describtion'
                 
                 ):
        super.__init__()
        self.lr=learning_rate
        self.loss=instantiate_from_config(lossconfig)
        self.encoder=instantiate_from_config(ecconfig)
        self.decoder=instantiate_from_config(dcconfig)
        self.vocabular=instantiate_from_config(vocabulary_fig)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.input_key=input_key
        self.gt_key=gt_key
        self.text_key=gt_text
        self.embed=torch.nn.Embedding(num_embeddings=len(self.vocabular),embedding_dim=embed_dim,padding_idx=0)
        self.output_layer = torch.nn.Linear(embed_dim,len(self.vocabular))
        
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
        elif len(x.shape) ==4 and x.shape[1]!=3:
            return x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
    
    def forward(self, input,target):
        src=input
        tgt=self.embed(target)
        hidden=self.encoder(src)
        output=self.decoder(tgt,hidden)
        logits = self.output_layer(output)
        return logits.argmax(dim=-1)
    
    def configure_optimizers(self):
        lr = self.lr
        opt = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))

        return opt
    def training_step(self, batch,batch_idx) -> STEP_OUTPUT:
        inputs = self.get_data(batch, self.input_key)
        gt=self.get_data(batch,self.gt_key)

        output=self(inputs,gt[:,:-1])
        loss=self.loss(gt[:,1:],output) #teacher forcing
        self.log('train/loss',loss,on_step=True,on_epoch=True,prog_bar=True)
    
    def validation_step(self, batch, batch_indx) :
        inputs = self.get_data(batch, self.input_key)
        gt = self.get_data(batch, self.gt_key)
        output=self.greedy_search(inputs,gt)
        # 计算损失
        loss = self.loss(output, gt)
        self.log('val/loss', loss, on_step=True, on_epoch=True)

        # 计算评价指标
        rouge = compute_rouge_score(gt, output)
        meteor = compute_meteor_score(gt, output)
        self.log('val/rouge-l', rouge, on_epoch=True)
        self.log('val/meteor', meteor, on_epoch=True)
    def greedy_search(self,inputs,gt):
        sos_batch = torch.full((inputs.shape[0], 1), 1, dtype=torch.long, device=inputs.device)

        # 初始化解码器的输入为 SOS
        decoder_input = sos_batch
        max_length = gt.size(1)  # 假设我们希望生成长度与 gt 一致的序列

        # 逐步生成序列
        output = []
        eos_reached = False
        for _ in range(max_length):
            decoder_output = self(inputs, decoder_input)
            next_token = decoder_output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            if next_token.item() == 2:  # 假设 2 是 EOS 的索引
                eos_reached = True
            
            if eos_reached:
                next_token.fill_(0)  # 用 PAD (0) 填充

            decoder_input = torch.cat([decoder_input, next_token], dim=-1)
            output.append(next_token)

        output = torch.cat(output, dim=1)
        return output
        
    @rank_zero_only
    def log_image_and_text(self,batch):
        image=self.get_data(batch,self.input_key)
        gt_text=self.get_data(batch,self.gt_text)
        gt_index=self.get_data(batch,self.gt_key)
        log = dict()
        index_output=self.greedy_search(image,gt_index)
        text_output=self.batch_int_sequence_to_text(index_output)
        log['input_img']=image
        log["gt_text"] = gt_text
        log["gen_text"]=text_output
        return log
    
    def batch_int_sequence_to_text(batch_int_sequences, vocabulary, special_tokens_indexes):
    # 反转词汇表映射：从整数到单词
        index_to_word = {index: word for word, index in vocabulary.items()}

        sentences = []
        for int_sequence in batch_int_sequences:
            # 转换每个整数序列为单词序列并过滤特殊词汇
            words = [index_to_word.get(index, "") for index in int_sequence if index not in special_tokens_indexes]

            # 将单词序列组合为句子
            sentence = ' '.join(words)
            sentences.append(sentence)

        return sentences

            
            