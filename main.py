import sys, datetime, time,json,string
import numpy as np
from pathlib import Path
import torch
import torchvision
from PIL import Image,ImageEnhance
from jsonargparse import lazy_instance
from omegaconf import OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI 
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from lightning.pytorch.utilities.rank_zero import rank_zero_only,rank_zero_info
from functools import partial
from torch.utils.data import  DataLoader, Dataset
from dpm.utils import instantiate_from_config

        

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id


    return np.random.seed(np.random.get_state()[1][0] + worker_id)

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,collect_fn=None,
                 wrap=False, num_workers=None,pin_memory=True,prefetch_factor=4,shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.pin_memory=pin_memory
        self.prefetch_factor=prefetch_factor
        self.use_worker_init_fn = use_worker_init_fn
        self.collect_fn=instantiate_from_config(collect_fn).pad
        self.shuffle_val=shuffle_val_dataloader
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
      
        init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          pin_memory=self.pin_memory,
                          worker_init_fn=init_fn,prefetch_factor=self.prefetch_factor,collate_fn=self.collect_fn)

    def _val_dataloader(self, shuffle=False):
       
        init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          worker_init_fn=init_fn,
                          shuffle=self.shuffle_val,prefetch_factor=self.prefetch_factor,collate_fn=self.collect_fn)

    def _test_dataloader(self, shuffle=False):
        init_fn = None
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle,pin_memory=True)

    def _predict_dataloader(self, shuffle=False):
        
        init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, now,logdir):
        super().__init__()
        self.logdir=Path(logdir)
        self.now = now
        self.ckptdir=self.logdir/'checkpoints'
    def on_exception(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = self.ckptdir / "last.ckpt"
            trainer.save_checkpoint(str(ckpt_path))

    def setup(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            self.logdir.mkdir(parents=True, exist_ok=True)
            self.ckptdir.mkdir(parents=True, exist_ok=True)
            
   



class CUDACallback(Callback):

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        #torch.cuda.reset_peak_memory_stats(self.root_gpu(trainer))
        torch.cuda.synchronize(self.root_gpu(trainer))
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(self.root_gpu(trainer))
        max_memory = torch.cuda.max_memory_allocated(self.root_gpu(trainer)) / 2 ** 20
        epoch_time = time.time() - self.start_time

        max_memory = trainer.strategy.reduce(max_memory)
        epoch_time = trainer.strategy.reduce(epoch_time)

        rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
        rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")

    def root_gpu(self, trainer):
        return trainer.strategy.root_device.index

class Image_text_logger(Callback):
    def __init__(self, save_dir,train_batch_frequency,val_batch_frequency, max_log,clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.save_dir=save_dir
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_bacth_freq=val_batch_frequency
        self.max_log = max_log
        self.logger_log_images = {
            TensorBoardLogger:self._testtube
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, data,batch_idx, split):

        
        for k in data:
            if k=="input_img":
                name=f'{split}/{pl_module.current_epoch}/image'
                grid = torchvision.utils.make_grid(data[k])
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,
                pl_module.logger.experiment.add_image(
                        name,grid,
                        pl_module.global_step)
            elif k=='gt_text':
                name=f'{split}/{pl_module.current_epoch}/gt_text'
                for i in range(len(data[k])):
                    pl_module.logger.experiment.add_text(
                        name,data[k][i],pl_module.global_step
                    )
            elif k=='gen_text':
                name=f'{split}/{pl_module.current_epoch}/gen_text'
                for i in range(len(data[k])):
                    pl_module.logger.experiment.add_text(
                        name,data[k][i],pl_module.global_step
                    )

    @rank_zero_only
    def log_local(self, split, data,
                  global_step, current_epoch, batch_idx):
        root = Path(self.save_dir)/'result'/split
        for k in data:
            if k=="input_img":
                grid = torchvision.utils.make_grid(data[k], nrow=4)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "gs-{:06}_e-{:06}_b-{:06}.png".format(
                    
                    global_step,
                    current_epoch,
                    batch_idx)
                path = root/k/filename
                
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                image = Image.fromarray(grid)
                image.save(path)
            
            if k=="gt_text":
                    filename="gs-{:06}_e-{:06}_b-{:06}-gt.txt".format(
                        
                        global_step,
                        current_epoch,
                        batch_idx)
                    path=root/k/filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, 'w', encoding='utf-8') as file:
                        for sentence in data[k]:
                            file.write(sentence + '\n')
            if k=="gen_text":
                    filename="gs-{:06}_e-{:06}_b-{:06}-gen.txt".format(
                        
                        global_step,
                        current_epoch,
                        batch_idx)
                    path=root/k/filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, 'w', encoding='utf-8') as file:
                        for sentence in data[k]:
                            file.write(sentence + '\n')
                
    def log_img_and_text(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx,split) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_image_and_text") and
                callable(pl_module.log_image_and_text) and
                self.max_log > 0):
           
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                log = pl_module.log_image_and_text(batch, **self.log_images_kwargs)

            for k in log:
                if k=='input_img':
                    N = min(log[k].shape[0], self.max_log)
                    log[k] = log[k][:N]
                    if isinstance(log[k], torch.Tensor):
                        log[k] = log[k].detach().cpu()
                        if self.clamp:
                            log[k] = torch.clamp(log[k], -1., 1.)
                else:
                    N = min(len(log[k]), self.max_log)
                    log[k] = log[k][:N]
            data=log
            self.log_local( split, data,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module,data, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx,split):
        if split=='train':
            if ((check_idx % self.train_batch_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
        elif split=='val':
            if ((check_idx % self.val_bacth_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx='train_dataloader'):
        
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
           
            self.log_img_and_text(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx='val_dataloader'):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img_and_text(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)



   
if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logdir=Path('logs')
    sys.path.append(Path.cwd())
    img_logger_callback=Image_text_logger(save_dir=str(logdir/now),train_batch_frequency=100,val_batch_frequency=50,
                                          max_log=2)
    cuda_callback=CUDACallback()
    lr_callback=LearningRateMonitor(logging_interval='step')
    model_ckpt_callback=ModelCheckpoint( 
                    dirpath=str(logdir/now/'checkpoints'),
                    monitor='val/loss',
                    verbose= True,
                    filename='{epoch:02d}-{val_loss:.2f}',
                    save_top_k=3,
                    mode='min',
                    save_last=True
    )
    
    cli=LightningCLI(
        trainer_defaults={
            'logger':lazy_instance(TensorBoardLogger,save_dir=logdir,name=now,version=0),
            'callbacks':[img_logger_callback,cuda_callback,lr_callback,model_ckpt_callback]
        }
    )