import sys, datetime, time
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
from pl_bolts.callbacks import PrintTableMetricsCallback
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from functools import partial
from torch.utils.data import  DataLoader, Dataset
from mri_ldm.util import instantiate_from_config

        

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
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None,pin_memory=True,prefetch_factor=4,shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.pin_memory=pin_memory
        self.prefetch_factor=prefetch_factor
        self.use_worker_init_fn = use_worker_init_fn
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
                          worker_init_fn=init_fn,prefetch_factor=self.prefetch_factor)

    def _val_dataloader(self, shuffle=False):
       
        init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          worker_init_fn=init_fn,
                          shuffle=shuffle,prefetch_factor=self.prefetch_factor)

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
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs=None,dataloader_idx='train_dataloader'):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            

            print(f"Average Epoch time: {epoch_time:.2f} seconds",rank_zero_only=True)
            print(f"Average Peak memory {max_memory:.2f}MiB",rank_zero_only=True)
        except AttributeError:
            pass


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logdir=Path('logs')
    sys.path.append(Path.cwd())
    cuda_callback=CUDACallback()
    printcall=PrintTableMetricsCallback()
    lr_callback=LearningRateMonitor(logging_interval='step')
    model_ckpt_callback=ModelCheckpoint( 
                    dirpath=str(logdir/now/'checkpoints'),
                    monitor='val_loss',
                    verbose= True,
                    filename='{epoch:02d}-{val_loss:.2f}',
                    save_top_k=3,
                    mode='min',
                    save_last=True
    )
    
    cli=LightningCLI(
        trainer_defaults={
            'logger':lazy_instance(TensorBoardLogger,save_dir=logdir,name=now,version=0),
            'callbacks':[printcall,cuda_callback,lr_callback,model_ckpt_callback]
        }
    )
    
    