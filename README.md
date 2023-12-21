# pic2text
BUPT-NNDL-COURSE-DESIGN

## linux-instruc
using 
```
conda env create -f environment.yaml
conda activate pic2text 
```
finish conda env create

using 
```
python set_vocab.py
```
set vocabulary

using 
```
python main.py fit --config /your/configs/config.yaml
```

train the model

autosave 

autoresume 

using 
```
tensorboard --logdir=your/root/path
```
to see training log

## windows-instruction
using 
```
conda env create -f environment.yaml
conda activate pic2text 
```
finish conda env create

reinstall conda from pytorch offical website, dpending on your CUDA version

you may need to install or update some extra package if you see error

*change your file path in config! they are different from linux*

using 
```
set PYTHONUTF8=1
python main-forwindows.py fit --config /your/configs/config.yaml
```

else same to Linux

