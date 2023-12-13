Continual and lifelong learning in longitudinal neuroimaging datasets.

## Installation
I recommend creating an Anaconda environment and setting it up as follows:

```bash
conda create -n lln python=3.9.0
pip install torch=2.0.0 torchvision torchaudio
pip install -r requirements.txt
```

## Setting paths
Inside *lln*, create a directory called *local* with a file *paths.py* where you specify your data and output paths.