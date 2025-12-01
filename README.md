# AISafety

## DiffAttack

**Requirements**

- python 3.8

```
pip install -r DiffAttack/requirements.txt
```


**Train a Cifar classifier**
1. Download [Cifar10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) dataset in `DiffAttack/third_party/pytorch_cifar/data`
```
cd DiffAttack/third_party/pytorch_cifar
python main.py --model_name simpledla
```

**Generate adversarial images**
1. Download Stable Diffusion checkpoint
```
cd DiffAttack
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh Manojb/stable-diffusion-2-base --local-dir pretrained_models/sd2
```

2. Put class sample in `DiffAttack/data`
```
data
├── images
│   ├── 0.png
|   ├── .....
```

2. Infer
```
cd DiffAttack
python main.py --model_name simpledla --dataset_name cifar
```

If mismatching size, add code in `def forward` in `third_party/pytorch_cifar/models/{your desired classifier model}`
```Python
def forward(self, x):
    if x.shape[2] != 32 or x.shape[3] != 32:
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
    # ...
```

3. Extract adversarial images
```
python scripts/resize_adv_images.py --src ./outputs/cifar_simpledla --dst ./outputs/cifar_simpledla_extract/images
```

