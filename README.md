# MCnet

This is the code for the ICLR 2017 paper [Decomposing Motion and Content for Natural Video Sequence Prediction](https://openreview.net/pdf?id=rkEFLFqee) by Ruben Villegas, Jimei Yang, Seunghoon Hong, Xunyu Lin and Honglak Lee.

Please follow the instructions to run the code.

## Requirements
MCnet works with
* Linux
* NVIDIA Titan X GPU
* Tensorflow version 1.1.0

## Installing Dependencies (Anaconda installation is recommended)
* pip install scipy
* pip install imageio
* pip install pyssim
* pip install joblib
* pip install Pillow
* pip install scikit-image
* pip install opencv-python
* pip install pytube
* pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp27-none-linux_x86_64.whl
* sudo apt-get install unrar

FFMPEG needs to be installed as well to generate gif videos.
If using anaconda, ffmpeg can be installed as follows:
* conda install -c menpo ffmpeg=3.1.3

## Downloading Data
* KTH
```
./data/KTH/download.sh
```
* UCF101
```
./data/UCF101/download.sh
```
* Sporst1M (will take a very long time --> 70,000 videos are downloaded only)
```
./data/S1M/download.sh
```

## Downloading Pre-trained models
* Download the models trained on KTH and Sports1M
```
./models/paper_models/download.sh
```

## KTH training/testing
* Training:
```
python train_kth.py --gpu=0 --batch_size=8 --K=10 --T=10 --alpha=1.0 --beta=0.02
```

* Testing with model used in paper:
```
python test_kth.py --gpu=0 --prefix=paper_models
```

* Testing with model from training command above:
```
python test_kth.py --gpu=0 --prefix=KTH_MCNET_image_size=128_K=10_T=10_batch_size=8_alpha=1.0_beta=0.02_lr=0.0001
```


## Sports1M training / UCF101 testing
* Training:
```
python train_s1m.py --gpu=0 --batch_size=8 --K=4 --T=1 --alpha=1.0 --beta=0.001
```

* Testing with model used in paper:
```
python test_ucf101.py --gpu=0 --prefix=paper_models
```

* Testing with model from training command above:
```
python test_ucf101.py --gpu=0 --prefix=S1M_MCNET_K=4_T=1_batch_size=8_alpha=1.0_beta=0.001_lr_0.0001
```


## Results
The generated gifs will be located in
```
./results/images/<dataset>
```

The quantative results will be in
```
./results/quantitative/<dataset>
```
The quantitative results for each video will be stored as dictionaries, and the mean results for all test data instances at every timestep can be displayed as
```
import numpy as np
results = np.load('<results_file_name>')
print(results['psnr'].mean(axis=0))
print(results['ssim'].mean(axis=0))
```

## Citation

If you find this useful, please cite our work as follows:
```
@article{villegas17mcnet,
  author = {Ruben Villegas and Jimei Yang and Seunghoon Hong and Xunyu Lin and Honglak Lee},
  title = {Decomposing Motion and Content for Natural Video Sequence Prediction},
  journal = {ICLR},
  year = {2017},
}
```

Please contact "ruben.e.villegas@gmail.com" if you have any questions.
