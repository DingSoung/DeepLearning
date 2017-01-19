# [TensorFlow](https://www.tensorflow.org/how_tos/using_gpu/#supported_devices)

## Environment
1. Python & pip  
    go to [python](https://www.python.org) to install Python, use follow command install pip
    ```bash
    sudo easy_install pip
    ```
2. TensorFlow  
    ```
    sudo pip install tensorflow
    sudo pip install tensorflow-gpu
    ```
3. Anaconda 、Ipython notebook etc.
    go to [continuum](https://www.continuum.io) to install Anaconda
## Basic Usage    
* active ipython:
    ```
    ipython notebook
    ```
* create environment named tersorflow
    ```
    conda create -n tensorflow python=2.7
    ```
    active tensorflow environment
    ```
    source activate tensorflow
    ```
    do some tensorflow work...,
    when finished, close environment
    ```
    source deactivate
    ```
* active tensorboard 
    ```
    python tensorflow/tensorboard/tensorboard.py --logdir=path/to/log-directory`
    ```

## [Exercises](exercises/)

## [tutorial](tutorial/)