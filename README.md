# Mask R-CNN for Object Detection and Segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

![Instance Segmentation Sample](assets/street.png)

## Requirements
Python 3.6.9 and other common packages listed in `requirements.txt`. **Do not install different versions of tensorflow and keras.**

## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run train.py file
   ```bash
   python3 train.py
   ```

# Getting Started
* Download the [tshirt dataset .zip file](https://drive.google.com/file/d/1aGLstfgFbBMZih_OaBaiuB40ZI8DxLpK/view?usp=sharing) from here and extract it. Annotations file is inside the folder with name ``` annotations_train.json ```

* ([model.py](mrcnn/model.py), [utils.py](mrcnn/utils.py), [config.py](mrcnn/config.py)): These files contain the main Mask RCNN implementation. 

* ([train.py](train.py)): To train the model use this file. Use the same path for _dataset_train_ and _dataset_val_ in **train.py** as we are using 10% of images from training dataset for measuring validation loss.

* ([train_mask_rcnn_demo.py](demo/train_mask_rcnn_demo.py)) is used give no. of classes, batch size, steps_per_epoch, epochs, gpu_count, images_per_gpu. Change them as per your convinence wrt your GPU.  You can change epochs in ```train_head``` and ```train_all_layers``` functions in `line267`  and `line274` respectively.

* For Downloading the dataset 

pip install gdown 
 ```gdown --fuzzy https://drive.google.com/file/d/1aGLstfgFbBMZih_OaBaiuB40ZI8DxLpK/view?usp=sharing```


