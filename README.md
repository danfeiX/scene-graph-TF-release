# Scene Graph Generation by Iterative Message Passing

![Scene Graph prediction samples](http://cs.stanford.edu/~danfei/scene-graph/preview.jpeg "Sample predictions")

## About this repository
This repository contains an implementation of the models introduced in the paper [Scene Graph Generation by Iterative Message Passing](https://arxiv.org/abs/1701.02426) by Xu et al. The model taks an image and a graph proposal as input and predicts the object and relationship categories in the graph. The network  is implemented using [TensorFlow](https://www.tensorflow.org/) and the rest of the framework is in Python. Because the model is built directly on top of [Faster-RCNN by Ren et al](https://arxiv.org/abs/1506.01497), a substantial amount of data processing code is adapted from the [py-faster-rcnn repository](https://github.com/rbgirshick/py-faster-rcnn). 


## Citing this work
If you find this work useful in your research, please consider citing:
```
@inproceedings{xu2017scenegraph,
  title={Scene Graph Generation by Iterative Message Passing},
  author={Xu, Danfei and Zhu, Yuke and Choy, Christopher and Fei-Fei, Li},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
 }
```

## Project page
The project page is available at [http://cs.stanford.edu/~danfei/scene-graph/](http://cs.stanford.edu/~danfei/scene-graph/).


## Prerequisites
1. The framework does not include a regional proposal network implementation. A RoI proposal database pre-extracted using the py-faster-rcnn framework is available for download.
2. You need CUDA-compatible GPUs to run the framework. A CPU-compatible version will be released soon.
3. You need at least 320 GB of free space to store the processed VisualGenome image dataset. A training script that reads image files directly will be released in the future.
However, if you just want to test/visualize some sample predictions, you may download a subset of the processed dataset (mini-vg) following the [instruction](data_tools/) or the "Quick Start" section. The subset takes ~4GB of space.

## Dependencies
To get started with the framework, install the following dependencies:
- Python 2.7
- [TensorFlow r0.11](https://www.tensorflow.org/versions/r0.11/get_started/os_setup#download-and-setup)
- [h5py](http://www.h5py.org/)
- [numpy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scipy](https://www.scipy.org/)
- [pyyaml](https://pypi.python.org/pypi/PyYAML)
- [easydict](https://pypi.python.org/pypi/easydict/)
- [cython](http://cython.org/)
- [graphviz](https://pypi.python.org/pypi/graphviz) (optional, if you wish to visualize the graph structure)
- CUDA 7.5/8.0 (required, CPU-only version will be released soon)

1. It is recommended that you install everything in an [Anaconda](https://www.continuum.io/downloads) environment, i.e., install Anaconda and run

    ```
    conda create -n sence-graph python=2.7
    source activate scene-graph
    ```

2. Run `pip install -r requirements.txt` to install all the requirement except TensorFlow and CUDA. Follow the provided [URL](https://www.tensorflow.org/versions/r0.11/get_started/os_setup#download-and-setup) to install TensorFlow r0.11 (0.10 and 0.12 also works).

The code has not been tested on TensorFlow 1.0 and above, but may potentially work once you convert all TF-related code using the offical transition [script](https://www.tensorflow.org/install/migration).

## Compile libraries

1. After you have installed all the dependencies, run the following command to compile nms and bbox libraries.
```
cd lib
make
```

2. Follow this [this instruction](lib/roi_pooling_layer/) to see if you can use the pre-compiled roi-pooling custom op or have to compile the op by yourself.



## Quick Start
1. Make sure you have installed all the dependencies and compiled the libraries.
2. Run the `download.sh` script to download the mini-vg dataset and a model checkpoint.

    `./download.sh`

3. Run the following command to visualize a predicted scene graph. Set `GPU_ID` to the ID of the GPU you want to use, e.g. `0`.

    ```
    ./experiments/scripts/test.sh mini-vg -1 \
                                  dual_graph_vrd_final 2 \
                                  checkpoints/dual_graph_vrd_final_iter2.ckpt \
                                  viz_cls \
                                  GPU_ID
    ```


## Dataset
The scene graph dataset used in the paper is the [VisualGenome dataset](https://visualgenome.org), although the framework can work with any scene graph dataset if converted to the desired format. Please refer to the [dataset README](data_tools/) for further instructions on converting the VG dataset into the desired format or downloading pre-processed datasets.

## Train a model
Follow the following steps to train a model:
1. Prepare or download the [full dataset](data_tools/).
2. Download a [Faster-RCNN model](http://cvgl.stanford.edu/scene-graph/dataset/coco_vgg16_faster_rcnn_final.npy) pretrained on the MS-COCO dataset and save the model to `data/pretrained/`.
3. Edit the training script `experiments/scripts/train.sh` such that all paths agree with the files on your file system.
4. To train the final model with inference iterations = 2, run:

   `./experiments/scripts/train.sh dual_graph_vrd_final 2 CHECKPOINT_DIRECTORY GPU_ID`

The program saves a checkpoint to `checkpoints/CHECKPOINT_DIRECTORY/` every 50000 iterations. Training a full model on a desktop with Intel i7 CPU, 64GB memory, and a TitanX graphics card takes around 20 hours. You may use tensorboard to visualize the training process. By default, the tf log directory is set to `checkpoints/CHECKPOINT_DIRECTORY/tf_logs/`.

## Evaluate a model
Follow the following steps to evaluate a model:
1. Prepare or download [the full dataset or the mini-vg dataset](data_tools/).
2. If you wish to evaluate a pre-trained model, first download a checkpoint in the "Checkpoints" section.
3. Edit the evaluation script `experiments/scripts/test.sh` such that all paths agree with the files on your file system.
4. To evaluate the final model with inference iterations = 2 using 100 images in the test set of the full VG dataset (use mini-vg for the [mini VG dataset](data_tools/)), run

    `./experiments/scripts/test.sh vg 100 dual_graph_vrd_final 2 CHECKPOINT_PATH(.ckpt) all GPU_ID`

Note that to reproduce the results as presented the paper, you have to evaluate the entire test set by setting the number of images to -1.
The evaluation process takes around 10 hours. Setting the evaluation mode to `all` is to evaluate the models on all three tasks, i.e., `pred_cls, sg_cls, sg_det`.
You can also set the evaluation mode to individual tasks.

## Visualize a predicted scene graph
Follow the following steps to visualize a scene graph predicted by the model:
1. Prepare or download [the full dataset or the mini-vg dataset](data_tools/).
2. If you wish to evaluate a pre-trained model, first download a checkpoint in the "Checkpoints" section.
3. Edit the evaluation script `experiments/scripts/test.sh` such that all paths agree with the files on your file system.
4. To visualize the predicted graph of the first 100 images in the test set of the full VG dataset (use mini-vg for the [mini VG dataset](data_tools/)), run

    `./experiments/scripts/test.sh vg 100 dual_graph_vrd_final 2 CHECKPOINT_PATH(.ckpt) viz_cls GPU_ID`

The `viz_cls` mode assumes ground truth bounding boxes and predicts the predicted object and relationship labels, which is of the same setting as the `sg_cls` task. 
`viz_det` mode uses the proposed bounding box from the regional proposal network as the object proposals, which is of the same setting as the `sg_det` task.

## Checkpoints
A TensorFlow checkpoint of the final model trained with 2 inference iterations:

[dual_graph_vrd_final_iter2_checkpoint.zip](http://cvgl.stanford.edu/scene-graph/checkpoints/dual_graph_vrd_final_iter2_checkpoint.zip)

## License
MIT License
