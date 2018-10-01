# Dataset Preperation
This instruction contains details about how to convert the VisualGenome dataset into a format that can be read by the framework. Alternatively,
you may follow the instruction to download a pre-processed dataset.

## Overview

A dataset for the framework consists of five files:
1. An image database in hdf5 format.
2. An scene graph database in hdf5 format.
3. An scene graph database metadata file in json format.
4. An RoI proposal database in hdf5 format.
5. An RoI distribution for normalizing the bounding boxes.

**Important:** Note that (1) takes ~320GB of space. Hence we recommend creating (1) by yourself and download (2-5). If you just want to test/visualize predictions,
you may download a subset of the processed dataset following the instructions in the next section.
Also note that the framework does not include a regional proposal network implementation. Hence (4) is needed to run the framework.

## Download pre-processed dataset.
You may download the pre-processd full VG dataset using the following links
1. Image database (currently unavailable, please refer to the next section on how to create an IMDB by yourself)
2. Scene graph database: [VG-SGG.h5](http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG.h5)
3. Scene graph database metadata: [VG-SGG-dicts.json](http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG-dicts.json)
4. RoI proposals: [proposals.h5](http://svl.stanford.edu/projects/scene-graph/dataset/proposals.h5)
5. RoI distribution: [bbox_distribution.npy](http://svl.stanford.edu/projects/scene-graph/dataset/bbox_distribution.npy)

**mini-vg:** Alternatively, you may download a subset of the dataset (mini-vg). The dataset contains 1000 images (973 have scene graph annotations) from the test set
of the full VG dataset and no training data. mini-vg takes around 4GB of space uncompressed. Note that you don't have to download it if you have ran the `download.sh` script.

[mini-vg.zip](http://svl.stanford.edu/projects/scene-graph/dataset/mini-vg.zip)

After downloading the dataset, place all files under `data/vg/`. If you use the mini-vg dataset, the directory should look like this:

```
data/vg/mini_imdb_1024.h5
data/vg/mini_proposals.h5
data/vg/mini_VG-SGG.h5
data/vg/mini_VG-SGG-dicts.json
```


## Convert VisualGenome to desired format
(i). To start with, download VisualGenome dataset using the following links:
- Images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
- [Image metadata](http://svl.stanford.edu/projects/scene-graph/VG/image_data.json)
- [VG scene graph](http://svl.stanford.edu/projects/scene-graph/VG/VG-scene-graph.zip)

(ii). Place all the json files under `data_tools/VG/`. Place the images under `data_tools/VG/images`

(iii). Create image database by executing `./create_imdb.sh` in this directory. This script creates a hdf5 databse of images `imdb_1024.h5`. 
The longer dimension of an image is resized to 1024 pixels and the shorter side is scaled accordingly. You may also create a image database of smaller dimension by
editing the size argument of the script. You may skip to (vii) if you chose to downloaded (2-4).

(iv). Create an ROI database and its metadata by executing `./create_roidb.sh` in this directory. The scripts creates a scene graph database file `VG-SGG.h5` and its metadata `VG-SGG-dicts.json`.
By default, the script reads the dimensions of the images from the imdb file created in (iii). If your imdb file is of different size than 512 and 1024, you must add the size to
the `img_long_sizes` list variable in the `vg_to_roidb.py` script.

(v). Use the script provided by py-faster-rcnn to generate (4).

(vi). Change line 93 of `tools/train_net.py` to `True` to generate (5).

(vii). Finally, place (1-5) in `data/vg`.

```
data/vg/imdb_1024.h5
data/vg/bbox_distribution.npy
data/vg/proposals.h5
data/vg/VG-SGG-dicts.json
data/vg/VG-SGG.h5
```
