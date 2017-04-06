# RoIPooling Layer

This is a RoIPooling layer implementation adapted from SubCNN\_TF (https://github.com/yuxng/SubCNN_TF).

It's pre-compiled with CUDA7.5 (`roi_pooling_op_gpu.so`) and CUDA8.0 (`roi_pooling_op_gpu_cuda8.so`) on a Ubuntu 16.04 system. Change to the verison you need in `roi_pooling_op.py`. If you find the compiled library is not compatible with your system, you should compile the op by yourself.

Test if you can load the custom op by the following command before proceed:

`python -c 'import tensorflow as tf; tf.load_op_library("roi_pooling_op_gpu.so")' # cuda 7.5`

`python -c 'import tensorflow as tf; tf.load_op_library("roi_pooling_op_gpu_cuda8.so")' # cuda 8.0`

If it gives you a "NotFoundError" error, specify the full path to the `.so` file. If it does not give you any error, you may proceed with the main instruction.

## Compile RoIPooling layer by yourself
Generally you can follow the [official guide](https://www.tensorflow.org/extend/adding_an_op) to compile a custom op.
Here we provide an instruction on how to compile the `roi_pooling` op specifically.
1. Move everything under `src/` to `YOUR_TENSORFLOW_PATH/lib/python2.7/site-packages/tensorflow/core/user_ops`. You can find out your tensorflow path by running

    `python -c 'import tensorflow as tf; print(tf.__file__)'`

2. `cd YOUR_TENSORFLOW_PATH/lib/python2.7/site-packages/tensorflow/core/user_ops/`
3. Run the following command to compile a GPU-capable RoI-Pooling layer

```
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
nvcc -std=c++11 -c -o roi_pooling_op_gpu.cu.o roi_pooling_op_gpu.cu.cc -I \
    $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared -o roi_pooling_op_gpu.so roi_pooling_op.cc \
    roi_pooling_op_gpu.cu.o -I $TF_INC -fPIC -lcudart
```

Note that if your CUDA is not installed in the default location, you have to specify the path by adding a `-L YOUR_CUDA_PATH/lib64/` flag in the last command.
For example, if your CUDA is under `/usr/local/cuda-8.0/`, you should run 
```
g++ -std=c++11 -shared -o roi_pooling_op_gpu.so roi_pooling_op.cc \
    roi_pooling_op_gpu.cu.o -I $TF_INC -fPIC -L /usr/local/cuda-8.0/lib64/ -lcudart
```

4. Test if you can load the library by running 

    `python -c 'import tensorflow as tf; tf.load_op_library('roi_pooling_op_gpu.so')'`.

5. Move the `roi_pooling_op_gpu.so` file back to your `roi_pooling_layer` directory.
