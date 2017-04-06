import numpy as np
import tensorflow as tf
import roi_pooling_layer.roi_pooling_op as roi_pool_op
import roi_pooling_layer.roi_pooling_op_grad

DEFAULT_PADDING = 'SAME'

"""
A wrapper for TensorFlow network layers
Adapted from SubCNN_TF (https://github.com/yuxng/SubCNN_TF)
"""

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        if name in self.layers:
            print('overriding layer %s!!!!' % name)
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False, load_fc=True):
        data_dict = np.load(data_path).item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                if not load_fc and key.startswith('fc'):
                    print('ignoring fc layers!')
                    continue
                print key
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def wrap(self, tensor, name):
        self.layers[name] = tensor
        return self.feed(tensor)

    @layer
    def stop_gradient(self, input, name):
        return tf.stop_gradient(input, name=name)

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def activation_summary(self, layer_name):
        try:
            layer = self.layers[layer_name]
            self._variable_summaries(layer, layer_name)

        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def _variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            tf.histogram_summary(name, var)

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1,
             trainable=True, reuse=False):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)
            if group==1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)

            if relu:
                bias = tf.nn.bias_add(conv, biases)
                conv_out = tf.nn.relu(bias, name=scope.name)
            else:
                conv_out = tf.nn.bias_add(conv, biases, name=scope.name)

            return conv_out

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def sigmoid(self, input, name):
        return tf.sigmoid(input, name=name)
    @layer
    def tanh(self, input, name):
        return tf.tanh(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        print input

        return roi_pool_op.roi_pool(input[0], input[1],
                              pooled_height,
                              pooled_width,
                              spatial_scale,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            print input_shape
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_biases = tf.constant_initializer(0.0)
            weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
            biases = self.make_var('biases', [num_out], init_biases, trainable)
            if relu:
                fc = tf.nn.relu(tf.matmul(feed_in, weights) + biases, name=scope.name)
            else:
                fc = tf.add(tf.matmul(feed_in, weights), biases, name=scope.name)

            return fc

    @layer
    def softmax(self, input, name):
        input = tf.cast(input, dtype=tf.float64)
        return tf.nn.softmax(input, name=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    @layer
    def argmax(self, input, dim, name):
        return tf.argmax(input, dim, name=name)

    @layer
    def identity(self, input, name):
        return tf.identity(input, name=name)

    @layer
    def batch_norm(self, input, is_training, name, relu=True):
        t = tf.contrib.layers.batch_norm(inputs=input,
                                         center=True,
                                         scale=True,
                                         is_training=is_training,
                                         trainable=True,
                                         scope=name)
        if relu:
            t = tf.nn.relu(t)
        return t
