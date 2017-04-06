import tensorflow as tf
import numpy as np
import roi_pooling_op_grad

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

array = np.random.rand(1, 10, 10, 1)
data = tf.convert_to_tensor(array, dtype=tf.float32)
rois = tf.convert_to_tensor([[0, 1, 1, 2, 2], [0, 3, 3, 4, 4]], dtype=tf.float32)

W = weight_variable([3, 3, 1, 1])
h = conv2d(data, W)

module = tf.load_op_library(tf.sysconfig.get_lib() + '/user_ops/roi_pooling_op_gpu.so')
print dir(module)
[y, argmax] = module.roi_pool(h, rois, 1, 1, 1.0/1)
y_data = tf.convert_to_tensor(np.ones((2, 1, 1, 1)), dtype=tf.float32)
print y_data, y, argmax

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)

for step in xrange(101):
    sess.run(train)
    print(step, sess.run(W))

#with tf.device('/gpu:0'):
#  result = module.roi_pool(data, rois, 1, 1, 1.0/1)
#  print result.eval()
#with tf.device('/cpu:0'):
#  run(init)
