import tensorflow as tf

def exp_average_summary(ops, dep_ops, decay=0.9, name='avg', scope_pfix='',
                        raw_pfix=' (raw)', avg_pfix=' (avg)'):
    averages = tf.train.ExponentialMovingAverage(decay, name=name)
    averages_op = averages.apply(ops)

    for op in ops:
        tf.scalar_summary(scope_pfix + op.name + raw_pfix, op)
        tf.scalar_summary(scope_pfix + op.name + avg_pfix,
                          averages.average(op))

    with tf.control_dependencies([averages_op]):
        for i, dep_op in enumerate(dep_ops):
            dep_ops[i] = tf.identity(dep_op, name=dep_op.name.split(':')[0])

    return dep_ops

def exp_average(vec, curr_avg, decay=0.9):
    vec_avg = tf.reduce_mean(vec, 0)
    avg = tf.assign(curr_avg, curr_avg * decay + vec_avg * (1-decay))
    return avg

def gather_vec_pairs(vecs, gather_inds):
    """
    gather obj-subj feature pairs
    """
    vec_pairs = tf.gather(vecs, gather_inds)
    vec_len = int(vec_pairs.get_shape()[2]) * 2
    vec_pairs = tf.reshape(vec_pairs, [-1, vec_len])
    return vec_pairs

def pad_and_gather(vecs, mask_inds, pad=None):
    """
    pad a vector with a zero row and gather with input inds
    """
    if pad is None:
        pad = tf.expand_dims(tf.zeros_like(vecs[0]), 0)
    else:
        pad = tf.expand_dims(pad, 0)
    vecs_padded = tf.concat(0, [vecs, pad])
    # flatten mask and edges
    vecs_gathered = tf.gather(vecs_padded, mask_inds)
    return vecs_gathered

def padded_segment_reduce(vecs, segment_inds, num_segments, reduction_mode):
    """
    Reduce the vecs with segment_inds and reduction_mode
    Input:
        vecs: A Tensor of shape (batch_size, vec_dim)
        segment_inds: A Tensor containing the segment index of each
        vec row, should agree with vecs in shape[0]
    Output:
        A tensor of shape (vec_dim)
    """
    if reduction_mode == 'max':
        print('USING MAX POOLING FOR REDUCTION!')
        vecs_reduced = tf.segment_max(vecs, segment_inds)
    elif reduction_mode == 'mean':
        print('USING AVG POOLING FOR REDUCTION!')
        vecs_reduced = tf.segment_mean(vecs, segment_inds)
    vecs_reduced.set_shape([num_segments, vecs.get_shape()[1]])
    return vecs_reduced
