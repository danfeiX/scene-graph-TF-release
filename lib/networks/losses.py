import tensorflow as tf
from net_utils import exp_average_summary


def sparse_softmax(logits, labels, name, loss_weight=1, ignore_bg=False):
    labels = tf.cast(labels, dtype=tf.int32)
    batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    if ignore_bg: # do not penalize background class
        loss_mask = tf.cast(tf.greater(labels, 0), tf.float32)
        batch_loss = tf.mul(batch_loss, loss_mask)
    loss = tf.reduce_mean(batch_loss)
    loss = tf.mul(loss, loss_weight, name=name)
    return loss


def l1_loss(preds, targets, name, target_weights=None, loss_weight=1):
    l1 = tf.abs(tf.sub(preds, targets))
    if target_weights is not None:
        l1 = tf.mul(target_weights, l1)
    batch_loss = tf.reduce_sum(l1, reduction_indices=[1])
    loss = tf.reduce_mean(batch_loss)
    loss = tf.mul(loss, loss_weight, name=name)
    return loss


def total_loss_and_summaries(losses, name):
    total_loss = tf.add_n(losses, name=name)
    losses.append(total_loss)
    total_loss = exp_average_summary(losses, [total_loss],
                                     decay=0.9, name='losses_avg',
                                     scope_pfix='losses/')[0]
    return total_loss


def accuracy(pred, labels, name, ignore_bg=False):
    correct_pred = tf.cast(tf.equal(labels, tf.cast(pred, tf.int32)), tf.float32)
    if ignore_bg:  # ignore background
        mask = tf.cast(tf.greater(labels, 0), tf.float32)
        one = tf.constant([1], tf.float32)
        # in case zero foreground preds
        num_preds = tf.maximum(tf.reduce_sum(mask), one)
        acc_op = tf.squeeze(tf.div(tf.reduce_sum(tf.mul(correct_pred, mask)), num_preds))
    else:
        acc_op = tf.reduce_mean(correct_pred, tf.float32)

    # override the name
    return tf.identity(acc_op, name=name)
