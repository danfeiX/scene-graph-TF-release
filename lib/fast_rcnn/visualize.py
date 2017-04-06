# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

"""
Visualize a generated scene graph
"""

from fast_rcnn.config import cfg
from roi_data_layer.roidb import prepare_roidb
from fast_rcnn.test import im_detect, gt_rois, non_gt_rois
from datasets.viz import viz_scene_graph, draw_scene_graph
from datasets.eval_utils import ground_predictions
from networks.factory import get_network
import numpy as np
import tensorflow as tf
from utils.cpu_nms import cpu_nms
import matplotlib.pyplot as plt

import json


def draw_graph_pred(im, boxes, cls_score, rel_score, gt_to_pred, roidb):
    """
    Draw a predicted scene graph. To keep the graph interpretable, only draw
    the node and edge predictions that have correspounding ground truth
    labels.
    args:
        im: image
        boxes: prediceted boxes
        cls_score: object classification scores
        rel_score: relation classification scores
        gt_to_pred: a mapping from ground truth box indices to predicted box indices
        idx: for saving
        roidb: roidb
    """
    gt_relations = roidb['gt_relations']
    im = im[:, :, (2, 1, 0)].copy()
    cls_pred = np.argmax(cls_score, 1)
    rel_pred_mat = np.argmax(rel_score, 2)
    rel_pred = []
    all_rels = []

    for i in xrange(rel_pred_mat.shape[0]):
        for j in xrange(rel_pred_mat.shape[1]):
            # find graph predictions (nodes and edges) that have
            # correspounding ground truth annotations
            # ignore nodes that have no edge connections
            for rel in gt_relations:
                if rel[0] not in gt_to_pred or rel[1] not in gt_to_pred:
                    continue
                # discard duplicate grounding
                if [i, j] in all_rels:
                    continue
                if i == gt_to_pred[rel[0]] and j == gt_to_pred[rel[1]]:
                    rel_pred.append([i, j, rel_pred_mat[i,j], 1])
                    all_rels.append([i, j])

    rel_pred = np.array(rel_pred)
    if rel_pred.size == 0:
        return

    # indices of predicted boxes
    pred_inds = rel_pred[:, :2].ravel()

    # draw graph predictions
    graph_dict = draw_scene_graph(cls_pred, pred_inds, rel_pred)
    viz_scene_graph(im, boxes, cls_pred, pred_inds, rel_pred, preprocess=False)
    """
    out_boxes = []
    for box, cls in zip(boxes[pred_inds], cls_pred[pred_inds]):
        out_boxes.append(box[cls*4:(cls+1)*4].tolist())

    graph_dict['boxes'] = out_boxes

    if do_save == 'y':
        scipy.misc.imsave('cherry/im_%i.png' % idx, im)
        fn = open('cherry/graph_%i.json' % idx, 'w+')
        json.dump(graph_dict, fn)
    print(idx)
    """

def viz_net(net_name, weight_name, imdb, viz_mode='viz_cls'):
    sess = tf.Session()

    # set up testing mode
    rois = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='rois')
    rel_rois = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='rois')
    ims = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='ims')
    relations = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='relations')

    inputs = {'rois': rois,
              'rel_rois': rel_rois,
              'ims': ims,
              'relations': relations,
              'num_roi': tf.placeholder(dtype=tf.int32, shape=[]),
              'num_rel': tf.placeholder(dtype=tf.int32, shape=[]),
              'num_classes': imdb.num_classes,
              'num_predicates': imdb.num_predicates,
              'rel_mask_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
              'rel_segment_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
              'rel_pair_mask_inds': tf.placeholder(dtype=tf.int32, shape=[None, 2]),
              'rel_pair_segment_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
              'n_iter': cfg.TEST.INFERENCE_ITER}



    net = get_network(net_name)(inputs)
    net.setup()
    print ('Loading model weights from {:s}').format(weight_name)
    saver = tf.train.Saver()
    saver.restore(sess, weight_name)

    roidb = imdb.roidb
    if cfg.TEST.USE_RPN_DB:
        imdb.add_rpn_rois(roidb, make_copy=False)
    prepare_roidb(roidb)

    num_images = len(imdb.image_index)

    if net.iterable:
        inference_iter = net.n_iter - 1
    else:
        inference_iter = 0
    print('=======================VIZ INFERENCE Iteration = '),
    print(inference_iter)
    print('=======================VIZ MODES = '),
    print(viz_mode)

    for im_i in xrange(num_images):
        im = imdb.im_getter(im_i)

        bbox_reg = True
        if viz_mode == 'viz_cls':
            # use ground truth bounding boxes
            bbox_reg = False
            box_proposals = gt_rois(roidb[im_i])
        elif viz_mode == 'viz_det':
            # use RPN-proposed object locations
            box_proposals, roi_scores = non_gt_rois(roidb[im_i])
            roi_scores = np.expand_dims(roi_scores, axis=1)
            nms_keep = cpu_nms(np.hstack((box_proposals, roi_scores)).astype(np.float32),
                        cfg.TEST.PROPOSAL_NMS)
            nms_keep = np.array(nms_keep)
            num_proposal = min(cfg.TEST.NUM_PROPOSALS, nms_keep.shape[0])
            keep = nms_keep[:num_proposal]
            box_proposals = box_proposals[keep, :]
        else:
            raise NotImplementedError('Incorrect visualization mode. Choose between [cls] and [det]')

        if box_proposals.size == 0 or box_proposals.shape[0] < 2:
            continue

        out_dict = im_detect(sess, net, inputs, im, box_proposals,
                                bbox_reg, [inference_iter])
        sg_entry = out_dict[inference_iter]

        # ground predicted graphs to ground truth annotations
        gt_to_pred = ground_predictions(sg_entry, roidb[im_i], 0.5)
        draw_graph_pred(im, sg_entry['boxes'], sg_entry['scores'], sg_entry['relations'],
                             gt_to_pred, roidb[im_i])
