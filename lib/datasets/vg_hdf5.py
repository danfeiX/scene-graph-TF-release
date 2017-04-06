import os
from datasets.imdb import imdb
import numpy as np
import copy
import scipy.sparse
import h5py, json
from fast_rcnn.config import cfg

class vg_hdf5(imdb):
    def __init__(self, roidb_file, dict_file, imdb_file, rpndb_file, split, num_im):
        imdb.__init__(self, roidb_file[:-3])

        # read in dataset from a h5 file and a dict (json) file
        self.im_h5 = h5py.File(os.path.join(cfg.VG_DIR, imdb_file), 'r')
        self.roi_h5 = h5py.File(os.path.join(cfg.VG_DIR, roidb_file), 'r')

        # roidb metadata
        self.info = json.load(open(os.path.join(cfg.VG_DIR,
                                                dict_file), 'r'))
        self.im_refs = self.im_h5['images'] # image data reference
        im_scale = self.im_refs.shape[2]

        print('split==%i' % split)
        data_split = self.roi_h5['split'][:]
        self.split = split
        if split >= 0:
            split_mask = data_split == split # current split
        else: # -1
            split_mask = data_split >= 0 # all
        # get rid of images that do not have box
        valid_mask = self.roi_h5['img_to_first_box'][:] > 0
        valid_mask = np.bitwise_and(split_mask, valid_mask)
        self._image_index = np.where(valid_mask)[0] # split index

        if num_im > -1:
            self._image_index = self._image_index[:num_im]

        # override split mask
        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[self.image_index] = True  # build a split mask
        # if use all images
        self.im_sizes = np.vstack([self.im_h5['image_widths'][split_mask],
                                   self.im_h5['image_heights'][split_mask]]).transpose()

        # filter rpn roidb with split_mask
        if cfg.TRAIN.USE_RPN_DB:
            self.rpn_h5_fn = os.path.join(cfg.VG_DIR, rpndb_file)
            self.rpn_h5 = h5py.File(os.path.join(cfg.VG_DIR, rpndb_file), 'r')
            self.rpn_rois = self.rpn_h5['rpn_rois']
            self.rpn_scores = self.rpn_h5['rpn_scores']
            self.rpn_im_to_roi_idx = np.array(self.rpn_h5['im_to_roi_idx'][split_mask])
            self.rpn_num_rois = np.array(self.rpn_h5['num_rois'][split_mask])

        # h5 file is in 1-based index
        self.im_to_first_box = self.roi_h5['img_to_first_box'][split_mask]
        self.im_to_last_box = self.roi_h5['img_to_last_box'][split_mask]
        self.all_boxes = self.roi_h5['boxes_%i' % im_scale][:]  # will index later
        self.all_boxes[:, :2] = self.all_boxes[:, :2]
        assert(np.all(self.all_boxes[:, :2] >= 0))  # sanity check
        assert(np.all(self.all_boxes[:, 2:] > 0))  # no empty box


        # convert from xc, yc, w, h to x1, y1, x2, y2
        self.all_boxes[:, :2] = self.all_boxes[:, :2] - self.all_boxes[:, 2:]/2
        self.all_boxes[:, 2:] = self.all_boxes[:, :2] + self.all_boxes[:, 2:]
        self.labels = self.roi_h5['labels'][:,0]

        # add background class
        self.info['label_to_idx']['__background__'] = 0
        self.class_to_ind = self.info['label_to_idx']
        self.ind_to_classes = sorted(self.class_to_ind, key=lambda k:
                               self.class_to_ind[k])
        cfg.ind_to_class = self.ind_to_classes

        # load relation labels
        self.im_to_first_rel = self.roi_h5['img_to_first_rel'][split_mask]
        self.im_to_last_rel = self.roi_h5['img_to_last_rel'][split_mask]
        self._relations = self.roi_h5['relationships'][:]
        self._relation_predicates = self.roi_h5['predicates'][:,0]
        assert(self.im_to_first_rel.shape[0] == self.im_to_last_rel.shape[0])
        assert(self._relations.shape[0] == self._relation_predicates.shape[0]) # sanity check
        self.predicate_to_ind = self.info['predicate_to_idx']
        self.predicate_to_ind['__background__'] = 0
        self.ind_to_predicates = sorted(self.predicate_to_ind, key=lambda k:
                                  self.predicate_to_ind[k])

        cfg.ind_to_predicate = self.ind_to_predicates

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

    def im_getter(self, idx):
        w, h = self.im_sizes[idx, :]
        ridx = self.image_index[idx]
        im = self.im_refs[ridx]
        im = im[:, :h, :w] # crop out
        im = im.transpose((1,2,0)) # c h w -> h w c
        return im

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        """
        gt_roidb = []
        for i in range(self.num_images):
            assert(self.im_to_first_box[i] >= 0)
            boxes = self.all_boxes[self.im_to_first_box[i]
                                   :self.im_to_last_box[i]+1,:]

            gt_classes = self.labels[self.im_to_first_box[i]
                                     :self.im_to_last_box[i]+1]

            overlaps = np.zeros((boxes.shape[0], self.num_classes))
            for j, o in enumerate(overlaps): # to one-hot
                #if gt_classes[j] > 0: # consider negative sample
                o[gt_classes[j]] = 1.0
            overlaps = scipy.sparse.csr_matrix(overlaps)

            # make ground-truth relations
            gt_relations = []
            if self.im_to_first_rel[i] >= 0: # if image has relations
                predicates = self._relation_predicates[self.im_to_first_rel[i]
                                             :self.im_to_last_rel[i]+1]
                obj_idx = self._relations[self.im_to_first_rel[i]
                                             :self.im_to_last_rel[i]+1]
                obj_idx = obj_idx - self.im_to_first_box[i]
                assert(np.all(obj_idx>=0) and np.all(obj_idx<boxes.shape[0])) # sanity check
                for j, p in enumerate(predicates):
                    gt_relations.append([obj_idx[j][0], obj_idx[j][1], p])

            gt_relations = np.array(gt_relations)

            seg_areas = np.multiply((boxes[:, 2] - boxes[:, 0] + 1),
                                    (boxes[:, 3] - boxes[:, 1] + 1)) # box areas
            gt_roidb.append({'boxes': boxes,
                             'gt_classes' : gt_classes,
                             'gt_overlaps' : overlaps,
                             'gt_relations': gt_relations,
                             'flipped' : False,
                             'seg_areas' : seg_areas,
                             'db_idx': i,
                             'image': lambda im_i=i: self.im_getter(im_i),
                             'roi_scores': np.ones(boxes.shape[0]),
                             'width': self.im_sizes[i][0],
                             'height': self.im_sizes[i][1]})
        return gt_roidb

    def add_rpn_rois(self, gt_roidb_batch, make_copy=True):
        """
        Load precomputed RPN proposals
        """
        gt_roidb = copy.deepcopy(gt_roidb_batch) if make_copy else gt_roidb_batch
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_gt_rpn_roidb(gt_roidb, rpn_roidb)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        # load an precomputed ROIDB to the current gt ROIDB
        box_list = []
        score_list = []
        for entry in gt_roidb:
            i = entry['db_idx']
            im_rois = self.rpn_rois[self.rpn_im_to_roi_idx[i]: self.rpn_im_to_roi_idx[i]+self.rpn_num_rois[i], :].copy()
            roi_scores = self.rpn_scores[self.rpn_im_to_roi_idx[i]: self.rpn_im_to_roi_idx[i]+self.rpn_num_rois[i], 0].copy()
            box_list.append(im_rois)
            score_list.append(roi_scores)
        roidb = self.create_roidb_from_box_list(box_list, gt_roidb)
        for i, rdb in enumerate(roidb):
            rdb['roi_scores'] = score_list[i]
        return roidb

    def _get_widths(self):
        return self.im_sizes[:,0]

if __name__ == '__main__':
    import roi_data_layer.roidb as roidb
    import roi_data_layer.layer as layer
    d = vg_hdf5('VG.h5', 'VG-dicts.json', 'imdb_512.h5', 'proposals.h5', 0)
    roidb.prepare_roidb(d)
    roidb.add_bbox_regression_targets(d.roidb)
    l = layer.RoIDataLayer(d.num_classes)
    l.set_roidb(d.roidb)
    l.next_batch()
    from IPython import embed; embed()
