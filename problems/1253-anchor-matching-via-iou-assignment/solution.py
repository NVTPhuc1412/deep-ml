import numpy as np

def match_anchors(anchors, gt_boxes, pos_threshold=0.5, neg_threshold=0.4):
    """
    Assign each anchor a training label via IoU matching.

    Args:
        anchors: (N, 4) boxes in xyxy format [x1, y1, x2, y2]
        gt_boxes: (M, 4) ground-truth boxes in xyxy format
        pos_threshold: IoU >= this → positive (default 0.5)
        neg_threshold: IoU <  this → negative (default 0.4)

    Returns:
        labels:     (N,) int array with values {1=pos, 0=neg, -1=ignore}
        matched_gt: (N,) int array of matched GT index, or -1
    """
    anchors = np.asarray(anchors, dtype=float)
    gt_boxes = np.asarray(gt_boxes, dtype=float)
    
    N = len(anchors)
    M = len(gt_boxes)
    if N == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    if M == 0:
        return np.full((len(anchors),), 0), np.full((len(anchors),), -1)

    ious = np.zeros((N,M))
    for i, anchor in enumerate(anchors):
        for j, gt in enumerate(gt_boxes):
            inter_x1 = max(anchor[0],gt[0])
            inter_y1 = max(anchor[1],gt[1])
            inter_x2 = min(anchor[2],gt[2])
            inter_y2 = min(anchor[3],gt[3])

            inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            a_anchor = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
            a_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])

            ious[i,j] = inter / (a_anchor + a_gt - inter)

    anchor_best_gt = np.argmax(ious, axis=1)
    anchor_best_ious = ious[range(N), anchor_best_gt]

    labels = np.full(N, -1, dtype=int)
    match_gt = np.full(N, -1, dtype=int)

    labels[anchor_best_ious<neg_threshold] = 0

    pos = anchor_best_ious >= pos_threshold
    labels[pos] = 1
    match_gt[pos] = anchor_best_gt[pos]

    gt_best_anchor = np.argmax(ious, axis=0)
    for gt_idx, anchor_idx in enumerate(gt_best_anchor):
        labels[anchor_idx] = 1
        match_gt[anchor_idx] = gt_idx

    return labels, match_gt