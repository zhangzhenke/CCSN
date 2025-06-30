import numpy as np

from skimage import segmentation
from scipy.optimize import linear_sum_assignment


__all__ = ["evaluate_f1_score_cellseg", "evaluate_f1_score"]


# 分数
def evaluate_f1_score_cellseg(masks_true, masks_pred, threshold=0.45):

    # 如果输入图像的像素数小于5000x5000，则直接处理整个图像。
    if np.prod(masks_true.shape) < (5000 * 5000):

        # 移除边界细胞
        masks_true = _remove_boundary_cells(masks_true.astype(np.int32))
        masks_pred = _remove_boundary_cells(masks_pred.astype(np.int32))

        # 计算混淆矩阵的元素（真阳性、假阳性、假阴性）。
        tp, fp, fn, iou_sum = get_confusion(masks_true, masks_pred, threshold)

    # 采用基于块的方式处理。
    else:
        H, W = masks_true.shape
        roi_size = 2000

        # Get patch grid by roi_size
        if H % roi_size != 0:
            n_H = H // roi_size + 1
            new_H = roi_size * n_H
        else:
            n_H = H // roi_size
            new_H = H

        if W % roi_size != 0:
            n_W = W // roi_size + 1
            new_W = roi_size * n_W
        else:
            n_W = W // roi_size
            new_W = W

        # Allocate values on the grid
        gt_pad = np.zeros((new_H, new_W), dtype=masks_true.dtype)
        pred_pad = np.zeros((new_H, new_W), dtype=masks_true.dtype)
        gt_pad[:H, :W] = masks_true
        pred_pad[:H, :W] = masks_pred

        tp, fp, fn = 0, 0, 0

        # Calculate confusion elements for each patch
        for i in range(n_H):
            for j in range(n_W):
                gt_roi = _remove_boundary_cells(
                    gt_pad[
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ]
                )
                pred_roi = _remove_boundary_cells(
                    pred_pad[
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ]
                )
                tp_i, fp_i, fn_i = get_confusion(gt_roi, pred_roi, threshold)
                tp += tp_i
                fp += fp_i
                fn += fn_i

    # 混淆矩阵元素计算精确率、召回率和F1分数。
    precision, recall, f1_score, ap, dq, pq = evaluate_f1_score(tp, fp, fn, iou_sum)

    return precision, recall, f1_score, ap, dq, pq


# 混淆矩阵元素计算精确率、召回率和F1分数。
def evaluate_f1_score(tp, fp, fn, iou_sum):
    """Evaluate F1-score for the given confusion elements"""

    # Do not Compute on trivial results
    if tp == 0:
        precision, recall, f1_score, ap, dq, pq = 0, 0, 0, 0, 0, 0

    else:
        ap = tp / (tp + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        dq = 2 * tp / (2 * tp + fp + fn)
        sq = iou_sum / tp
        pq = dq * sq

    return precision, recall, f1_score, ap, dq, pq


# 移除边界细胞。
def _remove_boundary_cells(mask):

    # 识别边界细胞：
    W, H = mask.shape
    bd = np.ones((W, H))
    bd[2 : W - 2, 2 : H - 2] = 0
    bd_cells = np.unique(mask * bd)

    # 移除边界细胞：
    for i in bd_cells[1:]:
        mask[mask == i] = 0

    # 重新分配标签：
    new_label, _, _ = segmentation.relabel_sequential(mask)

    # 最终返回移除边界细胞后的新标签掩码。
    return new_label


# 计算混肴矩阵
def get_confusion(masks_true, masks_pred, threshold=0.45):

    # 获取实例数量：
    # 使用连通区域标记计算实例数量：
    num_gt_instances = np.max(masks_true)
    num_pred_instances = np.max(masks_pred)

    # 处理无预测结果的情况：
    if num_pred_instances == 0:
        print("No segmentation results!")
        tp, fp, fn, iou_sum = 0, 0, 0, 0

    else:
        # 计算真实掩码和预测掩码之间的交并比 (IoU)。
        iou = _get_iou(masks_true, masks_pred)
        # 排除背景标签（0），只保留实际细胞的 IoU 值。
        iou = iou[1:, 1:]

        # 根据阈值 threshold 计算真阳性 (TP) 的数量。
        tp, iou_sum = _get_true_positive(iou, threshold)
        fp = num_pred_instances - tp
        fn = num_gt_instances - tp

    return tp, fp, fn, iou_sum


# 根据阈值 threshold 计算真阳性 (TP) 的数量。
def _get_true_positive(iou, threshold=0.45):
    """Get true positive (TP) pixels at the given threshold"""

    # Number of instances to be matched
    num_matched = min(iou.shape[0], iou.shape[1])

    # Find optimal matching by using IoU as tie-breaker
    costs = -(iou >= threshold).astype(np.float32) - iou / (2 * num_matched)
    matched_gt_label, matched_pred_label = linear_sum_assignment(costs)

    # Consider as the same instance only if the IoU is above the threshold
    match_ok = iou[matched_gt_label, matched_pred_label] >= threshold
    tp = match_ok.sum()

    # Sum the IoU values of the matched true positives
    iou_sum = iou[matched_gt_label[match_ok], matched_pred_label[match_ok]].sum()

    return tp, iou_sum


# 计算真实掩码和预测掩码之间的交并比 (IoU)。
def _get_iou(masks_true, masks_pred):
    """Get the iou between masks_true and masks_pred"""

    # Get overlap matrix (GT Instances Num, Pred Instance Num)
    overlap = _label_overlap(masks_true, masks_pred)

    # Predicted instance pixels
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)

    # GT instance pixels
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)

    # Calculate intersection of union (IoU)
    union = n_pixels_pred + n_pixels_true - overlap
    iou = overlap / union

    # Ensure numerical values
    iou[np.isnan(iou)] = 0.0

    return iou


#@jit(nopython=True)
def _label_overlap(x, y):
    """Get pixel overlaps between two masks

    Parameters
    ------------
    x, y (np array; dtype int): 0=NO masks; 1,2... are mask labels

    Returns
    ------------
    overlap (np array; dtype int): Overlaps of size [x.max()+1, y.max()+1]
    """

    # Make as 1D array
    x, y = x.ravel(), y.ravel()

    # Preallocate a Contact Map matrix
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)

    # Calculate the number of shared pixels for each label
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1

    return overlap
