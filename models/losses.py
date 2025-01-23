import torch
import torch.nn.functional as F

def compute_seg_loss(cls_pred, cls, opts):
    cls_pred = cls_pred.view(-1, opts.num_classes)
    cls = cls.view(-1)
    loss_seg = torch.nn.CrossEntropyLoss()
    loss = loss_seg(cls_pred, cls)

    return loss


def geo_l(pred, target, seg, opts):
    '''
    pred shape: [B, N, 3]
    target shape: [B, N, 3]
    '''
    pred = pred[seg == opts.part_num]
    target = target[seg == opts.part_num]
    N, _ = pred.shape
    diff01 = torch.abs((pred[:, 0] - pred[:, 1]) - (target[:, 0] - target[:, 1]))
    diff01 = torch.where(diff01 < 1, 0.5 * torch.square(diff01), diff01 - 0.5)
    diff02 = torch.abs((pred[:, 0] - pred[:, 2]) - (target[:, 0] - target[:, 2]))
    diff02 = torch.where(diff02 < 1, 0.5 * torch.square(diff02), diff02 - 0.5)
    diff12 = torch.abs((pred[:, 1] - pred[:, 2]) - (target[:, 1] - target[:, 2]))
    diff12 = torch.where(diff12 < 1, 0.5 * torch.square(diff12), diff12 - 0.5)
    loss = torch.sum(diff01 + diff02 + diff12) / (N * 3)
    return loss


def compute_r3d_loss(pred, gt, seg, opts):
    assert pred.shape == gt.shape, f"Pred shape: {pred.shape}, GT shape: {gt.shape}"
    mask = (seg == opts.part_num).float()
    mask = mask.unsqueeze(-1).repeat(1, 1, opts.kpt_num)
    loss_radial_3d = torch.nn.MSELoss()
    loss = loss_radial_3d(pred * mask, gt * mask)

    return loss

def compute_smooth_r3d_loss(pred, gt, seg, opts):
    assert pred.shape == gt.shape, f"Pred shape: {pred.shape}, GT shape: {gt.shape}"
    mask = (seg == opts.part_num).float()
    mask = mask.unsqueeze(-1).repeat(1, 1, opts.kpt_num)
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(pred * mask, gt * mask)

    return loss

def compute_vect_loss(pred_vect_per_point, gt_vect_per_point, mask):
    if pred_vect_per_point.shape[2] == 1:
        pred_vect_per_point = torch.squeeze(pred_vect_per_point, dim=2)
        diff_l2 = torch.abs(pred_vect_per_point - gt_vect_per_point) * mask
    else:
        diff_l2 = torch.norm(pred_vect_per_point - gt_vect_per_point, dim=2) * mask

    return torch.mean(torch.mean(diff_l2, axis=1), axis=0)

def compute_miou_loss(pred_seg_per_point, gt_seg_onehot):
    dot = torch.sum(pred_seg_per_point * gt_seg_onehot, axis=1)
    denominator = torch.sum(pred_seg_per_point, axis=1) + torch.sum(gt_seg_onehot, axis=1) - dot
    mIoU = dot / (denominator + 1e-10)
    return torch.mean(1.0 - mIoU)


def compute_joint_loss(axis, offset_heatmap, offset_unitvec, joint_cls, pred_heatmap, pred_axis, pred_unitvec, pred_joint_cls, joint_type_gt, opts):
    # Get the useful joint mask, gt['joint_cls_per_point'] == 0 means that
    # the point doesn't have a corresponding joint
    # B*N
    gt_joint_mask = (joint_cls > 0).float()
    # Get the heatmap and unitvec map, the loss should only be calculated for revolute joint
    gt_revolute_mask = torch.zeros_like(joint_cls) == 1
    revolute_index = torch.where(joint_type_gt[0] == 1)[0]
    assert (joint_type_gt[:, 0] == -1).all() == True
    for i in revolute_index:
        gt_revolute_mask = torch.logical_or(gt_revolute_mask, (joint_cls == i))
    gt_revolute_mask = gt_revolute_mask.float()
    # pred['heatmap_per_point']: B*N*1, gt['heatmap_per_point']: B*N, gt_revolute_mask: B*N

    heatmap_loss = compute_vect_loss(
        pred_heatmap, offset_heatmap, mask=gt_revolute_mask
    )
    # pred['unitvec_per_point']: B*N*3, gt['unitvec_per_point']: B*N*3, gt_revolute_mask: B*N
    unitvec_loss = compute_vect_loss(
        pred_unitvec, offset_unitvec, mask=gt_revolute_mask
    )
    # pred['axis_per_point]: B*N*3, gt['axis_per_point']: B*N*3, gt_joint_mask: B*N
    axis_loss = compute_vect_loss(
        pred_axis, axis, mask=gt_joint_mask
    )

    # Conver the gt['joint_cls_per_point'] into gt_joint_cls_onehot B*N*K
    gt_joint_cls_onehot = F.one_hot(
        joint_cls.long(), num_classes=opts.num_classes
    )
    joint_loss = compute_miou_loss(
        pred_joint_cls, gt_joint_cls_onehot
    )

    return heatmap_loss, unitvec_loss, axis_loss, joint_loss