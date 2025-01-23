import datetime
import os.path as osp
import torch
import numpy as np
import tqdm
import time
import math
import utils
import os
import shutil
from models.PAGENet import PAGE_Estimator
import models.losses as losses
from AccumulatorSpace_3d import estimate_6d_pose_Art


class Trainer():
    def __init__(self, data_loader, opts, vis=None):
        self.opts = opts
        self.train_loader = data_loader[0]
        self.val_loader = data_loader[1]
        self.scheduler = []
        self.epoch = 0
        self.input_channel = self.opts.input_channel
        self.n_sample_points = self.opts.n_sample_points
        self.kpt_num = self.opts.kpt_num
        self.max_epoch = self.opts.max_epoch

        if opts.mode in ['test', 'demo']:
            self.Test()
            return

        self.model = PAGE_Estimator(num_classes=self.opts.num_classes, pcld_input_channels=self.input_channel, num_kps=self.kpt_num, num_points=self.n_sample_points)

        self.model.cuda()

        if opts.mode == 'train':
            if opts.optim == 'Adam':
                self.optim = torch.optim.Adam(self.model.parameters(), lr=opts.initial_lr, weight_decay = 1e-4)
            else:
                self.optim = torch.optim.SGD(self.model.parameters(), lr=opts.initial_lr, momentum=0.9, weight_decay = 1e-4)
        print(self.optim)
        if (opts.resume_train):
            self.model, self.epoch, self.optim, self.loss_func = utils.load_checkpoint(self.model, self.optim,
                                                                                       opts.out + "/ckpt.pth.tar")
            for param_group in self.optim.param_groups:
                param_group['lr'] = opts.initial_lr

        self.loss_seg = torch.nn.CrossEntropyLoss()
        self.loss_radial_3d = torch.nn.MSELoss()
        self.iter_val = 0
        self.best_acc_mean = math.inf
        self.best_r_acc_mean = 0
        # visualizer
        self.vis = vis

        self.out = opts.out
        if not osp.exists(self.out):
            os.makedirs(self.out)

    def shift_point_cloud(self, batch_data, shift_range=0.1):
        """ Randomly shift point cloud. Shift is per point cloud.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, shifted batch of point clouds
        """
        B, N, C = batch_data.shape
        shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
        for batch_index in range(B):
            batch_data[batch_index, :, :] += shifts[batch_index, :]
        return batch_data

    def random_scale_point_cloud(self, batch_data, scale_low=0.8, scale_high=1.25):
        """ Randomly scale the point cloud. Scale is per point cloud.
            Input:
                BxNx3 array, original batch of point clouds
            Return:
                BxNx3 array, scaled batch of point clouds
        """
        B, N, C = batch_data.shape
        scales = np.random.uniform(scale_low, scale_high, B)
        for batch_index in range(B):
            batch_data[batch_index, :, :] *= scales[batch_index]
        return batch_data

    def validate(self):
        self.model.eval()

        val_loss = 0
        val_acc = 0
        self.Val_ACC = []
        self.Val_seg_ACC = []
        with torch.no_grad():
            for batch_idx, (_, _, img, cld, cld_rgb, choose, cls, radial_3d, _, axis, offset_heatmap, offset_unitvec, joint_cls, joint_type_gt) in tqdm.tqdm(
                    enumerate(self.val_loader),
                    total=len(self.val_loader),
                    ncols=80,
                    leave=False):

                img, cld, cld_rgb, choose, cls, radial_3d, axis, offset_heatmap, offset_unitvec, joint_cls, joint_type_gt = \
                    img.cuda(), cld.cuda(), cld_rgb.cuda(), choose.cuda(), cls.cuda(), radial_3d.cuda(), axis.cuda(), offset_heatmap.cuda(), offset_unitvec.cuda(), joint_cls.cuda(), joint_type_gt

                cls_pred, radial_3d_pred, pred_heatmap, pred_unitvec, pred_axis, pred_joint_cls = self.model(cld, img, choose)

                loss_s = losses.compute_seg_loss(cls_pred, cls, self.opts)
                loss_l1r = losses.compute_r3d_loss(radial_3d_pred, radial_3d, cls, self.opts)
                loss_g = losses.geo_l(radial_3d_pred, radial_3d, cls, self.opts)

                heatmap_loss, unitvec_loss, axis_loss, joint_cls_loss = losses.compute_joint_loss(axis, offset_heatmap,
                                                                                              offset_unitvec, joint_cls,
                                                                                              pred_heatmap, pred_axis,
                                                                                              pred_unitvec,
                                                                                              pred_joint_cls,
                                                                                              joint_type_gt, self.opts)
                joint_loss = heatmap_loss + unitvec_loss + axis_loss + joint_cls_loss

                loss_r = 0.8*loss_l1r + 0.2*loss_g

                loss = 10*loss_r + loss_s + heatmap_loss + unitvec_loss + axis_loss + joint_cls_loss

                if np.isnan(float(loss.item())):
                    raise ValueError('loss is nan while validating')
                val_loss += float(loss_r.item())

                cls = cls.view(-1)
                cls_pred = cls_pred.view(-1,self.opts.num_classes)
                radial_3d_pred = radial_3d_pred.view(-1,self.kpt_num)[cls==self.opts.part_num].flatten()
                radial_3d = radial_3d.view(-1, self.kpt_num)[cls == self.opts.part_num].flatten()
                len_part = torch.sum(cls==self.opts.part_num)

                acc = float(torch.sum(torch.where(torch.abs(radial_3d_pred - radial_3d) <= 0.05, 1,
                                                    0)) / (len_part*self.kpt_num))
                seg_acc = (torch.argmax(cls_pred, dim=1) == cls).sum().item() / cls.size(0)
                self.Val_ACC.append(acc)
                self.Val_seg_ACC.append(seg_acc)

                if self.vis is not None:
                    self.vis.add_scalar('Val_r+s',
                                        float(loss.detach().cpu().numpy()),
                                        self.iter_val)
                    self.vis.add_scalar('Val_r', float(loss_r.detach().cpu().numpy()), self.iter_val)
                    self.vis.add_scalar('Val_ACC', acc)

                print('Val_sum', loss, ('Val_r', loss_r),
                      ('Val_s', loss_s), ('Val_l1r', loss_l1r),
                      ('Val_g', loss_g),
                      ('Val_joint', joint_loss),
                      ('Seg_ACC', seg_acc),
                      ('Val_ACC', acc))
                self.iter_val = self.iter_val + 1

            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Val_ACC>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('Val_ACC', np.mean(self.Val_ACC))
            print('Val_seg_ACC', np.mean(self.Val_seg_ACC))

        val_acc = np.mean(self.Val_ACC)
        val_loss /= len(self.val_loader)
        mean_acc = val_loss

        is_best = mean_acc < self.best_acc_mean
        is_r_best = val_acc > self.best_r_acc_mean
        if is_best:
            self.best_acc_mean = mean_acc
        if is_r_best:
            self.best_r_acc_mean = val_acc
        save_name = "ckpt.pth.tar"
        torch.save(
            {
                'epoch': self.epoch,
                'arch': self.model.__class__.__name__,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': self.model.state_dict(),
                'best_acc_mean': self.best_acc_mean,
                'loss': loss,
            }, osp.join(self.out, save_name))
        if is_best:
            shutil.copy(osp.join(self.out, save_name),
                        osp.join(self.out, 'model_best.pth.tar'))
        if is_r_best:
            shutil.copy(osp.join(self.out, save_name),
                        osp.join(self.out, 'model_best_acc.pth.tar'))

    def train_epoch(self):
        self.model.train()
        self.Train_ACC = []
        self.Train_seg_ACC = []
        for batch_idx, (_, _, img, cld, cld_rgb, choose, cls, radial_3d, _, axis, offset_heatmap, offset_unitvec, joint_cls, joint_type_gt) in tqdm.tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch,
                ncols=80,
                leave=True):

            img, cld, cld_rgb, choose, cls, radial_3d , axis, offset_heatmap, offset_unitvec, joint_cls, joint_type_gt=\
            img.cuda(), cld.cuda(), cld_rgb.cuda(),  choose.cuda(), cls.cuda(), radial_3d.cuda(), axis.cuda(), offset_heatmap.cuda(), offset_unitvec.cuda(), joint_cls.cuda(), joint_type_gt
            self.optim.zero_grad()

            cls_pred, radial_3d_pred, pred_heatmap, pred_unitvec, pred_axis, pred_joint_cls = self.model(cld, img, choose)

            loss_s = losses.compute_seg_loss(cls_pred, cls, self.opts)
            loss_l1r = losses.compute_r3d_loss(radial_3d_pred, radial_3d, cls, self.opts)
            loss_g = losses.geo_l(radial_3d_pred, radial_3d, cls, self.opts)
            heatmap_loss, unitvec_loss, axis_loss, joint_cls_loss = losses.compute_joint_loss(axis, offset_heatmap, offset_unitvec, joint_cls, pred_heatmap, pred_axis, pred_unitvec, pred_joint_cls, joint_type_gt, self.opts)

            joint_loss = heatmap_loss+unitvec_loss+axis_loss+joint_cls_loss

            loss_r = 0.8*loss_l1r + 0.2*loss_g

            loss = 10 * loss_r + loss_s + heatmap_loss + unitvec_loss + axis_loss + joint_cls_loss
            loss.backward()
            self.optim.step()

            np_loss, np_loss_r, np_loss_s ,np_loss_l1r, np_loss_g, joint_loss = loss.detach().cpu().numpy(), loss_r.detach().cpu().numpy(), loss_s.detach().cpu().numpy(),loss_l1r.detach().cpu().numpy(),loss_g.detach().cpu().numpy(),joint_loss.detach().cpu().numpy()

            if np.isnan(np_loss):
                raise ValueError('loss is nan while training')

            if self.vis is not None:
                self.vis.add_scalar('Train_sum', np_loss)
                self.vis.add_scalar('Train_r', np_loss_r)
                self.vis.add_scalar('Train_s', np_loss_s)
                self.vis.add_scalar('Train_l1r', np_loss_l1r)
                self.vis.add_scalar('Train_g', np_loss_g)
                self.vis.add_scalar('Train_joint', joint_loss)
                cls = cls.view(-1)
                cls_pred = cls_pred.view(-1, self.opts.num_classes)
                radial_3d_pred = radial_3d_pred.view(-1,self.kpt_num)[cls==self.opts.part_num].flatten()
                radial_3d = radial_3d.view(-1, self.kpt_num)[cls == self.opts.part_num].flatten()
                len_part = torch.sum(cls==self.opts.part_num)
                acc = float(torch.sum(
                                        torch.where(torch.abs(radial_3d_pred - radial_3d) <= 0.05, 1,
                                                    0)) / (len_part*self.kpt_num))
                seg_acc = (torch.argmax(cls_pred, dim=1) == cls).sum().item() / cls.size(0)
                self.Train_ACC.append(acc)
                self.Train_seg_ACC.append(seg_acc)
                self.vis.add_scalar('Train_ACC', acc)

            print('Train_sum', np_loss, ('Train_r', np_loss_r), ('Train_s', np_loss_s), ('Train_l1r', np_loss_l1r),('Train_g', np_loss_g),('Train_joint', joint_loss),
                  ('Seg_ACC', seg_acc),('Train_ACC', acc))

        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Train_ACC>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('Train_ACC', np.mean(self.Train_ACC))
        print('Train_seg_ACC', np.mean(self.Train_seg_ACC))

    def Train(self):
        # self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, verbose=True)
        for epoch in tqdm.trange(self.epoch, self.max_epoch, desc='Train',
                                 ncols=80):
            self.epoch = epoch
            self.train_epoch()
            self.validate()
            if self.epoch % 70 == 0 and self.epoch != 0:
                for g in self.optim.param_groups:
                    g['lr'] /= 10
    def Test(self):
            estimate_6d_pose_Art(self.opts)