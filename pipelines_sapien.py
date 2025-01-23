import os.path as osp
import warnings
from collections.abc import Sequence
import matplotlib.pyplot as plt
import random
import mmcv
import open3d as o3d
import numpy as np
import json
import copy
import torch
import h5py
import pycocotools.mask as maskUtils

# from mmcv.parallel import DataContainer as DC
import os
import sys

from arti_utils import point_3d_offset_joint
# from mmdet.datasets.arti_vis_utils import plot3d_pts, plot_arrows, plot_imgs, plot_arrows_list
# from mmdet.datasets.registry import PIPELINES


epsilon = 10e-8
# thres_r = 0.07 #eyeglasses
thres_r = 0.05 #laptop

INSTANCE_CLASSES = ['BG', 'box', 'stapler', 'cutter', 'drawer', 'scissor']
PART_CLASSES = {'box': ['BG', 'base_link', 'link1'],
                'stapler': ['BG', 'base_link', 'link1'],
                'cutter': ['BG', 'base_link', 'link1'],
                'drawer': ['BG', 'base_link', 'link1'],
                'scissor': ['BG', 'link1', 'link2']}
JOINT_CLASSES = ['none', 'prismatic', 'revolute']
scissors_urdf_list = [10449,10450,10499,10537,10537,10546,10558,10562,10844,10889,10960,10962,10973,10975,11020,11028,11029,11040,11052,11077,11080,11100,11113,11013,11021]


def rgbd2pc(rgb_path, depth_path, camera_intrinsic, vis=False, save_pcd=False):
    #color_raw = o3d.io.read_image(rgb_path)
    #depth_raw = o3d.io.read_image(depth_path)
    rgb_path = o3d.geometry.Image(rgb_path)
    depth_path = o3d.geometry.Image(depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_path,
                                                                    depth_path,
                                                                    1000.0,
                                                                    20.0,
                                                                    convert_rgb_to_intensity=False)


    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsic
    )
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    if vis:
        o3d.visualization.draw_geometries([pcd])
    if save_pcd:
        basename = osp.basename(rgb_path)
        pcd_save_name = basename.split('.png')[0] + '.pcd'
        o3d.io.write_point_cloud(pcd_save_name, pcd)

    return pcd


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


# @PIPELINES.register_module
class LoadImageFromFileSapien(object):

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        filename = osp.join(results['img_prefix'],
                            'color',
                            results['sample_name'] + '.jpg')
        img = mmcv.imread(filename)
        if self.to_float32:
            img = img.astype(np.float32)
        results['img'] = img
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


# @PIPELINES.register_module
class CreatePointDataSapien(object):
    def __init__(self, cat=None, downsample_voxel=0.005, with_rgb=False):
        self.downsample_voxel = downsample_voxel
        self.with_rgb = with_rgb
        self.cat = cat

    def __call__(self, results):
        # n_max_parts = results['n_max_parts']
        instance_info = results['instance_info']

        n_parts = len(instance_info['links'])
        # parts_map = [instance_info['links'][l]['link_category_id']
        #              for l in range(n_parts)]
        # print(parts_map)
        joint_part = results['joint_ins']['parent']
        n_total_points = 0
        parts_pts = [None] * n_parts
        parts_pts_feature = [None] * n_parts
        parts_gts = [None] * n_parts
        parts_cls = [None] * n_parts
        parts_parent_joint = [None] * n_parts
        parts_child_joint = [None] * n_parts
        state_act = [None] * n_parts
        part_trans = [None] * n_parts

        img_prefix = results['img_prefix']
        img_height = results['img_height']
        img_width = results['img_width']
        camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(results['camera_intrinsic_path'])
        urdf_id = results['urdf_id']

        for j in range(n_parts):
            # part_id = label_map[j]
            part_id = j
            # color = copy.deepcopy(color_image)
            # depth = copy.deepcopy(depth_image)
            if 'ReArtMix' in img_prefix:
                color_image = o3d.io.read_image(osp.join(img_prefix, 'color', results['color_path']))
                depth_image = o3d.io.read_image(osp.join(img_prefix, 'depth', results['depth_path']))
            else:
                color_image = o3d.io.read_image(osp.join(img_prefix, results['color_path']))
                depth_image = o3d.io.read_image(osp.join(img_prefix, results['depth_path']))
            try:
                if 'ReArtMix' in img_prefix:
                    link_id = [i for i in range(len(instance_info['links'])) if
                               (instance_info['links'][i]['link_category_id']-1) == part_id]
                elif self.cat == 'scissors':
                    if urdf_id not in scissors_urdf_list:
                        if part_id==0:
                            link_id=[1]
                        elif part_id==1:
                            link_id=[0]
                    else:
                        link_id = [i for i in range(len(instance_info['links'])) if
                                   instance_info['links'][i]['link_category_id'] == part_id]
                else:
                    link_id = [i for i in range(len(instance_info['links'])) if
                               instance_info['links'][i]['link_category_id'] == part_id]
                assert len(link_id) == 1
                link_id = link_id[0]

                part_seg = instance_info['links'][link_id]['segmentation']
                part_tran = np.array(instance_info['links'][link_id]['transformation'])
                part_trans[part_id] = part_tran
                try:
                    state = instance_info['links'][link_id]['state']
                    if self.cat == 'drawer':
                        state_degree = state
                    else:
                        state_degree = np.radians(state)
                    state_act[part_id] = state_degree
                except:
                    state_act[part_id] = 0.0


                # part_seg = instance_info['links'][part_id]['segmentation']

                rle = maskUtils.frPyObjects(part_seg, img_height, img_width)
                part_mask = np.sum(maskUtils.decode(rle), axis=2).clip(max=1).astype(np.uint8)

                part_color = color_image * np.repeat(part_mask[..., np.newaxis], 3, 2)
                part_depth = depth_image * part_mask

                # part_pc = rgbd2pc(part_color, part_depth, results['camera_intrinsic'])
                part_pc = rgbd2pc(part_color, part_depth, camera_intrinsic)
                if self.downsample_voxel > 0:
                    part_pc = o3d.geometry.PointCloud.voxel_down_sample(part_pc, self.downsample_voxel)

                parts_pts[part_id] = np.array(part_pc.points)
                if self.with_rgb:
                    parts_pts_feature[part_id] = np.array(part_pc.colors)
                # part_tran = np.array(instance_info['links'][part_id]['transformation'])
                if 'rest_transformation' in results.keys():
                    part_tran = part_tran @ np.linalg.inv(results['rest_transformation'][part_id])
                part_pc_copy = copy.deepcopy(part_pc)
                part_coord = part_pc_copy.transform(np.linalg.inv(np.array(part_tran)))
                part_coord = np.array(part_coord.points)
                parts_gts[part_id] = part_coord
                if 'label_map' in results.keys():
                    part_label = results['label_map'][part_id]
                else:
                    part_label = part_id
                parts_cls[part_id] = part_label * np.ones((parts_pts[part_id].shape[0]), dtype=np.float32)

                n_total_points += parts_pts[part_id].shape[0]

            except:
                parts_pts[part_id] = np.array([0.,0.,0.]).reshape(1, 3)
                if self.with_rgb:
                    parts_pts_feature[part_id] = np.array([0.,0.,0.]).reshape(1, 3)
                parts_gts[part_id] = np.array([0.,0.,0.]).reshape(1, 3)
                parts_cls[part_id] = np.array([0.])

            # print(parts_pts[part_id].shape[0])

            # parts_parent_joint[part_id] = label_map[part_id]
            #             # parts_child_joint[part_id] = [ind for ind, x in enumerate(joint_part) if x == label_map[part_id]]
            parts_parent_joint[part_id] = part_id
            parts_child_joint[part_id] = [ind for ind, x in enumerate(joint_part) if x == part_id]

        if n_total_points == 0:
            print(results['bbox'])
            print(results['color_path'], instance_info['id'])
            print(p.shape[0] for p in parts_pts)

        results['n_parts'] = n_parts
        results['parts_pts'] = parts_pts
        results['part_trans'] = part_trans
        if self.with_rgb:
            results['parts_pts_feature'] = parts_pts_feature
        results['parts_gts'] = parts_gts
        results['parts_cls'] = parts_cls
        results['parts_parent_joint'] = parts_parent_joint
        results['parts_child_joint'] = parts_child_joint
        results['n_total_points'] = n_total_points
        results['state_act'] = state_act

        return results

class Create_Art_DataSapien(object):
    def __init__(self, part_num, cat=None, downsample_voxel=0.005, with_rgb=False):
        self.downsample_voxel = downsample_voxel
        self.with_rgb = with_rgb
        self.cat = cat
        self.part_num = part_num

    def __call__(self, results, part_num):
        # n_max_parts = results['n_max_parts']
        instance_info = results['instance_info']

        n_parts = len(instance_info['links'])
        joint_part = results['joint_ins']['parent']
        n_total_points = 0
        parts_pts = [None] * n_parts
        parts_pts_feature = [None] * n_parts
        parts_gts = [None] * n_parts
        parts_cls = [None] * n_parts
        parts_parent_joint = [None] * n_parts
        parts_child_joint = [None] * n_parts
        state_act = [None] * n_parts
        part_trans = [None] * n_parts

        img_prefix = results['img_prefix']
        img_height = results['img_height']
        img_width = results['img_width']
        camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(results['camera_intrinsic_path'])
        urdf_id = results['urdf_id']

        # part_id = label_map[j]
        part_id = part_num
        # color = copy.deepcopy(color_image)
        # depth = copy.deepcopy(depth_image)
        if 'ReArtMix' in img_prefix:
            color_image = o3d.io.read_image(osp.join(img_prefix, 'color', results['color_path']))
            depth_image = o3d.io.read_image(osp.join(img_prefix, 'depth', results['depth_path']))
        else:
            color_image = o3d.io.read_image(osp.join(img_prefix, results['color_path']))
            depth_image = o3d.io.read_image(osp.join(img_prefix, results['depth_path']))
        try:
            if 'ReArtMix' in img_prefix:
                link_id = [i for i in range(len(instance_info['links'])) if
                           (instance_info['links'][i]['link_category_id'] - 1) == part_id]
            elif self.cat == 'scissors':
                if urdf_id not in scissors_urdf_list:
                    if part_id == 0:
                        link_id = [1]
                    elif part_id == 1:
                        link_id = [0]
                else:
                    link_id = [i for i in range(len(instance_info['links'])) if
                               instance_info['links'][i]['link_category_id'] == part_id]
            else:
                link_id = [i for i in range(len(instance_info['links'])) if
                           instance_info['links'][i]['link_category_id'] == part_id]
            assert len(link_id) == 1
            link_id = link_id[0]

            part_seg = instance_info['links'][link_id]['segmentation']
            part_tran = np.array(instance_info['links'][link_id]['transformation'])
            part_trans[part_id] = part_tran
            try:
                state = instance_info['links'][link_id]['state']
                if self.cat == 'drawer':
                    state_degree = state
                else:
                    state_degree = np.radians(state)
                state_act[part_id] = state_degree
            except:
                state_act[part_id] = 0.0

            # part_seg = instance_info['links'][part_id]['segmentation']

            rle = maskUtils.frPyObjects(part_seg, img_height, img_width)
            part_mask = np.sum(maskUtils.decode(rle), axis=2).clip(max=1).astype(np.uint8)

            part_color = color_image * np.repeat(part_mask[..., np.newaxis], 3, 2)
            part_depth = depth_image * part_mask

            # part_pc = rgbd2pc(part_color, part_depth, results['camera_intrinsic'])
            part_pc = rgbd2pc(part_color, part_depth, camera_intrinsic)
            if self.downsample_voxel > 0:
                part_pc = o3d.geometry.PointCloud.voxel_down_sample(part_pc, self.downsample_voxel)

            parts_pts[part_id] = np.array(part_pc.points)
            if self.with_rgb:
                parts_pts_feature[part_id] = np.array(part_pc.colors)
            # part_tran = np.array(instance_info['links'][part_id]['transformation'])
            if 'rest_transformation' in results.keys():
                part_tran = part_tran @ np.linalg.inv(results['rest_transformation'][part_id])
            part_pc_copy = copy.deepcopy(part_pc)
            part_coord = part_pc_copy.transform(np.linalg.inv(np.array(part_tran)))
            part_coord = np.array(part_coord.points)
            parts_gts[part_id] = part_coord
            if 'label_map' in results.keys():
                part_label = results['label_map'][part_id]
            else:
                part_label = part_id
            parts_cls[part_id] = part_label * np.ones((parts_pts[part_id].shape[0]), dtype=np.float32)

            n_total_points += parts_pts[part_id].shape[0]

        except:
            parts_pts[part_id] = np.array([0., 0., 0.]).reshape(1, 3)
            if self.with_rgb:
                parts_pts_feature[part_id] = np.array([0., 0., 0.]).reshape(1, 3)
            parts_gts[part_id] = np.array([0., 0., 0.]).reshape(1, 3)
            parts_cls[part_id] = np.array([0.])
            part_tran = np.eye(4)
            part_mask = np.zeros((img_height, img_width), dtype=np.float32)
        # print(parts_pts[part_id].shape[0])

        # parts_parent_joint[part_id] = label_map[part_id]
        #             # parts_child_joint[part_id] = [ind for ind, x in enumerate(joint_part) if x == label_map[part_id]]
        parts_parent_joint[part_id] = part_id
        parts_child_joint[part_id] = [ind for ind, x in enumerate(joint_part) if x == part_id]

        if n_total_points == 0:
            print(results['bbox'])
            print(results['color_path'], instance_info['id'])
            print(p.shape[0] for p in parts_pts)

        results['n_parts'] = n_parts
        results['parts_pts'] = parts_pts
        results['part_tran'] = part_tran
        results['part_mask'] = part_mask
        if self.with_rgb:
            results['parts_pts_feature'] = parts_pts_feature
        results['parts_gts'] = parts_gts
        results['parts_cls'] = parts_cls
        results['parts_parent_joint'] = parts_parent_joint
        results['parts_child_joint'] = parts_child_joint
        results['n_total_points'] = n_total_points
        results['state_act'] = state_act

        return results

class Create_Art_3d_PointDataSapien(object):
    def __init__(self, cat=None, downsample_voxel=0.005, with_rgb=False):
        self.downsample_voxel = downsample_voxel
        self.with_rgb = with_rgb
        self.cat = cat

    def __call__(self, results):
        # n_max_parts = results['n_max_parts']
        instance_info = results['instance_info']

        n_parts = len(instance_info['links'])
        # parts_map = [instance_info['links'][l]['link_category_id']
        #              for l in range(n_parts)]
        # print(parts_map)
        joint_part = results['joint_ins']['parent']
        n_total_points = 0
        parts_pts = [None] * n_parts
        parts_pts_feature = [None] * n_parts
        parts_gts = [None] * n_parts
        parts_cls = [None] * n_parts
        parts_parent_joint = [None] * n_parts
        parts_child_joint = [None] * n_parts
        state_act = [None] * n_parts
        part_trans = [None] * n_parts
        part_masks = [None] * n_parts

        img_prefix = results['img_prefix']
        img_height = results['img_height']
        img_width = results['img_width']
        camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(results['camera_intrinsic_path'])
        urdf_id = results['urdf_id']

        for j in range(n_parts):
            # part_id = label_map[j]
            part_id = j
            # color = copy.deepcopy(color_image)
            # depth = copy.deepcopy(depth_image)
            if 'ReArtMix' in img_prefix:
                color_image = o3d.io.read_image(osp.join(img_prefix, 'color', results['color_path']))
                depth_image = o3d.io.read_image(osp.join(img_prefix, 'depth', results['depth_path']))
            else:
                color_image = o3d.io.read_image(osp.join(img_prefix, results['color_path']))
                depth_image = o3d.io.read_image(osp.join(img_prefix, results['depth_path']))
            try:
                if 'ReArtMix' in img_prefix:
                    link_id = [i for i in range(len(instance_info['links'])) if
                               (instance_info['links'][i]['link_category_id']-1) == part_id]
                elif self.cat == 'scissors':
                    if urdf_id not in scissors_urdf_list:
                        if part_id==0:
                            link_id=[1]
                        elif part_id==1:
                            link_id=[0]
                    else:
                        link_id = [i for i in range(len(instance_info['links'])) if
                                   instance_info['links'][i]['link_category_id'] == part_id]
                else:
                    link_id = [i for i in range(len(instance_info['links'])) if
                               instance_info['links'][i]['link_category_id'] == part_id]
                assert len(link_id) == 1
                link_id = link_id[0]

                part_seg = instance_info['links'][link_id]['segmentation']
                part_tran = np.array(instance_info['links'][link_id]['transformation'])

                try:
                    state = instance_info['links'][link_id]['state']
                    if self.cat == 'drawer':
                        state_degree = state
                    else:
                        state_degree = np.radians(state)
                    state_act[part_id] = state_degree
                except:
                    state_act[part_id] = 0.0


                # part_seg = instance_info['links'][part_id]['segmentation']

                rle = maskUtils.frPyObjects(part_seg, img_height, img_width)
                part_mask = np.sum(maskUtils.decode(rle), axis=2).clip(max=1).astype(np.uint8)
                part_masks[part_id] = part_mask

                part_color = color_image * np.repeat(part_mask[..., np.newaxis], 3, 2)
                part_depth = depth_image * part_mask

                # part_pc = rgbd2pc(part_color, part_depth, results['camera_intrinsic'])
                part_pc = rgbd2pc(part_color, part_depth, camera_intrinsic)
                if self.downsample_voxel > 0:
                    part_pc = o3d.geometry.PointCloud.voxel_down_sample(part_pc, self.downsample_voxel)

                parts_pts[part_id] = np.array(part_pc.points)
                if self.with_rgb:
                    parts_pts_feature[part_id] = np.array(part_pc.colors)
                # part_tran = np.array(instance_info['links'][part_id]['transformation'])
                if 'rest_transformation' in results.keys():
                    part_tran = part_tran @ np.linalg.inv(results['rest_transformation'][part_id])
                part_trans[part_id] = part_tran
                part_pc_copy = copy.deepcopy(part_pc)
                part_coord = part_pc_copy.transform(np.linalg.inv(np.array(part_tran)))
                part_coord = np.array(part_coord.points)
                parts_gts[part_id] = part_coord
                if 'label_map' in results.keys():
                    part_label = results['label_map'][part_id]
                else:
                    part_label = part_id
                parts_cls[part_id] = part_label * np.ones((parts_pts[part_id].shape[0]), dtype=np.float32)

                n_total_points += parts_pts[part_id].shape[0]

            except:
                parts_pts[part_id] = np.array([0.,0.,0.]).reshape(1, 3)
                if self.with_rgb:
                    parts_pts_feature[part_id] = np.array([0.,0.,0.]).reshape(1, 3)
                parts_gts[part_id] = np.array([0., 0., 0.]).reshape(1, 3)
                parts_cls[part_id] = np.array([0.])
                part_tran = np.eye(4)

            # parts_parent_joint[part_id] = label_map[part_id]
            #             # parts_child_joint[part_id] = [ind for ind, x in enumerate(joint_part) if x == label_map[part_id]]
            parts_parent_joint[part_id] = part_id
            parts_child_joint[part_id] = [ind for ind, x in enumerate(joint_part) if x == part_id]

        if n_total_points == 0:
            print(results['bbox'])
            print(results['color_path'], instance_info['id'])
            print(p.shape[0] for p in parts_pts)

        results['n_parts'] = n_parts
        results['parts_pts'] = parts_pts
        results['part_trans'] = part_trans
        if self.with_rgb:
            results['parts_pts_feature'] = parts_pts_feature
        results['parts_gts'] = parts_gts
        results['parts_cls'] = parts_cls
        results['part_masks'] = part_masks
        results['parts_parent_joint'] = parts_parent_joint
        results['parts_child_joint'] = parts_child_joint
        results['n_total_points'] = n_total_points
        results['state_act'] = state_act

        return results

class Create_Art_3d_scale_PointDataSapien(object):
    def __init__(self, cat=None, downsample_voxel=0.005, with_rgb=False):
        self.downsample_voxel = downsample_voxel
        self.with_rgb = with_rgb
        self.cat = cat

    def __call__(self, results):
        # n_max_parts = results['n_max_parts']
        instance_info = results['instance_info']

        n_parts = len(instance_info['links'])
        # parts_map = [instance_info['links'][l]['link_category_id']
        #              for l in range(n_parts)]
        # print(parts_map)
        joint_part = results['joint_ins']['parent']
        n_total_points = 0
        parts_pts = [None] * n_parts
        parts_pts_feature = [None] * n_parts
        parts_gts = [None] * n_parts
        parts_cls = [None] * n_parts
        parts_parent_joint = [None] * n_parts
        parts_child_joint = [None] * n_parts
        state_act = [None] * n_parts
        part_trans = [None] * n_parts
        part_trans_rest = [None] * n_parts
        part_masks = [None] * n_parts

        img_prefix = results['img_prefix']
        img_height = results['img_height']
        img_width = results['img_width']
        camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(results['camera_intrinsic_path'])
        urdf_id = results['urdf_id']

        for j in range(n_parts):
            # part_id = label_map[j]
            part_id = j
            # color = copy.deepcopy(color_image)
            # depth = copy.deepcopy(depth_image)
            if 'ReArtMix' in img_prefix:
                color_image = o3d.io.read_image(osp.join(img_prefix, 'color', results['color_path']))
                depth_image = o3d.io.read_image(osp.join(img_prefix, 'depth', results['depth_path']))
            else:
                color_image = o3d.io.read_image(osp.join(img_prefix, results['color_path']))
                depth_image = o3d.io.read_image(osp.join(img_prefix, results['depth_path']))
            try:
                if 'ReArtMix' in img_prefix:
                    link_id = [i for i in range(len(instance_info['links'])) if
                               (instance_info['links'][i]['link_category_id']-1) == part_id]
                elif self.cat == 'scissors':
                    if urdf_id not in scissors_urdf_list:
                        if part_id==0:
                            link_id=[1]
                        elif part_id==1:
                            link_id=[0]
                    else:
                        link_id = [i for i in range(len(instance_info['links'])) if
                                   instance_info['links'][i]['link_category_id'] == part_id]
                else:
                    link_id = [i for i in range(len(instance_info['links'])) if
                               instance_info['links'][i]['link_category_id'] == part_id]
                assert len(link_id) == 1
                link_id = link_id[0]

                part_seg = instance_info['links'][link_id]['segmentation']
                part_tran = np.array(instance_info['links'][link_id]['transformation'])

                try:
                    state = instance_info['links'][link_id]['state']
                    if self.cat == 'drawer':
                        state_degree = state
                    else:
                        state_degree = np.radians(state)
                    state_act[part_id] = state_degree
                except:
                    state_act[part_id] = 0.0


                # part_seg = instance_info['links'][part_id]['segmentation']

                rle = maskUtils.frPyObjects(part_seg, img_height, img_width)
                part_mask = np.sum(maskUtils.decode(rle), axis=2).clip(max=1).astype(np.uint8)
                part_masks[part_id] = part_mask

                part_color = color_image * np.repeat(part_mask[..., np.newaxis], 3, 2)
                part_depth = depth_image * part_mask

                # part_pc = rgbd2pc(part_color, part_depth, results['camera_intrinsic'])
                part_pc = rgbd2pc(part_color, part_depth, camera_intrinsic)
                if self.downsample_voxel > 0:
                    part_pc = o3d.geometry.PointCloud.voxel_down_sample(part_pc, self.downsample_voxel)

                parts_pts[part_id] = np.array(part_pc.points)
                parts_pts_feature[part_id] = np.array(part_pc.colors)
                # part_tran = np.array(instance_info['links'][part_id]['transformation'])
                if 'rest_transformation' in results.keys():
                    part_tran_rest = part_tran @ np.linalg.inv(results['rest_transformation'][part_id])
                    part_trans_rest[part_id] = part_tran_rest

                part_trans[part_id] = part_tran
                part_pc_copy = copy.deepcopy(part_pc)
                part_coord = part_pc_copy.transform(np.linalg.inv(np.array(part_tran)))
                part_coord = np.array(part_coord.points)
                parts_gts[part_id] = part_coord
                if 'label_map' in results.keys():
                    part_label = results['label_map'][part_id]
                else:
                    part_label = part_id
                parts_cls[part_id] = part_label * np.ones((parts_pts[part_id].shape[0]), dtype=np.float32)

                n_total_points += parts_pts[part_id].shape[0]

            except:
                parts_pts[part_id] = np.array([0.,0.,0.]).reshape(1, 3)
                parts_pts_feature[part_id] = np.array([0.,0.,0.]).reshape(1, 3)
                parts_gts[part_id] = np.array([0.,0.,0.]).reshape(1, 3)
                parts_cls[part_id] = np.array([0.])
                part_trans[part_id] = np.eye(4)
                part_trans_rest[part_id] = np.eye(4)
            # print(parts_pts[part_id].shape[0])

            # parts_parent_joint[part_id] = label_map[part_id]
            #             # parts_child_joint[part_id] = [ind for ind, x in enumerate(joint_part) if x == label_map[part_id]]
            parts_parent_joint[part_id] = part_id
            parts_child_joint[part_id] = [ind for ind, x in enumerate(joint_part) if x == part_id]

        if n_total_points == 0:
            print(results['bbox'])
            print(results['color_path'], instance_info['id'])
            print(p.shape[0] for p in parts_pts)

        results['n_parts'] = n_parts
        results['parts_pts'] = parts_pts
        results['part_trans'] = part_trans
        results['part_trans_rest'] = part_trans_rest
        results['part_masks'] = part_masks
        results['parts_pts_feature'] = parts_pts_feature
        results['parts_gts'] = parts_gts
        results['parts_cls'] = parts_cls
        results['parts_parent_joint'] = parts_parent_joint
        results['parts_child_joint'] = parts_child_joint
        results['n_total_points'] = n_total_points
        results['state_act'] = state_act

        return results

# @PIPELINES.register_module
class SamplePairParts(object):
    def __init__(self, sampled_ids=None):
        self.sampled_ids = sampled_ids

    def __call__(self, results):
        if self.sampled_ids is None:
            n_max_parts = results['n_max_parts']
            valid_part_ids = [ind for ind in range(n_max_parts)
                              if results['parts_cls'][ind] is not None and results['parts_cls'][ind].shape[0] > 1]
            # if len(valid_part_ids) < 2:
            #     print(results['img_prefix'])
            #     print(results['sample_name'])
            #     print(valid_part_ids)
            try:
                # sampled_part_ids = [0] + random.sample(valid_part_ids[1:], 1)
                sampled_part_ids = random.sample(valid_part_ids, 2)
            except:
                sampled_part_ids = [results['label_map'][0], results['label_map'][1]]
        else:
            sampled_part_ids = self.sampled_ids
        sampled_part_ids.sort()

        results['parts_pts'] = [results['parts_pts'][sampled_part_ids[0]], results['parts_pts'][sampled_part_ids[1]]]
        results['parts_gts'] = [results['parts_gts'][sampled_part_ids[0]], results['parts_gts'][sampled_part_ids[1]]]
        results['parts_cls'] = [results['parts_cls'][sampled_part_ids[0]], results['parts_cls'][sampled_part_ids[1]]]

        results['sampled_part_ids'] = sampled_part_ids
        results['n_total_points'] = results['parts_pts'][0].shape[0] + results['parts_pts'][1].shape[0]

        return results


# @PIPELINES.register_module
class LoadPairArtiNOCSDataSapien(object):

    def __call__(self, results):
        nocs_p = [None] * 2
        nocs_g = [None] * 2
        sampled_part_ids = results['sampled_part_ids']

        for j, sampled_part_id in enumerate(sampled_part_ids):
            info_index = results['label_map'].index(sampled_part_id) + 1
            norm_factor = results['norm_factors'][info_index]
            norm_corner = results['corner_pts'][info_index]
            nocs_p[j] = (results['parts_gts'][j][:, :3] - norm_corner[0]) * norm_factor + \
                        np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (
                                    norm_corner[1] - norm_corner[0]) * norm_factor

            norm_factor = results['norm_factors'][0]
            norm_corner = results['corner_pts'][0]
            nocs_g[j] = (results['parts_gts'][j][:, :3] - norm_corner[0]) * norm_factor + \
                        np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (
                                    norm_corner[1] - norm_corner[0]) * norm_factor

        results['nocs_p'] = nocs_p
        results['nocs_g'] = nocs_g

        return results

# @PIPELINES.register_module
class LoadPairArtiJointDataSapien(object):
    def __call__(self, results):
        parts_offset_joint = [None] * 2
        parts_joints = [None] * 2
        joint_index = [None] * 2
        jtype_label = [None] * 2

        norm_factor = results['norm_factors'][0] #global
        norm_corner = results['corner_pts'][0]

        joint_xyz = results['joint_ins']['xyz']
        joint_rpy = results['joint_ins']['axis']
        joint_type = results['joint_ins']['type']
        joint_parents = results['joint_ins']['parent']
        joint_children = results['joint_ins']['child']

        #eg: eyeglasses : joint_parents:[0,0] joint_children:[1,2]
        joint_parents = [results['label_map'][i] for i in joint_parents]
        joint_children = [results['label_map'][i] for i in joint_children]
        part_combinations = [[p, c] for p, c in zip(joint_parents, joint_children)]
        sampled_part_ids = results['sampled_part_ids']
        if sampled_part_ids in part_combinations or sampled_part_ids[::-1] in part_combinations:
            try:
                joint_id = part_combinations.index(sampled_part_ids)
            except:
                joint_id = part_combinations.index(sampled_part_ids[::-1])

            joint_P0 = np.array(joint_xyz[joint_id])
            joint_P0 = (joint_P0 - norm_corner[0]) * norm_factor + \
                       np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (
                               norm_corner[1] - norm_corner[0]) * norm_factor
            joint_l = np.array(joint_rpy[joint_id])
            jtype_name = joint_type[joint_id]
            for j in range(2):
                if jtype_name == 'prismatic':
                    offset_arr = np.ones_like(results['nocs_g'][j]) * 0.5 * thres_r
                else:
                    offset_arr = point_3d_offset_joint([joint_P0, joint_l], results['nocs_g'][j])
                parts_offset_joint[j] = offset_arr
                parts_joints[j] = [joint_P0, joint_l]
                joint_index[j] = joint_id + 1
                jtype_label[j] = JOINT_CLASSES.index(jtype_name)
                # joint_index[j] = JOINT_CLASSES.index(jtype)

                # plot_arrows(results['nocs_g'][j], offset_arr, [joint_P0, joint_l], title_name='NOCS {} to joint {}'.format(j, j))

        else:
            for j in range(2):
                parts_offset_joint[j] = np.zeros_like(results['nocs_g'][j])
                parts_joints[j] = [np.array([0., 0., 0.]), np.array([0.,0.,0.])]
                joint_index[j] = 0
                jtype_label[j] = 0

        results['parts_offset_joint'] = parts_offset_joint #heatmap
        results['parts_joints'] = parts_joints #xyz rpy
        results['joint_index'] = joint_index #joint index
        results['jtype_label'] = jtype_label #joint type index

        return results


# @PIPELINES.register_module
class CreatePairArtiJointGTSapien(object):
    def __call__(self, results):
        offset_heatmap = [None] * 2
        offset_unitvec = [None] * 2
        joint_orient = [None] * 2
        joint_cls = [None] * 2

        for j, offset in enumerate(results['parts_offset_joint']):
            offset_heatmap[j] = np.zeros((results['parts_gts'][j].shape[0]))
            offset_unitvec[j] = np.zeros((results['parts_gts'][j].shape[0], 3))
            joint_orient[j] = np.zeros((results['parts_gts'][j].shape[0], 3))
            joint_cls[j] = np.zeros((results['parts_gts'][j].shape[0]))

            if results['joint_index'][0] != 0:
                heatmap = np.linalg.norm(offset, axis=1)
                unitvec = offset / (heatmap.reshape(-1, 1) + epsilon)
                if results['joint_index'][j] == 1: # prismatic
                    idc = np.where(heatmap > 0)[0]
                elif results['joint_index'][j] == 2: # revolute
                    idc = np.where(heatmap < thres_r)[0]
                # idc = np.where(heatmap > 0)[0]
                offset_heatmap[j][idc] = 1 - heatmap[idc] / thres_r
                offset_unitvec[j][idc, :] = unitvec[idc, :]
                joint_orient[j][idc, :] = results['parts_joints'][j][1]
                # joint_cls[j][idc] = results['joint_index'][j]
                joint_cls[j][:] = results['joint_index'][j]

        cls_arr = np.concatenate(results['parts_cls'], axis=0)
        pts_arr = np.concatenate(results['parts_pts'], axis=0)
        if 'parts_pts_feature' in results.keys():
            pts_feature_arr = np.concatenate(results['parts_pts_feature'], axis=0)
        offset_heatmap = np.concatenate(offset_heatmap, axis=0)
        offset_unitvec = np.concatenate(offset_unitvec, axis=0)
        joint_orient = np.concatenate(joint_orient, axis=0)
        joint_cls = np.concatenate(joint_cls, axis=0)
        if results['nocs_p'][0] is not None:
            p_arr = np.concatenate(results['nocs_p'], axis=0)
        if results['nocs_g'][0] is not None:
            g_arr = np.concatenate(results['nocs_g'], axis=0)

        results['parts_cls'] = cls_arr
        results['parts_pts'] = pts_arr.astype(np.float32)
        if 'parts_pts_feature' in results.keys():
            results['parts_pts_feature'] = pts_feature_arr.astype(np.float32)
        results['offset_heatmap'] = offset_heatmap
        results['offset_unitvec'] = offset_unitvec
        results['joint_orient'] = joint_orient  #axis_per_point rpy
        results['joint_cls'] = joint_cls
        results['cls_arr'] = cls_arr
        results['nocs_p'] = p_arr.astype(np.float32)
        results['nocs_g'] = g_arr.astype(np.float32)

        return results


# @PIPELINES.register_module
class CreatePairArtiJointGTSapien2(object):
    def __call__(self, results):
        offset_heatmap = [None] * 2
        offset_unitvec = [None] * 2
        joint_orient = [None] * 2
        joint_cls = [None] * 2
        joint_axis_gt = [None] * 2

        for j, offset in enumerate(results['parts_offset_joint']):
            offset_heatmap[j] = np.zeros((results['parts_gts'][j].shape[0]))
            offset_unitvec[j] = np.zeros((results['parts_gts'][j].shape[0], 3))
            joint_orient[j] = np.zeros((results['parts_gts'][j].shape[0], 3))
            joint_cls[j] = np.zeros((results['parts_gts'][j].shape[0]))

            if results['joint_index'][0] != 0:
                heatmap = np.linalg.norm(offset, axis=1)
                unitvec = offset / (heatmap.reshape(-1, 1) + epsilon)
                # if results['joint_index'][j] == 1: # prismatic
                #     idc = np.where(heatmap > 0)[0]
                # elif results['joint_index'][j] == 2: # revolute
                #     idc = np.where(heatmap < thres_r)[0]
                idc = np.where(heatmap > 0)[0]
                offset_heatmap[j][idc] = 1 - heatmap[idc] / thres_r
                offset_unitvec[j][idc, :] = unitvec[idc, :]
                joint_orient[j][idc, :] = results['parts_joints'][j][1]
                # joint_cls[j][idc] = results['joint_index'][j]
                joint_axis_gt[j] = results['parts_joints'][j][1]
                joint_cls[j][:] = results['joint_index'][j]

        results['offset_heatmap'] = offset_heatmap
        results['offset_unitvec'] = offset_unitvec
        results['joint_orient'] = joint_orient
        results['joint_axis_gt'] = joint_axis_gt
        results['joint_cls'] = joint_cls

        return results


# @PIPELINES.register_module
class DownSamplePairArtiSapien(object):
    def __init__(self, num_points=1024, linspace=False):
        self.num_points = num_points
        self.linspace = linspace

    def __call__(self, results):
        if results['n_total_points'] < self.num_points:
            tile_n = int(self.num_points / results['n_total_points']) + 1
            results['n_total_points'] = tile_n * results['n_total_points']
            results['parts_cls'] = np.concatenate([results['parts_cls']] * tile_n, axis=0)
            results['parts_pts'] = np.concatenate([results['parts_pts']] * tile_n, axis=0)
            if 'parts_pts_feature' in results.keys():
                results['parts_pts_feature'] = np.concatenate([results['parts_pts_feature']] * tile_n, axis=0)
            results['offset_heatmap'] = np.concatenate([results['offset_heatmap']] * tile_n, axis=0)
            results['offset_unitvec'] = np.concatenate([results['offset_unitvec']] * tile_n, axis=0)
            results['joint_orient'] = np.concatenate([results['joint_orient']] * tile_n, axis=0)
            results['joint_cls']  = np.concatenate([results['joint_cls']] * tile_n, axis=0)
            if results['nocs_p'][0] is not None:
                results['nocs_p'] = np.concatenate([results['nocs_p']] * tile_n, axis=0)
            if results['nocs_g'][0] is not None:
                results['nocs_g'] = np.concatenate([results['nocs_g']] * tile_n, axis=0)

        # if results['n_total_points'] > self.num_points:
        mask_array = np.zeros([self.num_points, results['n_max_parts']], dtype=np.float32)
        if self.linspace:
            perm = np.linspace(0, results['n_total_points']-1, self.num_points).astype(np.uint16)
        else:
            perm = np.random.permutation(results['n_total_points'])
        results['parts_cls'] = results['parts_cls'][perm[:self.num_points]].reshape(self.num_points, 1).astype(np.float32)

        results['parts_pts'] = results['parts_pts'][perm[:self.num_points]]
        if 'parts_pts_feature' in results.keys():
            results['parts_pts_feature'] = results['parts_pts_feature'][perm[:self.num_points]]
        results['offset_heatmap'] = results['offset_heatmap'][perm[:self.num_points]].reshape(self.num_points, 1).astype(np.float32)
        results['offset_unitvec'] = results['offset_unitvec'][perm[:self.num_points]].astype(np.float32)
        results['joint_orient'] = results['joint_orient'][perm[:self.num_points]].astype(np.float32)
        results['joint_cls'] = results['joint_cls'][perm[:self.num_points]].reshape(self.num_points, 1).astype(np.float32)
        if 'part_relation' in results.keys():
            results['part_relation'] = results['part_relation'][perm[:self.num_points]]
        # print('joint_cls_arr has shape: ', joint_cls_arr.shape)
        joint_cls_mask = np.zeros((results['joint_cls'].shape[0]), dtype=np.float32).reshape(self.num_points, 1)
        id_valid = np.where(results['offset_heatmap'] > 0)[0]
        joint_cls_mask[id_valid] = results['joint_index'][0]
        mask_array[np.arange(self.num_points), results['parts_cls'][:, 0].astype(np.int8)] = 1.0
        results['mask_array'] = mask_array
        id_object = np.where(results['parts_cls'] > 0)[0]

        if results['nocs_p'][0] is not None:
            results['nocs_p'] = results['nocs_p'][perm[:self.num_points]]
        if results['nocs_g'][0] is not None:
            results['nocs_g'] = results['nocs_g'][perm[:self.num_points]]

        results['joint_cls_mask'] = joint_cls_mask
        return results


# @PIPELINES.register_module
class DownSamplePairArtiSapien2(object):
    def __init__(self, num_points=1024, linspace=False):
        self.num_points = num_points
        self.linspace = linspace

    def __call__(self, results):
        results['joint_cls_mask'] = [[] for _ in range(2)]
        for j in range(2):
            part_num_points = results['parts_pts'][j].shape[0]
            if part_num_points < self.num_points:
                tile_n = int(self.num_points / part_num_points) + 1
                results['parts_cls'][j] = np.concatenate([results['parts_cls'][j]] * tile_n, axis=0)
                results['parts_pts'][j] = np.concatenate([results['parts_pts'][j]] * tile_n, axis=0)
                results['offset_heatmap'][j] = np.concatenate([results['offset_heatmap'][j]] * tile_n, axis=0)
                results['offset_unitvec'][j] = np.concatenate([results['offset_unitvec'][j]] * tile_n, axis=0)
                results['joint_orient'][j] = np.concatenate([results['joint_orient'][j]] * tile_n, axis=0)
                results['joint_cls'][j] = np.concatenate([results['joint_cls'][j]] * tile_n, axis=0)

                results['nocs_p'][j] = np.concatenate([results['nocs_p'][j]] * tile_n, axis=0)
                results['nocs_g'][j] = np.concatenate([results['nocs_g'][j]] * tile_n, axis=0)

                part_num_points = tile_n * part_num_points

            perm = np.random.permutation(part_num_points)
            results['parts_cls'][j] = results['parts_cls'][j][perm[:self.num_points]].reshape(self.num_points, 1).astype(
                np.float32)
            results['parts_pts'][j] = results['parts_pts'][j][perm[:self.num_points]].astype(np.float32)
            results['offset_heatmap'][j] = results['offset_heatmap'][j][perm[:self.num_points]].reshape(self.num_points, 1).astype(np.float32)
            results['offset_unitvec'][j] = results['offset_unitvec'][j][perm[:self.num_points]].astype(np.float32)
            results['joint_orient'][j] = results['joint_orient'][j][perm[:self.num_points]].astype(np.float32)
            results['joint_cls'][j] = results['joint_cls'][j][perm[:self.num_points]].reshape(self.num_points, 1).astype(
                np.float32)

            joint_cls_mask = np.zeros((results['joint_cls'][j].shape[0]), dtype=np.float32).reshape(self.num_points, 1)
            id_valid = np.where(results['offset_heatmap'][j] > 0)[0]
            joint_cls_mask[id_valid] = 1.0
            results['nocs_p'][j] = results['nocs_p'][j][perm[:self.num_points]]
            results['nocs_g'][j] = results['nocs_g'][j][perm[:self.num_points]]

            results['joint_cls_mask'][j] = joint_cls_mask

        if isinstance(results['parts_pts'], list):
            results['parts_pts'] = np.concatenate(results['parts_pts'], axis=0)
        if isinstance(results['parts_cls'], list):
            results['parts_cls'] = np.concatenate(results['parts_cls'], axis=0)
        if isinstance(results['offset_heatmap'], list):
            results['offset_heatmap'] = np.concatenate(results['offset_heatmap'], axis=0)
        if isinstance(results['offset_unitvec'], list):
            results['offset_unitvec'] = np.concatenate(results['offset_unitvec'], axis=0)
        if isinstance(results['joint_orient'], list):
            results['joint_orient'] = np.concatenate(results['joint_orient'], axis=0)
        if isinstance(results['joint_cls'], list):
            results['joint_cls'] = np.concatenate(results['joint_cls'], axis=0)
        if isinstance(results['joint_cls_mask'], list):
            results['joint_cls_mask'] = np.concatenate(results['joint_cls_mask'], axis=0)
        if isinstance(results['nocs_p'], list):
            results['nocs_p'] = np.concatenate(results['nocs_p'], axis=0)
        if isinstance(results['nocs_g'], list):
            results['nocs_g'] = np.concatenate(results['nocs_g'], axis=0)

        mask_array = np.zeros([self.num_points, results['n_max_parts']], dtype=np.float32)
        mask_array[np.arange(self.num_points), results['parts_cls'][:, 0].astype(np.int8)] = 1.0
        results['mask_array'] = mask_array

        return results


# @PIPELINES.register_module
class LoadArtiNOCSDataSapien(object):

    def __call__(self, results):
        nocs_p = [None] * results['n_parts']
        nocs_g = [None] * results['n_parts']
        for j in range(results['n_parts']):
            norm_factor = results['norm_factors'][j+1]
            norm_corner = results['corner_pts'][j+1]
            nocs_p[j] = (results['parts_gts'][j][:, :3] - norm_corner[0]) * norm_factor + \
                        np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (
                                    norm_corner[1] - norm_corner[0]) * norm_factor
            #assert nocs_p[j].min() > 0
            #assert nocs_p[j].max() < 1

            norm_factor = results['norm_factors'][0]
            norm_corner = results['corner_pts'][0]
            nocs_g[j] = (results['parts_gts'][j][:, :3] - norm_corner[0]) * norm_factor + \
                        np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (
                                    norm_corner[1] - norm_corner[0]) * norm_factor
            #assert nocs_g[j].min() > 0
            #assert nocs_g[j].max() < 1
        # plot3d_pts([nocs_g], [['0', '1', '2', '3', '4']], s=5, title_name=['pts'])

        results['nocs_p'] = nocs_p
        results['nocs_g'] = nocs_g

        return results

    def __repr__(self):
        return self.__class__.__name__


# @PIPELINES.register_module
class LoadArtiJointDataSapien_scale(object):

    def __call__(self, results, scale):
        parts_offset_joint = [[] for _ in range(results['n_parts'])] #[]*n_parts
        parts_joints = [[] for _ in range(results['n_parts'])]
        joint_index = [[] for _ in range(results['n_parts'])]
        joint_xyz = np.array(results['joint_ins']['xyz']) /scale
        joint_rpy = np.array(results['joint_ins']['axis'])

        gt_poses = results['gt_poses']
        for i in range(results['n_parts']):
            gt_pose = gt_poses[i]
            joint_xyz[i] = np.dot(joint_xyz[i], gt_pose[:3, :3].T) + gt_pose[:3, 3].T
            joint_rpy[i] = np.dot(joint_rpy[i], gt_pose[:3, :3].T)

        camera_pts = results['camera_pts']
        joint_type = results['joint_ins']['type']
        joint_params = np.zeros((results['n_parts'], 7))
        joint_type_dict = {'prismatic': 0, 'revolute': 1}

        for j in range(results['n_parts']):
            if j > 0:
                joint_P0 = np.array(joint_xyz[j])
                joint_l = np.array(joint_rpy[j])
                if joint_type[j] == 'prismatic':
                    orth_vect = np.ones_like(np.array([0, 0, 0]).reshape(1, 3)) * 0.5 * thres_r
                else:
                    orth_vect = point_3d_offset_joint([joint_P0, joint_l], np.array([0, 0, 0]).reshape(1, 3))
                joint_params[j, 0:3] = joint_l
                joint_params[j, 6] = np.linalg.norm(orth_vect)
                joint_params[j, 3:6] = orth_vect / joint_params[j, 6]

            if results['parts_parent_joint'][j] != 0:
                joint_P0 = np.array(joint_xyz[results['parts_parent_joint'][j]])
                joint_l = np.array(joint_rpy[j])
                if joint_type[j] == 'prismatic':
                    offset_arr = np.ones_like(results['camera_pts'][j]) * 0.5 * thres_r
                else:
                    offset_arr = point_3d_offset_joint([joint_P0, joint_l], results['camera_pts'][j])

                parts_offset_joint[j].append(offset_arr)
                parts_joints[j].append([joint_P0, joint_l])
                joint_index[j].append(results['parts_parent_joint'][j])
                # plot_arrows(results['nocs_g'][j], offset_arr, [joint_P0, joint_l], title_name='NOCS {} to joint {}'.format(j, j))

            if results['parts_child_joint'][j] is not None:
                for m in results['parts_child_joint'][j]:
                    joint_P0 = np.array(joint_xyz[m])
                    joint_l = np.array(joint_rpy[m])
                    if joint_type[j] == 'prismatic':
                        offset_arr = np.ones_like(results['camera_pts'][j]) * 0.5 * thres_r
                    else:
                        offset_arr = point_3d_offset_joint([joint_P0, joint_l], results['camera_pts'][j])
                    parts_offset_joint[j].append(offset_arr)
                    parts_joints[j].append([joint_P0, joint_l])
                    joint_index[j].append(m)
                    # plot_arrows(results['nocs_g'][j], offset_arr, [joint_P0, joint_l], title_name='NOCS {} to joint {}'.format(j, m))
        # encode joint type
        joint_type_gt = np.zeros(results['n_max_parts'])
        for i, part_joint_type in enumerate(joint_type):
            if 'label_map' in results.keys():
                part_id = results['label_map'][i]
            else:
                part_id = i
            joint_type_gt[part_id] = joint_type_dict[part_joint_type] if part_joint_type is not None else -1

        results['parts_offset_joint'] = parts_offset_joint
        results['parts_joints'] = parts_joints
        results['joint_index'] = joint_index
        results['joint_params'] = joint_params
        results['joint_type_gt'] = joint_type_gt

        return results

    def __repr__(self):
        return self.__class__.__name__

# @PIPELINES.register_module
class LoadArtiJointDataSapien(object):

    def __call__(self, results):
        parts_offset_joint = [[] for _ in range(results['n_parts'])] #[]*n_parts
        parts_joints = [[] for _ in range(results['n_parts'])]
        joint_index = [[] for _ in range(results['n_parts'])]
        joint_xyz = results['joint_ins']['xyz']
        joint_rpy = results['joint_ins']['axis']
        joint_type = results['joint_ins']['type']
        joint_params = np.zeros((results['n_parts'], 7))
        joint_type_dict = {'prismatic': 0, 'revolute': 1}

        for j in range(results['n_parts']):
            norm_factor = results['norm_factors'][0]
            norm_corner = results['corner_pts'][0]

            if j > 0:
                joint_P0 = np.array(joint_xyz[j])
                joint_P0 = (joint_P0 - norm_corner[0]) * norm_factor + \
                           np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (
                                   norm_corner[1] - norm_corner[0]) * norm_factor
                joint_l = np.array(joint_rpy[j])
                if joint_type[j] == 'prismatic':
                    orth_vect = np.ones_like(np.array([0, 0, 0]).reshape(1, 3)) * 0.5 * thres_r
                else:
                    orth_vect = point_3d_offset_joint([joint_P0, joint_l], np.array([0, 0, 0]).reshape(1, 3))
                joint_params[j, 0:3] = joint_l
                joint_params[j, 6] = np.linalg.norm(orth_vect)
                joint_params[j, 3:6] = orth_vect / joint_params[j, 6]

            if results['parts_parent_joint'][j] != 0:
                joint_P0 = np.array(joint_xyz[results['parts_parent_joint'][j]])
                joint_P0 = (joint_P0 - norm_corner[0]) * norm_factor + \
                           np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (
                                   norm_corner[1] - norm_corner[0]) * norm_factor
                joint_l = np.array(joint_rpy[j])
                if joint_type[j] == 'prismatic':
                    offset_arr = np.ones_like(results['nocs_g'][j]) * 0.5 * thres_r
                else:
                    offset_arr = point_3d_offset_joint([joint_P0, joint_l], results['nocs_g'][j])
                parts_offset_joint[j].append(offset_arr)
                parts_joints[j].append([joint_P0, joint_l])
                joint_index[j].append(results['parts_parent_joint'][j])
                # plot_arrows(results['nocs_g'][j], offset_arr, [joint_P0, joint_l], title_name='NOCS {} to joint {}'.format(j, j))

            if results['parts_child_joint'][j] is not None:
                for m in results['parts_child_joint'][j]:
                    joint_P0 = np.array(joint_xyz[m])
                    joint_P0 = (joint_P0 - norm_corner[0]) * norm_factor + \
                               np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (
                                       norm_corner[1] - norm_corner[0]) * norm_factor
                    joint_l = np.array(joint_rpy[m])
                    if joint_type[j] == 'prismatic':
                        offset_arr = np.ones_like(results['nocs_g'][j]) * 0.5 * thres_r
                    else:
                        offset_arr = point_3d_offset_joint([joint_P0, joint_l], results['nocs_g'][j])
                    parts_offset_joint[j].append(offset_arr)
                    parts_joints[j].append([joint_P0, joint_l])
                    joint_index[j].append(m)
                    # plot_arrows(results['nocs_g'][j], offset_arr, [joint_P0, joint_l], title_name='NOCS {} to joint {}'.format(j, m))
        # encode joint type
        joint_type_gt = np.zeros(results['n_max_parts'])
        for i, part_joint_type in enumerate(joint_type):
            if 'label_map' in results.keys():
                part_id = results['label_map'][i]
            else:
                part_id = i
            joint_type_gt[part_id] = joint_type_dict[part_joint_type] if part_joint_type is not None else -1

        results['parts_offset_joint'] = parts_offset_joint
        results['parts_joints'] = parts_joints
        results['joint_index'] = joint_index
        results['joint_params'] = joint_params
        results['joint_type_gt'] = joint_type_gt

        return results

    def __repr__(self):
        return self.__class__.__name__

# @PIPELINES.register_module
class CreateArtiJointGTSapien_scale(object):

    def __call__(self, results):
        offset_heatmap = [None] * results['n_parts']
        offset_unitvec = [None] * results['n_parts']
        joint_orient = [None] * results['n_parts']
        joint_cls = [None] * results['n_parts']
        for j, offsets in enumerate(results['parts_offset_joint']):
            offset_heatmap[j] = np.zeros((results['camera_pts'][j].shape[0]))
            offset_unitvec[j] = np.zeros((results['camera_pts'][j].shape[0], 3))
            joint_orient[j] = np.zeros((results['camera_pts'][j].shape[0], 3))
            joint_cls[j] = np.zeros((results['camera_pts'][j].shape[0]))
            for k, offset in enumerate(offsets):
                heatmap = np.linalg.norm(offset, axis=1)
                unitvec = offset / (heatmap.reshape(-1, 1) + epsilon)
                idc = np.where(heatmap < thres_r)[0]
                offset_heatmap[j][idc] = 1 - heatmap[idc] / thres_r
                offset_unitvec[j][idc, :] = unitvec[idc, :]
                joint_orient[j][idc, :] = results['parts_joints'][j][k][1]
                if 'label_map' in results.keys():
                    joint_cls[j][idc] = results['label_map'][results['joint_index'][j][k]]
                else:
                    joint_cls[j][idc] = results['joint_index'][j][k]

        cls_arr = np.concatenate(results['parts_cls'], axis=0)
        # pts_arr = np.concatenate(results['parts_pts'], axis=0)
        if 'parts_pts_feature' in results.keys():
            pts_feature_arr = np.concatenate(results['parts_pts_feature'], axis=0)
        offset_heatmap = np.concatenate(offset_heatmap, axis=0)
        offset_unitvec = np.concatenate(offset_unitvec, axis=0)
        joint_orient = np.concatenate(joint_orient, axis=0)
        joint_cls = np.concatenate(joint_cls, axis=0)

        # results['parts_pts'] = pts_arr.astype(np.float32)
        if 'parts_pts_feature' in results.keys():
            results['parts_pts_feature'] = pts_feature_arr.astype(np.float32)
        results['offset_heatmap'] = offset_heatmap
        results['offset_unitvec'] = offset_unitvec
        results['joint_orient'] = joint_orient
        results['joint_cls'] = joint_cls
        return results

    def __repr__(self):
        return self.__class__.__name__

# @PIPELINES.register_module
class CreateArtiJointGTSapien(object):

    def __call__(self, results):
        offset_heatmap = [None] * results['n_parts']
        offset_unitvec = [None] * results['n_parts']
        joint_orient = [None] * results['n_parts']
        joint_cls = [None] * results['n_parts']
        for j, offsets in enumerate(results['parts_offset_joint']):
            offset_heatmap[j] = np.zeros((results['parts_gts'][j].shape[0]))
            offset_unitvec[j] = np.zeros((results['parts_gts'][j].shape[0], 3))
            joint_orient[j] = np.zeros((results['parts_gts'][j].shape[0], 3))
            joint_cls[j] = np.zeros((results['parts_gts'][j].shape[0]))
            for k, offset in enumerate(offsets):
                heatmap = np.linalg.norm(offset, axis=1)
                unitvec = offset / (heatmap.reshape(-1, 1) + epsilon)
                idc = np.where(heatmap < thres_r)[0]
                offset_heatmap[j][idc] = 1 - heatmap[idc] / thres_r
                offset_unitvec[j][idc, :] = unitvec[idc, :]
                joint_orient[j][idc, :] = results['parts_joints'][j][k][1]
                if 'label_map' in results.keys():
                    joint_cls[j][idc] = results['label_map'][results['joint_index'][j][k]]
                else:
                    joint_cls[j][idc] = results['joint_index'][j][k]

        cls_arr = np.concatenate(results['parts_cls'], axis=0)
        # pts_arr = np.concatenate(results['parts_pts'], axis=0)
        if 'parts_pts_feature' in results.keys():
            pts_feature_arr = np.concatenate(results['parts_pts_feature'], axis=0)
        offset_heatmap = np.concatenate(offset_heatmap, axis=0)
        offset_unitvec = np.concatenate(offset_unitvec, axis=0)
        joint_orient = np.concatenate(joint_orient, axis=0)
        joint_cls = np.concatenate(joint_cls, axis=0)
        if results['nocs_p'][0] is not None:
            p_arr = np.concatenate(results['nocs_p'], axis=0)
        if results['nocs_g'][0] is not None:
            g_arr = np.concatenate(results['nocs_g'], axis=0)

        results['parts_cls'] = cls_arr
        # results['parts_pts'] = pts_arr.astype(np.float32)
        if 'parts_pts_feature' in results.keys():
            results['parts_pts_feature'] = pts_feature_arr.astype(np.float32)
        results['offset_heatmap'] = offset_heatmap
        results['offset_unitvec'] = offset_unitvec
        results['joint_orient'] = joint_orient
        results['joint_cls'] = joint_cls
        results['cls_arr'] = cls_arr
        results['nocs_p'] = p_arr.astype(np.float32)
        results['nocs_g'] = g_arr.astype(np.float32)
        return results

    def __repr__(self):
        return self.__class__.__name__


# @PIPELINES.register_module
class CreatePartRelationGTSapien(object):
    def __init__(self, joint_types=('none', 'revolute', 'prismatic')):
        self.joint_types = joint_types

    def __call__(self, results):
        n_parts = results['n_parts']
        n_possible_joint = n_parts ** 2

        joint_type = results['joint_ins']['type']
        joint_parent = results['joint_ins']['parent']
        joint_child = results['joint_ins']['child']
        n_joints = len(joint_parent)

        part_relation = np.zeros([results['n_total_points'], n_possible_joint], dtype=np.float32)
        for j in range(n_joints):
            if j == 0:
                continue
            parent_id = joint_parent[j]
            child_id = joint_child[j]
            type_id = self.joint_types.index(joint_type[j])

            idc = np.where(results['parts_cls'] == parent_id)[0]

            part_relation[idc, parent_id * n_parts + child_id] = type_id
            part_relation[idc, child_id * n_parts + parent_id] = type_id

        results['part_relation'] = part_relation

        return results


# @PIPELINES.register_module
class CreateSegmentationSapien(object):
    def __call__(self, results):
        ann_file = osp.join(results['img_prefix'],
                            'annotations',
                            results['sample_name'] + '.json')
        ann_info = json.load(open(ann_file))
        instance_info = ann_info['instances'][0]
        n_parts = len(instance_info['links'])
        img_height = ann_info['height']
        img_width = ann_info['width']
        mask = np.zeros((img_height, img_width))
        for j in range(n_parts):
            part_seg = instance_info['links'][j]['segmentation']
            try:
                rle = maskUtils.frPyObjects(part_seg, img_height, img_width)
                part_mask = np.sum(maskUtils.decode(rle), axis=2).clip(max=1).astype(np.uint8)
                # part id start from 1, bg id is 0
                if results['label_map'] is not None:
                    label_id = results['label_map'][j] + 1
                else:
                    label_id = j + 1
                part_mask[part_mask == 1] *= label_id
                # filter the overlap mask positions
                label_positions = np.where(mask != 0)
                part_mask[label_positions] = 0
                mask += part_mask
            except:
                warnings.warn(osp.join(results['img_prefix'] + results['sample_name']))
        # vis segmentation mask
        # import seaborn as sns
        # import cv2
        # color = sns.color_palette("hls", results['n_max_parts'])
        # seg_color = np.zeros((img_height, img_width, 3))
        # labels = np.unique(mask)
        # for l in labels:
        #     seg_color[mask == l] = color[int(l)]
        # cv2.imwrite('seg_color.jpg', seg_color * 255)

        # maske part id start from 0 and bg id to be num_part + 1
        mask -= 1
        mask[mask == -1] = results['n_max_parts']

        results['gt_semantic_seg'] = mask.astype(np.int64)

        return results


# @PIPELINES.register_module
class DownSampleArtiSapien(object):
    def __init__(self, num_points=1024, linspace=False, point_norm=False):
        self.num_points = num_points
        self.linspace = linspace
        self.point_norm = point_norm

    def __call__(self, results):
        if results['n_total_points'] < self.num_points:
            tile_n = int(self.num_points / results['n_total_points']) + 1
            results['n_total_points'] = tile_n * results['n_total_points']
            results['parts_cls'] = np.concatenate([results['parts_cls']] * tile_n, axis=0)
            results['parts_pts'] = np.concatenate([results['parts_pts']] * tile_n, axis=0)
            if 'parts_pts_feature' in results.keys():
                results['parts_pts_feature'] = np.concatenate([results['parts_pts_feature']] * tile_n, axis=0)
            results['offset_heatmap'] = np.concatenate([results['offset_heatmap']] * tile_n, axis=0)
            results['offset_unitvec'] = np.concatenate([results['offset_unitvec']] * tile_n, axis=0)
            results['joint_orient'] = np.concatenate([results['joint_orient']] * tile_n, axis=0)
            results['joint_cls']  = np.concatenate([results['joint_cls']] * tile_n, axis=0)
            if results['nocs_p'][0] is not None:
                results['nocs_p'] = np.concatenate([results['nocs_p']] * tile_n, axis=0)
            if results['nocs_g'][0] is not None:
                results['nocs_g'] = np.concatenate([results['nocs_g']] * tile_n, axis=0)

        # if results['n_total_points'] > self.num_points:
        mask_array = np.zeros([self.num_points, results['n_max_parts']], dtype=np.float32)
        if self.linspace:
            perm = np.linspace(0, results['n_total_points']-1, self.num_points).astype(np.uint16)
        else:
            perm = np.random.permutation(results['n_total_points'])
        results['parts_cls'] = results['parts_cls'][perm[:self.num_points]].reshape(self.num_points, 1).astype(np.float32)

        results['parts_pts'] = results['parts_pts'][perm[:self.num_points]]
        if self.point_norm:
            results['points_mean'] = results['parts_pts'].mean(axis=0)
            results['parts_pts'] -= results['points_mean']
        # results['parts_pts'] = results['parts_pts'][perm[:self.num_points]] * results['norm_factors'][0]
        if 'parts_pts_feature' in results.keys():
            results['parts_pts_feature'] = results['parts_pts_feature'][perm[:self.num_points]]
        results['offset_heatmap'] = results['offset_heatmap'][perm[:self.num_points]].reshape(self.num_points, 1).astype(np.float32)
        results['offset_unitvec'] = results['offset_unitvec'][perm[:self.num_points]].astype(np.float32)
        results['joint_orient'] = results['joint_orient'][perm[:self.num_points]].astype(np.float32)
        results['joint_cls'] = results['joint_cls'][perm[:self.num_points]].reshape(self.num_points, 1).astype(np.float32)
        if 'part_relation' in results.keys():
            results['part_relation'] = results['part_relation'][perm[:self.num_points]]
        # print('joint_cls_arr has shape: ', joint_cls_arr.shape)
        joint_cls_mask = np.zeros((results['joint_cls'].shape[0]), dtype=np.float32).reshape(self.num_points, 1)
        id_valid = np.where(results['joint_cls'] > 0)[0]
        joint_cls_mask[id_valid] = 1.0
        mask_array[np.arange(self.num_points), results['parts_cls'][:, 0].astype(np.int8)] = 1.0
        results['mask_array'] = mask_array
        id_object = np.where(results['parts_cls'] > 0)[0]

        if results['nocs_p'][0] is not None:
            results['nocs_p'] = results['nocs_p'][perm[:self.num_points]]
        if results['nocs_g'][0] is not None:
            results['nocs_g'] = results['nocs_g'][perm[:self.num_points]]

        results['joint_cls_mask'] = joint_cls_mask

        joint_type_mask = np.zeros((results['joint_cls'].shape[0]), dtype=np.float32).reshape(self.num_points, 1)
        for type_id in range(results['joint_type_gt'].shape[0]):
            if type_id ==0:
                continue
            type_id_valid = np.where(results['joint_cls'] == type_id)[0]
            joint_type_mask[type_id_valid] = results['joint_type_gt'][type_id]
        results['joint_type_mask'] = joint_type_mask
        return results

    def __repr__(self):
        return self.__class__.__name__


# @PIPELINES.register_module
class CreatePartPointsSapien(object):
    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, results):
        parts_cls = results['parts_cls']
        parts_pts = results['parts_pts']
        n_max_parts = results['n_max_parts']
        points_per_part = np.zeros((n_max_parts, self.num_points, 3)).astype(np.float32)
        moving_part_ids = []
        for part_id in np.unique(parts_cls):
            # if part_id == 0:
            #     continue
            inds = parts_cls == part_id
            P_part = parts_pts[inds.squeeze(-1)]
            if P_part.shape[0] < self.num_points:
                tile_n = int(self.num_points / P_part.shape[0]) + 1
                P_part = np.concatenate([P_part] * tile_n, axis=0)
            perm = np.random.permutation(P_part.shape[0])
            points_per_part[int(part_id)] = np.expand_dims(P_part[perm[:self.num_points]], 0)
            moving_part_ids.append(int(part_id))

        results['points_per_part'] = points_per_part
        results['moving_part_ids'] = moving_part_ids
        return results