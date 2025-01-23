from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import random
import copy
import torchvision.transforms as transforms
import os
import torch
import open3d as o3d
import arti_utils
import pipelines_sapien as pipelines

def read_depth(path):
    depth = np.asarray(Image.open(path)).copy()
    return depth

def get_normal(cld):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cld)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)
    return normals

def dpt_2_cld(dpt, cam_scale, K, dname):
    if dname=='Art':
        xmap = np.array([[j for i in range(640)] for j in range(640)])
        ymap = np.array([[i for i in range(640)] for j in range(640)])
    elif dname=='ReArt':
        xmap = np.array([[j for i in range(1280)] for j in range(720)])
        ymap = np.array([[i for i in range(1280)] for j in range(720)])
    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]
    msk_dp = dpt > 1e-6
    choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
    if len(choose) < 1:
        return None, None

    dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_mskd = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_mskd = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

    pt2 = dpt_mskd / cam_scale
    cam_cx, cam_cy = K[0][2], K[1][2]
    cam_fx, cam_fy = K[0][0], K[1][1]
    pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
    cld = np.concatenate((pt0, pt1, pt2), axis=1)
    return cld, choose

def scale_pts(cld_dict, results, scale, part_num):
    urdf_id = results['urdf_id']

    part_trans_rest = results['part_trans_rest']
    part_trans = results['part_trans']
    n_parts = results['n_parts']

    scale_pts_list = []
    scale_pts_rest_list = []
    scale_camera_pts_list = []
    pcd_scale_list = []

    tran = part_trans[0]
    for i in range(n_parts):
        tran_rest = part_trans_rest[i]
        pts = cld_dict[i]
        pts_rest = cld_dict[i]
        pcd_rest = o3d.geometry.PointCloud()
        pcd = o3d.geometry.PointCloud()
        pcd_scale = o3d.geometry.PointCloud()
        pcd_camera = o3d.geometry.PointCloud()

        pcd_camera.points = o3d.utility.Vector3dVector(pts)

        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd_copy = copy.deepcopy(pcd)
        pcd_copy.transform(np.linalg.inv(tran))
        scale_pts = np.asarray(pcd_copy.points) / scale
        scale_pts_list.append(scale_pts)

        pcd_rest.points = o3d.utility.Vector3dVector(pts_rest)
        pcd_rest_copy = copy.deepcopy(pcd_rest)
        pcd_rest_copy.transform(np.linalg.inv(tran_rest))
        scale_pts_rest = np.asarray(pcd_rest_copy.points) / scale
        scale_pts_rest_list.append(scale_pts_rest)

        pcd_scale.points = o3d.utility.Vector3dVector(scale_pts)
        pcd_scale_copy = copy.deepcopy(pcd_scale)
        pcd_scale_copy.transform(tran)
        scale_camera_pts_list.append(np.asarray(pcd_scale_copy.points))
        pcd_scale_list.append(pcd_scale_copy)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    part_trans_scale = [None] * n_parts

    for i in range(n_parts):
        scale_pts1 = scale_pts_rest_list[i]
        scale_camera_pts1 = scale_camera_pts_list[i]
        pcd_scale = pcd_scale_list[i]
        part_tran = part_trans_rest[i]
        if i!=0:
            R12 = part_tran[:3, :3]
            t12 = arti_utils.translation_pts(scale_pts1, scale_camera_pts1, 1, R12)
            T12 = np.eye(4)
            T12[:3, :3] = R12
            T12[:3, 3] = t12
            part_trans_scale[i] = T12
        else:
            part_trans_scale[i] = part_tran

    if scale_camera_pts_list[part_num].size == 0:
        print('Error', scale_camera_pts_list[part_num].shape)
        xyz = np.array([0., 0., 0.]).reshape(1, 3)
        RT = [np.eye(4) for _ in range(n_parts)]
    else:
        xyz = scale_camera_pts_list[part_num]
        RT = part_trans_scale

    xyz = np.vstack(scale_camera_pts_list)

    return xyz, RT, scale_camera_pts_list

class RMapDataset(Dataset):

    def __init__(self, root, dname, set, obj_name, transform, part_num, kpt_num, kpt_class, n_sample_points, kpts_path):
        self.root = root
        self.set = set
        self.transform3d = transform
        self.obj_name = obj_name
        self.dname = dname
        self.part_num = part_num
        self.kpt_num = kpt_num
        self.kpt_class = kpt_class
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.n_sample_points = n_sample_points
        self.depth_scale = 1000
        self.kpts_path = kpts_path
        self.Art_K = np.array([[914., 0., 320.],
                      [0., 914., 320.],
                      [0., 0., 1.]])
        self.ReArt_K = np.array([[908.92468262, 0., 651.9619750976562],
                                [0., 909.5477294921875, 351.6006164550781],
                                [0., 0., 1.]])

        if self.set == 'train':
            self.add_noise = True
        else:
            self.add_noise = False

        if self.dname == 'Art':
            self._imgsetpath = os.path.join(self.root, self.obj_name, '%s.txt')
            if kpt_class=='KP':
                self.kpts = np.load(os.path.join('logs_kp/ArtImage/radii', f'{self.obj_name}_{self.part_num}_{self.kpt_num}/{self.obj_name}_{self.part_num}_mean_points.npy'))
            with open(self._imgsetpath % self.set) as f:
                self.ids = f.readlines()
            self.ids = [x.strip().replace('.json', '') for x in self.ids]
        if self.dname == 'ReArt':
            if kpt_class=='KP':
                self.kpts = np.load(os.path.join('logs_kp/ReArtMix/radii', f'{self.obj_name}_{self.part_num}_{self.kpt_num}/{self.obj_name}_{self.part_num}_mean_points.npy'))
            self.ids = []
            self.datapath = []
            if self.obj_name == 'box':
                self.root1 = os.path.join(root, 'box_train_v2/annotations')
                self.root2 = os.path.join(root, 'box_train_v2/annotations')
            elif self.obj_name == 'scissor':
                self.root1 = os.path.join(root, 'train_scissor/annotations')
                self.root2 = os.path.join(root, 'scissor_val/annotations')
            else:
                self.root1 = os.path.join(root, f'{self.obj_name}_train/annotations')
                self.root2 = os.path.join(root, f'{self.obj_name}_val/annotations')

            train_ids = [f"{i:06d}" for i in range(1, 5000)]
            if self.obj_name == 'box':
                test_ids = [f"{i:06d}" for i in range(5000, 6000)]
            else:
                test_ids = [f"{i:06d}" for i in range(1, 1200)]

            if self.set == 'train':
                self.set_ids = train_ids
                for scenename in os.listdir(os.path.join(self.root1)):
                    if os.path.isdir(os.path.join(self.root1, scenename)):
                        file_path = os.path.join(self.root1, scenename)
                        for file in os.listdir(file_path):
                            file_name = os.path.join(file_path, file)
                            id=file.split('.')[0]
                            if id in train_ids:
                                self.ids.append(file_name)
            else:
                self.set_ids = test_ids
                for scenename in os.listdir(os.path.join(self.root2)):
                    if os.path.isdir(os.path.join(self.root2, scenename)):
                        file_path = os.path.join(self.root2, scenename)
                        for file in os.listdir(file_path):
                            file_name = os.path.join(file_path, file)
                            id=file.split('.')[0]
                            if id in test_ids:
                                self.ids.append(file_name)

    def __getitem__(self, item):
        if self.dname == 'Art':
            id = self.ids[item]
            self.cycle_path = os.path.join(self.root, f'{self.obj_name}/train')
            ann_json_path = os.path.join(self.cycle_path, f'annotations/{id}.json')
            joint_param_path = os.path.join(self.root, f'urdf_metas/{self.obj_name}/urdf_metas.json')
        elif self.dname == 'ReArt':
            ann_json_path = self.ids[item]
            id = ann_json_path.split('/')[-1].split('.')[0]
            scene_name = os.path.basename(os.path.dirname(ann_json_path))
            joint_param_path = os.path.join(self.root, 'urdf_metas.json')
            self.cycle_path = os.path.dirname(os.path.dirname(os.path.dirname(ann_json_path)))

        results = {}
        results['camera_intrinsic_path'] = os.path.join(self.root, 'camera_intrinsic.json')
        results['img_prefix'] = self.cycle_path

        results = arti_utils.fecth_instances(results, ann_json_path)
        results = arti_utils.fetch_joint_params(results, joint_param_path, self.obj_name)
        if self.dname == 'Art':
            results = arti_utils.fetch_rest_trans(self.obj_name, results['urdf_id'], results)
        point_data_creator = pipelines.Create_Art_3d_scale_PointDataSapien(downsample_voxel=0.005, with_rgb=False)
        results = point_data_creator(results)

        mask_list = results['part_masks']
        mask = np.zeros_like(mask_list[0])
        for part_id, part_mask in enumerate(mask_list):
            part_id += 1
            mask[part_mask == 1] = part_id
        if self.dname == 'Art':
            img = Image.open(os.path.join(self.root, f'{self.obj_name}/train/color/{id}.jpg')).convert('RGB')
            depth = read_depth(os.path.join(self.root, f'{self.obj_name}/train/depth/{id}.png'))
        elif self.dname == 'ReArt':
            try:
                img = Image.open(os.path.join(self.cycle_path, f'color/{scene_name}/{id}.jpg')).convert('RGB')
            except Exception as e:
                print(f"Error encountered: {e}")
                img = Image.open(os.path.join(self.cycle_path, f'color/{scene_name}/{"{:06}".format(int(id)-1)}.jpg')).convert('RGB')
            depth = read_depth(os.path.join(self.cycle_path, f'depth/{scene_name}/{id}.png'))

        if self.add_noise:
            img = self.trancolor(img)
        img = np.array(img)[:, :, :3]
        img = img.transpose(2, 0, 1)

        if self.dname == 'Art':
            cld, choose = dpt_2_cld(depth, self.depth_scale, self.Art_K, self.dname)
        elif self.dname == 'ReArt':
            mask_obj = (mask != 0).astype(int)
            try:
                depth = depth * mask_obj
            except Exception as e:
                print(f"Error encountered: {e}")
                depth = np.full((720, 1280), 700)
                depth = depth * mask_obj
            cld, choose = dpt_2_cld(depth, self.depth_scale, self.ReArt_K, self.dname)
        cls = mask.flatten()[choose]

        rgb_lst = []
        for ic in range(img.shape[0]):
            rgb_lst.append(
                img[ic].flatten()[choose].astype(np.float32)
            )
        rgb_pt = np.transpose(np.array(rgb_lst), (1, 0)).copy()
        cld_rgb = np.concatenate((cld, rgb_pt), axis=1)

        # normal = get_normal(cld)[:, :3]
        # normal[np.isnan(normal)] = 0.0
        # cld_rgb_nrm = np.concatenate((cld_rgb, normal), axis=1)

        zero_indices = np.where(cls == 0)[0]

        cld = np.delete(cld, zero_indices, axis=0)
        cld_rgb = np.delete(cld_rgb, zero_indices, axis=0)
        # cld_rgb_nrm = np.delete(cld_rgb_nrm, zero_indices, axis=0)
        choose = np.delete(choose, zero_indices, axis=0)
        cls = np.delete(cls, zero_indices, axis=0)-1

        #scale_pts
        if self.dname == 'Art':
            try:
                split_clouds = {}
                split_rgb_pt = [None]*len(mask_list)
                split_choose = [None]*len(mask_list)
                split_cls = [None]*len(mask_list)
                unique_classes = np.unique(cls)
                for label in unique_classes:
                    split_clouds[label] = cld[cls == label]
                    split_rgb_pt[label] = cld_rgb[:, 3:][cls == label]
                    split_choose[label] = choose[cls == label][np.newaxis, :]
                    split_cls[label] = cls[cls == label][:, np.newaxis]
                cld, gtpose_list, cld_list = scale_pts(split_clouds, results, 10, self.part_num)
            except Exception as e:
                print(e)
                num1 = random.randint(0, cld_rgb.shape[0])
                num2 = random.randint(0, cld_rgb.shape[0] - num1)
                num3 = random.randint(0, cld_rgb.shape[0] - num1 - num2)
                num4 = cld_rgb.shape[0] - num1 - num2 - num3

                split_clouds = {
                    0: np.ones((num1, 3)),
                    1: np.ones((num2, 3)),
                    2: np.ones((num3, 3)),
                    3: np.ones((num4, 3))
                }
                split_rgb_pt = [np.ones((num1, 3)), np.ones((num2, 3)), np.ones((num3, 3)), np.ones((num4, 3))]
                split_choose = [np.ones((1, num1)), np.ones((1, num2)), np.ones((1, num3)), np.ones((1, num4))]
                split_cls = [np.ones((num1, 1)), np.ones((num2, 1)), np.ones((num3, 1)), np.ones((num4, 1))]

                cld, gtpose_list, cld_list = scale_pts(split_clouds, results, 10, self.part_num)

        elif self.dname == 'ReArt':
            gtpose_list = results['part_trans']

        gtpose_list = np.array(gtpose_list)
        results['gt_poses'] = gtpose_list
        if self.dname == 'Art':
            results['camera_pts'] = cld_list
            joint_data_creator = pipelines.LoadArtiJointDataSapien_scale()
            results = joint_data_creator(results, 10)
            joint_GT_creator = pipelines.CreateArtiJointGTSapien_scale()
            results = joint_GT_creator(results)
        elif self.dname == 'ReArt':
            cld_list = [None]*results['n_parts']
            for i in range(results['n_parts']):
                cld_list[i] = cld[cls == i]
            results['camera_pts'] = cld_list
            joint_data_creator = pipelines.LoadArtiJointDataSapien_scale()
            results = joint_data_creator(results, 1)
            joint_GT_creator = pipelines.CreateArtiJointGTSapien_scale()
            results = joint_GT_creator(results)

        axis = results['joint_orient']
        offset_heatmap = results['offset_heatmap']
        offset_unitvec = results['offset_unitvec']
        joint_cls = results['joint_cls']
        joint_type_gt = results['joint_type_gt']

        if self.dname=='Art':
            rgb_pt = np.vstack(split_rgb_pt)
            choose = np.concatenate(split_choose, axis=1)
            cls = np.concatenate(split_cls, axis=0)
        elif self.dname == 'ReArt':
            choose = choose[np.newaxis, :]
        cls = np.squeeze(cls)
        gtpose = gtpose_list[self.part_num][:3, :]
        cld_rgb[:, :3] = cld
        cld_rgb[:, 3:] = rgb_pt

        if self.transform3d is not None:
            radial_3d = self.transform3d(gtpose, self.kpts, self.kpt_num, cld)*10

        cld[:, 0] -= np.average(cld[:, 0])
        cld[:, 1] -= np.average(cld[:, 1])
        cld[:, 2] -= np.average(cld[:, 2])

        # cld_rgb[:, 0] -= np.average(cld_rgb[:, 0])
        # cld_rgb[:, 1] -= np.average(cld_rgb[:, 1])
        # cld_rgb[:, 2] -= np.average(cld_rgb[:, 2])

        choose_2 = np.array([i for i in range(len(choose[0, :]))])

        if len(choose_2) < self.n_sample_points:
            choose_2 = np.pad(choose_2, ((0, self.n_sample_points - len(choose_2))), mode='constant')
        if len(choose_2) > self.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:self.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, self.n_sample_points - len(choose_2)), 'wrap')

        cld = cld[choose_2, :]
        cld_rgb = cld_rgb[choose_2, :]
        # cld_rgb_nrm = cld_rgb_nrm[choose_2, :]
        choose = choose[:, choose_2]
        cls = cls[choose_2].astype(np.int32)
        radial_3d = radial_3d[choose_2, :]

        axis = axis[choose_2, :]
        offset_heatmap = offset_heatmap[choose_2]
        offset_unitvec = offset_unitvec[choose_2, :]
        joint_cls = joint_cls[choose_2].astype(np.int32)

        return id, ann_json_path,\
            torch.from_numpy(img.astype(np.float32)), \
            torch.from_numpy(cld.astype(np.float32)), \
            torch.from_numpy(cld_rgb.astype(np.float32)), \
            torch.LongTensor(choose.astype(np.int32)), \
            torch.LongTensor(cls.astype(np.int32)), \
            torch.from_numpy(radial_3d.astype(np.float32)), \
            torch.from_numpy(gtpose_list.astype(np.float32)), \
            torch.from_numpy(axis.astype(np.float32)), \
            torch.from_numpy(offset_heatmap.astype(np.float32)), \
            torch.from_numpy(offset_unitvec.astype(np.float32)), \
            torch.from_numpy(joint_cls.astype(np.int32)), \
            joint_type_gt

    def __len__(self):
        return len(self.ids)


