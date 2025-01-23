import os
import torch
from torch.utils.data import Dataset
import arti_utils
import copy
import numpy as np
import open3d as o3d
import pipelines_sapien as pipelines

def rgbd_to_point_cloud(K, depth,rgb):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    #print(zs.min())
    #print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs,rgb[vs,us,0],rgb[vs,us,1],rgb[vs,us,2]]).T
    return pts

def get_scale_data(scale, results, ann_json_path, artimage_path, cat, part_num):
    rootDict = os.path.join(artimage_path, f'{cat}/train')

    joint_param_path = os.path.join(artimage_path, f'urdf_metas/{cat}/urdf_metas.json')
    results['camera_intrinsic_path'] = os.path.join(artimage_path, 'camera_intrinsic.json')
    results['img_prefix'] = rootDict

    results = arti_utils.fecth_instances(results, ann_json_path)
    results = arti_utils.fetch_joint_params(results, joint_param_path, cat)
    results = arti_utils.fetch_rest_trans(cat, results['urdf_id'], results)
    point_data_creator = pipelines.Create_Art_3d_scale_PointDataSapien(downsample_voxel=0.005, with_rgb=False)
    results = point_data_creator(results)

    urdf_id = results['urdf_id']

    part_trans_rest = results['part_trans_rest']
    part_trans = results['part_trans']
    parts_pts = results['parts_pts']
    parts_cls = results['parts_cls']
    n_parts = results['n_parts']

    scale_pts_list = []
    scale_pts_rest_list = []
    scale_camera_pts_list = []
    pcd_scale_list = []

    tran = part_trans[0]
    for i in range(n_parts):
        tran_rest = part_trans_rest[i]
        pts = parts_pts[i]
        pts_rest = parts_pts[i]
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

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

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
            pcd_scale_copy = copy.deepcopy(pcd_scale)
            pcd_scale_copy.transform(np.linalg.inv(T12))
            part_trans_scale[i] = T12
        else:
            part_trans_scale[i] = part_tran

    # o3d.visualization.draw_geometries(
    #     [pcd_can_list[0], pcd_can_list[1], axis])
    if scale_camera_pts_list[part_num].size == 0:
        print('Error', scale_camera_pts_list[part_num].shape)
        xyz = np.array([0., 0., 0.]).reshape(1, 3)
        RT = np.eye(4)
    else:
        xyz = scale_camera_pts_list[part_num]
        RT = part_trans_scale[part_num]

    xyz_color = results['parts_pts_feature'][part_num]

    return xyz, xyz_color

class ArtDataset(Dataset):
    def __init__(self,root,set,cat,n_parts,part_num,dname,scale,min_visible_points=2000,points_count_net = 1024) -> None:
        super().__init__()
        self.root = root
        self.set = set
        self.cat = cat
        self.part_num = part_num
        self.n_parts = n_parts
        self.dname = dname
        self.scale = scale
        self.points_count_net = points_count_net
        self.cycle_path = os.path.join(self.root, f'{self.cat}/train')
        self.split_path = os.path.join(self.root, self.cat)
        #standarization of ImageNet
        self.mean = np.array([0.485, 0.456, 0.406],dtype=np.float64)
        self.std = np.array([0.229, 0.224, 0.225],dtype=np.float64)

        #generate splits
        if dname == 'Art':
            if not os.path.exists(self.split_path):
                os.mkdir(self.split_path)
            with open(os.path.join(self.split_path, self.set+".txt")) as f:
                self.ids = f.readlines()
            self.ids = [x.strip('\n').replace('.json', '') for x in self.ids]
        elif dname == 'ReArt':
            self.ids = []
            if self.cat == 'box':
                self.root1 = os.path.join(root, 'box_train_v2/annotations')
                self.root2 = os.path.join(root, 'box_train_v2/annotations')
            elif self.cat == 'scissor':
                self.root1 = os.path.join(root, 'train_scissor/annotations')
                self.root2 = os.path.join(root, 'scissor_val/annotations')
            else:
                self.root1 = os.path.join(root, f'{self.cat}_train/annotations')
                self.root2 = os.path.join(root, f'{self.cat}_val/annotations')

            train_ids = [f"{i:06d}" for i in range(1, 5000)]
            if self.cat == 'box':
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

    def transform(self,pts,part_num):
        if pts.shape[0]<self.points_count_net:
            pts = np.concatenate((pts,np.zeros((self.points_count_net-pts.shape[0],3))),axis=0)
            #print(pts.shape)
        else:
            idx=np.random.choice(np.arange(pts.shape[0]),self.points_count_net,replace=False)
            pts = pts[idx]
        for i in range(3):
            #print(i)
            pts[:,i] -= np.mean(pts[:,i])
            # pts[:,i] /=self.coor_dims[part_num][i]

        pts = torch.from_numpy(pts).float()
        return pts

    def visualize_point_cloud(self, pts):
        pcd = o3d.geometry.PointCloud()
        points = pts[:, 0:3]
        pcd.points = o3d.utility.Vector3dVector(points)
        xyz_path1 = '/mnt/7797b2ec-a944-4795-abb2-f472a7fc833e/Data/ArtImage/dishwasher/urdf/11622/part_point_sample_rest/0_scale.xyz'
        xyz_path2 = '/mnt/7797b2ec-a944-4795-abb2-f472a7fc833e/Data/ArtImage/dishwasher/urdf/11622/part_point_sample_rest/1_scale.xyz'

        points1 = np.loadtxt(xyz_path1)
        points2 = np.loadtxt(xyz_path2)
        points1 = np.vstack([points1, points2])
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, axis])


    def __getitem__(self, index):
        if self.dname == 'Art':
            id = self.ids[index]
            cycle = id
            ann_json_path = os.path.join(self.cycle_path,'annotations', f'{cycle}.json')
        elif self.dname == 'ReArt':
            ann_json_path = self.ids[index]
        results = {}

        scale = self.scale
        if self.dname == 'Art':
            xyz, xyz_color = get_scale_data(scale, results, ann_json_path, self.root, self.cat, self.part_num)
        elif self.dname == 'ReArt':
            try:
                results={}
                results['img_prefix'] = os.path.dirname(os.path.dirname(os.path.dirname(ann_json_path)))
                results['camera_intrinsic_path'] = os.path.join(self.root, 'camera_intrinsic.json')
                results = arti_utils.fecth_instances(results, ann_json_path)
                joint_param_path = os.path.join(self.root, 'urdf_metas.json')
                results = arti_utils.fetch_joint_params(results, joint_param_path, self.cat)
                data_creator = pipelines.CreatePointDataSapien(with_rgb=True)
                results = data_creator(results)
                xyz = results['parts_pts'][self.part_num]
                xyz_color = results['parts_pts_feature'][self.part_num]
            except:
                xyz = np.ones((self.points_count_net, 3))

        pts = xyz
        pts=self.transform(pts,self.part_num)

        return pts

    def __len__(self):
        return len(self.ids)
