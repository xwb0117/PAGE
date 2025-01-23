from util.horn import HornPoseFitting
import utils
import torch
import numpy as np
from PIL import Image
import copy
from numba import jit, cuda, int32
import tqdm
from dataset.dataloader_page import get_loader
from models.PAGENet import PAGE_Estimator
import os
import open3d as o3d
import time
from numba import prange
from pose_optimizer import optimize_pose,PoseEstimator
import math
import arti_utils
import pipelines_sapien as pipelines

Art_K = np.array([[914., 0., 320.],
                      [0., 914., 320.],
                      [0., 0., 1.]])


# IO function from PVNet
def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    # pointc->actual scene
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    actual_xyz = xyz
    xyz = np.dot(xyz, K.T)
    uv = xyz[:, :2] / xyz[:, 2:]
    return uv, actual_xyz


def rgbd_to_point_cloud(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    # print(zs.min())
    # print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts


def rgbd_to_color_point_cloud(K, depth, rgb):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    r = rgb[vs, us, 0]
    g = rgb[vs, us, 1]
    b = rgb[vs, us, 2]
    # print(zs.min())
    # print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs, r, g, b]).T
    return pts

@cuda.jit
def cuda_internal1(VoteMap_3D,xyz,radius):
    factor = (3 ** 0.5) / 4
    m, i,j,k=cuda.grid(4)
    if m<xyz.shape[0] and i<VoteMap_3D.shape[0] and j<VoteMap_3D.shape[1] and k<VoteMap_3D.shape[2]:
        distance = ((i-xyz[m,0])**2+(j-xyz[m,1])**2+(k-xyz[m,2])**2)**0.5
        if radius - distance < factor and radius - distance >=0:
            VoteMap_3D[i,j,k]+=1

@cuda.jit
def fast_for_cuda(xyz_mm,radial_list_mm,VoteMap_3D):
    threadsperblock = (8, 8, 8, 8)
    blockspergrid_w = math.ceil(xyz_mm.shape[0] / threadsperblock[0])
    blockspergrid_x = math.ceil(VoteMap_3D.shape[0] / threadsperblock[1])
    blockspergrid_y = math.ceil(VoteMap_3D.shape[1] / threadsperblock[2])
    blockspergrid_z = math.ceil(VoteMap_3D.shape[2] / threadsperblock[3])
    blockspergrid = (blockspergrid_w, blockspergrid_x, blockspergrid_y, blockspergrid_z)
    blockspergrid = (8, 8, 8)
    cuda_internal1[blockspergrid, threadsperblock](VoteMap_3D,xyz_mm,radial_list_mm)


@cuda.jit
def update_VoteMap_3D(VoteMap_3D, xyz_mm, radial_list_mm):
    m = cuda.grid(1)
    if m < xyz_mm.shape[0]:
        factor = (3 ** 0.5) / 4
        xyz = xyz_mm[m]
        radius = radial_list_mm[m]
        radius = int32(math.floor(radius + 0.5))

        min_i = int32(max(0, math.floor(xyz[0] - radius)))
        max_i = int32(min(VoteMap_3D.shape[0], math.ceil(xyz[0] + radius)))
        min_j = int32(max(0, math.floor(xyz[1] - radius)))
        max_j = int32(min(VoteMap_3D.shape[1], math.ceil(xyz[1] + radius)))
        min_k = int32(max(0, math.floor(xyz[2] - radius)))
        max_k = int32(min(VoteMap_3D.shape[2], math.ceil(xyz[2] + radius)))

        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                for k in range(min_k, max_k):
                    distance = math.sqrt((i - xyz[0]) ** 2 + (j - xyz[1]) ** 2 + (k - xyz[2]) ** 2)
                    if radius - distance < factor and radius - distance > 0:
                        cuda.atomic.add(VoteMap_3D, (i, j, k), 1)

@jit(nopython=True, parallel=True)
# @jit(parallel=True)
def fast_for(xyz_mm, radial_list_mm, VoteMap_3D):
    factor = (3 ** 0.5) / 4
    for count in prange(xyz_mm.shape[0]):
        xyz = xyz_mm[count]
        radius = radial_list_mm[count]
        radius = int(np.around(radial_list_mm[count]))
        shape = VoteMap_3D.shape
        for i in prange(VoteMap_3D.shape[0]):
            for j in prange(VoteMap_3D.shape[1]):
                for k in prange(VoteMap_3D.shape[2]):
                    distance = ((i - xyz[0]) ** 2 + (j - xyz[1]) ** 2 + (k - xyz[2]) ** 2) ** 0.5
                    if radius - distance < factor and radius - distance > 0:
                        VoteMap_3D[i, j, k] += 1

    return VoteMap_3D

def Accumulator_3D(xyz, radial_list):
    try:
        acc_unit = 5
        # unit 5mm
        xyz_mm = xyz * 1000 / acc_unit  # point cloud is in meter

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz_mm)

        # recenter the point cloud
        x_mean_mm = np.mean(xyz_mm[:, 0])
        y_mean_mm = np.mean(xyz_mm[:, 1])
        z_mean_mm = np.mean(xyz_mm[:, 2])
        xyz_mm[:, 0] -= x_mean_mm
        xyz_mm[:, 1] -= y_mean_mm
        xyz_mm[:, 2] -= z_mean_mm

        radial_list_mm = radial_list * 100 / acc_unit  # radius map is in decimetre for training purpose

        # Ensure xyz_mm is not empty before performing min/max
        xyz_mm_min = xyz_mm.min()
        xyz_mm_max = xyz_mm.max()
        radius_max = radial_list_mm.max()

        zero_boundary = int(xyz_mm_min - radius_max) + 1

        if (zero_boundary < 0):
            xyz_mm -= zero_boundary
            # length of 3D vote map

        length = int(xyz_mm.max())

        VoteMap_3D = np.zeros((length + int(radius_max), length + int(radius_max), length + int(radius_max)))
        tic = time.perf_counter()
        start = time.time()
        VoteMap_3D = fast_for(xyz_mm, radial_list_mm, VoteMap_3D)
        final = time.time()
        # print("CPU time:", final - start)
        toc = time.perf_counter()

        center = np.argwhere(VoteMap_3D == VoteMap_3D.max())
        # print("debug center raw: ",center)
        center = center.astype("float64")
        if (zero_boundary < 0):
            center = center + zero_boundary

        # return to global coordinate
        center[0, 0] = (center[0, 0] + x_mean_mm + 0.5) * acc_unit
        center[0, 1] = (center[0, 1] + y_mean_mm + 0.5) * acc_unit
        center[0, 2] = (center[0, 2] + z_mean_mm + 0.5) * acc_unit

        # center = center*acc_unit+((3**0.5)/2)

        return center
    except:
        center = np.ones((1, 3))
        return center

def read_depth(path):
    if (path[-3:] == 'dpt'):
        with open(path) as f:
            h, w = np.fromfile(f, dtype=np.uint32, count=2)
            data = np.fromfile(f, dtype=np.uint16, count=w * h)
            depth = data.reshape((h, w))
    else:
        depth = np.asarray(Image.open(path)).copy()
    return depth

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

    pcd_cam_list = []
    pcd_can_list = []

    part_trans_scale = [None] * n_parts

    for i in range(n_parts):
        scale_pts1 = scale_pts_rest_list[i]
        scale_camera_pts1 = scale_camera_pts_list[i]
        pcd_scale = pcd_scale_list[i]
        part_tran = part_trans_rest[i]
        if i != 0:
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

    joint_xyz = results['joint_ins']['xyz']
    for i, xyz in enumerate(joint_xyz):
        xyz_scale = [x / 10 for x in xyz]
        results['joint_ins']['xyz'][i] = xyz_scale

    joint_data = results['joint_ins']

    return xyz, RT, urdf_id, joint_data

depthList = []

def estimate_6d_pose_Art(opts):
    horn = HornPoseFitting()
    depth_scale = 0.001
    class_name = opts.class_name
    n_parts = opts.num_classes

    print("Evaluation on ", class_name)
    rootPath = os.path.join(opts.root_dataset, class_name)
    rootpvPath = os.path.join(opts.root_dataset, class_name, 'train')

    # time consumption
    net_time = 0
    acc_time = 0
    general_counter = 0

    # counters
    bf_icp = 0
    af_icp = 0
    rot_diff_all_bf = [[] for _ in range(n_parts)]
    dist_diff_all_bf = [[] for _ in range(n_parts)]
    rot_diff_all_af = [[] for _ in range(n_parts)]
    dist_diff_all_af = [[] for _ in range(n_parts)]
    optimize_rot = [[] for _ in range(n_parts)]
    optimize_dist = [[] for _ in range(n_parts)]
    model_list = []

    if opts.using_ckpts:
        for i in range(n_parts):
            if opts.kpt_class=='KP':
                model_path = os.path.join(f'logs_page/{opts.dname}',f'{class_name}_{i}_PAGENet_{opts.kpt_num}/model_best_acc.pth.tar')
            model = PAGE_Estimator(num_classes=opts.num_classes, pcld_input_channels=opts.input_channel, num_kps=opts.kpt_num, num_points=opts.n_sample_points).cuda()
            # model = torch.nn.DataParallel(model)
            # checkpoint = torch.load(model_path)
            # model.load_state_dict(checkpoint);
            optim = torch.optim.Adam(model.parameters(), lr=1e-3)
            model, _, _, _ = utils.load_checkpoint(model, optim, model_path)
            model.eval()
            model_list.append(model)

    kpts_list = [None]*n_parts
    for i in range(n_parts):
        # kpts_list[i] = keypoints
        if opts.kpt_class=='KP':
            if opts.dname=='Art':
                keypoints = np.load(os.path.join('logs_kp/ArtImage/radii', f'{class_name}_{i}_{opts.kpt_num}/{class_name}_{i}_mean_points.npy'))
            elif opts.dname=='ReArt':
                keypoints = np.load(os.path.join('logs_kp/ReArtMix/radii',f'{class_name}_{i}_{opts.kpt_num}/{class_name}_{i}_mean_points.npy'))
        kpts_list[i] = keypoints

    if opts.dname == 'Art':
        urdf_path = f'{opts.root_dataset}/{class_name}/urdf'
    elif opts.dname == 'ReArt':
        class_name = opts.class_name[0].upper() + opts.class_name[1:]
        urdf_path = f'ReArt-48/{class_name}'
    directories = [d for d in os.listdir(urdf_path) if os.path.isdir(os.path.join(urdf_path, d))]

    max_radii_dm_list = [None]*n_parts
    for part_num in range(n_parts):
        max_radii_dm = np.zeros(opts.kpt_num)
        global_max_radii_dm = np.zeros(opts.kpt_num)
        for id in directories:
            if opts.dname=='Art':
                xyz_path = os.path.join(urdf_path, f'{id}/part_point_sample_rest/{part_num}_scale.xyz')
            elif opts.dname=='ReArt':
                xyz_path = os.path.join(urdf_path, f'{id}/meshes/{part_num}.txt')
            cad_model_points = np.loadtxt(xyz_path)

            for i in range(opts.kpt_num):
                distances = ((cad_model_points[:, 0] - kpts_list[part_num][i, 0]) ** 2
                             + (cad_model_points[:, 1] - kpts_list[part_num][i, 1]) ** 2
                             + (cad_model_points[:, 2] - kpts_list[part_num][i, 2]) ** 2) ** 0.5
                max_radii_dm[i] = distances.max()

            global_max_radii_dm = np.maximum(global_max_radii_dm, max_radii_dm)

        max_radii_dm = global_max_radii_dm*10
        max_radii_dm_list[part_num] = max_radii_dm

    # print(max_radii_dm)

    data = get_loader(opts)
    val_loader = data[1]

    # for id in test_list:
    for batch_idx, (id, ann_json_path, img, cld, cld_rgb, choose, cls, radial_3d, gtpose_list, axis, offset_heatmap, offset_unitvec, joint_cls, joint_type_gt) in tqdm.tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            ncols=80,
            leave=False):

        img, cld, cld_rgb, choose = img.cuda(), cld.cuda(), cld_rgb.cuda(), choose.cuda()
        id, xyz, cls, radial_3d, gtpose_list = id[0], cld_rgb[0][:, :3].cpu().numpy(), cls[0].cpu().numpy(), radial_3d[0].cpu().numpy(), gtpose_list[0].cpu().numpy()

        RTGT_list = [None]*n_parts
        RT_list = [None]*n_parts
        icpRT_list = [None]*n_parts
        cad_pts_list = [None]*n_parts
        camera_pts_list = [None]*n_parts

        for part_num in range(n_parts):
            if opts.using_ckpts:
                if (os.path.exists(model_path) == False):
                    raise ValueError(
                        f"{model_path} not found")

            RTGT = gtpose_list[part_num]

            # if filename in test_list:
            print("Evaluating ", id)
            estimated_kpts = np.zeros((opts.kpt_num, 3))
            transformed_gt_kpts = np.zeros((opts.kpt_num, 3))
            if opts.dname=='Art':
                ann_json_path = os.path.join(rootpvPath, f'annotations/{id}.json')
            artimage_path = opts.root_dataset

            results = {}
            if opts.dname=='Art':
                _, _, urdf_id, joint_data = get_scale_data(10.0, results, ann_json_path, artimage_path, class_name, part_num)
                xyz_path = os.path.join(urdf_path, f'{urdf_id}/part_point_sample_rest/{part_num}_scale.xyz')
            elif opts.dname=='ReArt':
                joint_param_path = os.path.join(opts.root_dataset, 'urdf_metas.json')
                results = arti_utils.fecth_instances(results, ann_json_path[0])
                urdf_id = results['urdf_id']
                results = arti_utils.fetch_joint_params(results, joint_param_path, opts.class_name)
                joint_data = results['joint_ins']
                xyz_path = os.path.join(urdf_path, f'{urdf_id}/meshes/{part_num}.txt')


            cad_model_points = np.loadtxt(xyz_path)

            if opts.using_ckpts:
                model = model_list[part_num]
                cls_pred, radial_3d_pred, pred_heatmap, pred_unitvec, pred_axis, pred_joint_cls = model(cld, img, choose)
                cls_pred, radial_3d_pred = cls_pred[0], radial_3d_pred[0]
                pred_labels = torch.argmax(cls_pred, dim=1)
                xyz = cld_rgb[0][:, :3][pred_labels == part_num].detach().cpu().numpy()
                radial_list_part = radial_3d_pred[pred_labels == part_num].detach().cpu().numpy()
                # radial_list_part = radial_3d[cls == part_num]
                # xyz = cld_rgb[0][:, :3][cls == part_num].detach().cpu().numpy()
                radial_list_part_gt = radial_3d[pred_labels.detach().cpu().numpy() == part_num]
                print('acc:', float(np.sum(np.where(np.abs(radial_list_part_gt - radial_list_part) <= 0.05, 1, 0)) / (
                            len(radial_list_part) * 3)))
            else:
                xyz = cld_rgb[0][:, :3][cls == part_num].detach().cpu().numpy()
                radial_list_part = radial_3d[cls == part_num]

            keypoint_count = 0
            xyz_icp = []
            keypoints = kpts_list[part_num]
            for keypoint in keypoints:
                keypoint = keypoints[keypoint_count]

                iter_count = 0

                transformed_gt_center = (np.dot(keypoints, RTGT[:3, :3].T) + RTGT[:3, 3].T)

                transformed_gt_center = transformed_gt_center[keypoint_count]

                if opts.using_ckpts:
                    radial_est = radial_list_part[:, keypoint_count]
                    radial_list = np.where(radial_est <= max_radii_dm[keypoint_count], radial_est, 0)
                else:
                    radial_est = radial_list_part[:, keypoint_count]
                    radial_list = np.where(radial_est <= max_radii_dm[keypoint_count], radial_est, 0)

                if keypoint_count == 0:
                    xyz_icp = xyz
                else:
                    for coor in xyz:
                        if not (coor == xyz_icp).all(1).any():
                            xyz_icp = np.append(xyz_icp, np.expand_dims(coor, axis=0), axis=0)

                tic = time.time_ns()
                center_s = Accumulator_3D(xyz, radial_list)/1000
                toc = time.time_ns()
                acc_time += toc - tic

                pre_center_off = math.inf

                estimated_center = center_s[0]

                '''
                index 0: original keypoint
                index 1: applied gt transformation keypoint
                index 2: network estimated keypoint
                '''
                centers = np.zeros((1, 3, 3))
                centers[0, 0] = keypoint
                centers[0, 1] = transformed_gt_center
                centers[0, 2] = estimated_center
                estimated_kpts[keypoint_count] = estimated_center
                transformed_gt_kpts[keypoint_count] = transformed_gt_center
                iter_count += 1

                keypoint_count += 1
                if keypoint_count > opts.kpt_num-1:
                    break
            kpts = keypoints[0:opts.kpt_num, :]
            RT = np.zeros((4, 4))
            horn.lmshorn(kpts, estimated_kpts, opts.kpt_num, RT)

            RT_list[part_num] = RT
            RTGT_list[part_num] = RTGT
            cad_pts_list[part_num] = torch.from_numpy(cad_model_points).cuda()
            camera_pts_list[part_num] = torch.from_numpy(xyz).cuda()

            dump, xyz_load_est_transformed = project(cad_model_points, Art_K, RT[0:3, :])
            # print(RTGT_mm)
            dump, xyz_load_transformed = project(cad_model_points, Art_K, RTGT[0:3, :])

            rot_diff = arti_utils.rot_diff_degree(RT[:3, :3], RTGT[:3, :3])
            if np.isnan(rot_diff):
                rot_diff = 0
            if rot_diff > 90:
                rot_diff = 180 - rot_diff
            if rot_diff > 50:
                rot_diff = 50
            rot_diff_all_bf[part_num].append(rot_diff)
            dis_diff = np.linalg.norm(RT[:3, 3] - RTGT[:3, 3])
            if np.isnan(dis_diff):
                dis_diff = 0
            dist_diff_all_bf[part_num].append(dis_diff)
            print(rot_diff, dis_diff)

            sceneGT = o3d.geometry.PointCloud()
            sceneEst = o3d.geometry.PointCloud()
            sceneGT.points = o3d.utility.Vector3dVector(xyz_load_transformed)
            sceneEst.points = o3d.utility.Vector3dVector(xyz_load_est_transformed)
            sceneGT.paint_uniform_color(np.array([0, 0, 1]))
            sceneEst.paint_uniform_color(np.array([1, 0, 0]))
            if opts.demo_mode:
                o3d.visualization.draw_geometries([sceneGT, sceneEst], window_name='gt vs est before icp')

            scene = o3d.geometry.PointCloud()
            scene.points = o3d.utility.Vector3dVector(xyz_icp)
            cad_model = o3d.geometry.PointCloud()
            cad_model.points = o3d.utility.Vector3dVector(cad_model_points)
            trans_init = RT
            distance = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).mean()
            threshold = distance
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria()
            reg_p2p = o3d.pipelines.registration.registration_icp(
                cad_model, scene, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria)

            rot_diff = arti_utils.rot_diff_degree(reg_p2p.transformation[:3, :3], RTGT[:3, :3])
            if np.isnan(rot_diff):
                rot_diff = 0
            if rot_diff > 90:
                rot_diff = 180 - rot_diff
            if rot_diff > 50:
                rot_diff = 50
            rot_diff_all_af[part_num].append(rot_diff)
            dis_diff = np.linalg.norm(reg_p2p.transformation[:3, 3] - RTGT[:3, 3])
            if np.isnan(dis_diff):
                dis_diff = 0
            dist_diff_all_af[part_num].append(dis_diff)
            icpRT_list[part_num] = reg_p2p.transformation

            cad_model.transform(reg_p2p.transformation)
            if opts.demo_mode:
                o3d.visualization.draw_geometries([sceneGT, cad_model], window_name='gt vs est after icp')

            general_counter += 1

        if opts.optimize:
            RTGT_child_r = [None] * (n_parts - 1)
            RTGT_child_t = [None] * (n_parts - 1)
            init_joint_state_list = [None] * (n_parts - 1)
            xyz_list = [None] * (n_parts - 1)
            rpy_list = [None] * (n_parts - 1)
            #optimize
            for i in range(n_parts):
                if i == 0:
                    RTGT_base_r = RTGT_list[i][:3, :3]
                    RTGT_base_t = RTGT_list[i][:3, 3]
                    init_base_r = RT_list[i][:3, :3]
                    init_base_t = RT_list[i][:3, 3]
                    # init_base_r = icpRT_list[i][:3, :3]
                    # init_base_t = icpRT_list[i][:3, 3]
                    joint_type = joint_data['type'][1]
                else:
                    RTGT_child_r[i-1] = RTGT_list[i][:3, :3]
                    RTGT_child_t[i-1] = RTGT_list[i][:3, 3]
                    init_child_r = RT_list[i][:3, :3]
                    # RTGT_child_r[i-1] = icpRT_list[i][:3, :3]
                    # RTGT_child_t[i-1] = icpRT_list[i][:3, 3]
                    # init_child_r = icpRT_list[i][:3, :3]

                    if opts.class_name == 'drawer' or opts.class_name == 'cutter':
                        init_joint_state_list[i-1] = arti_utils.calculate_rotation_angle(init_base_r, init_child_r)
                    else:
                        init_joint_state_list[i - 1] = -arti_utils.calculate_rotation_angle(init_base_r, init_child_r)

                    thres_r = 0.05
                    # Calculate the predicted joint info
                    pred_offset = (
                            pred_unitvec
                            * (1 - pred_heatmap.reshape(-1, 1))
                            * thres_r
                    )
                    pred_joint_pts = camera_pts_list[i] + pred_offset
                    pred_joint_points_index = np.where(
                        pred_joint_cls == i
                    )[0]
                    pred_joint_axis = np.median(
                        pred_axis[pred_joint_points_index], axis=0
                    )
                    pred_joint_pt = np.median(
                        pred_joint_pts[pred_joint_points_index], axis=0
                    )
                    xyz_list[i-1] = np.array(pred_joint_pt)
                    rpy_list[i-1] = np.array(pred_joint_axis)

            init_base_r = torch.from_numpy(init_base_r).cuda()
            init_base_t = torch.from_numpy(init_base_t).unsqueeze(-1).cuda()
            init_joint_state = torch.tensor(init_joint_state_list, dtype=torch.float32).cuda()
            xyz = torch.tensor(xyz_list, dtype=torch.float64).cuda()
            rpy = torch.tensor(rpy_list, dtype=torch.float64).cuda()
            device = 'cuda:0'
            reg_weight = 0

            xyz_camera = camera_pts_list
            cad = cad_pts_list

            pose_estimator = PoseEstimator(num_parts=opts.num_classes, init_base_r=init_base_r, init_base_t=init_base_t,
                                           init_joint_state=init_joint_state, device=device,
                                           joint_type=joint_type, reg_weight=reg_weight)

            loss_list = [[] for _ in range(n_parts)]
            transform_list = [[] for _ in range(n_parts)]
            for i in range(n_parts):
                part_weight = [1]* n_parts
                part_weight[i] = part_weight[i]*10
                loss, loss_base, loss_child_list, base_transform, relative_transform_all, new_joint_params_all, joint_state = optimize_pose(pose_estimator, xyz_camera, cad, xyz, rpy, part_weight)
                print(loss_base, loss_child_list)
                for j in range(n_parts):
                    if j == 0:
                        loss_list[i].append(loss_base.detach().cpu().numpy())
                        transform_list[i].append(base_transform.detach().cpu().numpy())
                    else:
                        loss_list[i].append(loss_child_list[j-1].detach().cpu().numpy())
                        child_transform = np.dot(relative_transform_all[j-1].cpu().numpy(), base_transform.detach().cpu().numpy())
                        transform_list[i].append(child_transform)

            rot_error_list = [float('inf')] * n_parts
            dist_error_list = [float('inf')] * n_parts
            if opts.choose=='loss':
                loss_final = [float('inf')] * n_parts
                transforms_final = [None] * n_parts
                for i in range(n_parts):
                    loss = loss_list[i]
                    transforms = transform_list[i]
                    for j in range(n_parts):
                        loss_base = loss[j]
                        if loss_base < loss_final[j]:
                            transforms_final[j] = transforms[j]

                for i in range(n_parts):
                    if i == 0:
                        base_rot_error = arti_utils.rot_diff_degree(transforms_final[i][:3, :3], RTGT_base_r[:3, :3])
                        base_dist_error = np.linalg.norm(transforms_final[i][:3, 3] - RTGT_base_t)
                        if np.isnan(base_rot_error):
                            base_rot_error = 0
                        if base_rot_error > 90:
                            base_rot_error = 180 - base_rot_error
                        if base_rot_error > 50:
                            base_rot_error = 50
                        print(base_rot_error, base_dist_error)
                        rot_error_list[i] = base_rot_error
                        dist_error_list[i] = base_dist_error
                    else:
                        child_rot_error = arti_utils.rot_diff_degree(transforms_final[i][:3, :3], RTGT_child_r[i - 1][:3, :3])
                        child_dist_error = np.linalg.norm(transforms_final[i][:3, 3] - RTGT_child_t[i - 1])
                        if np.isnan(child_rot_error):
                            child_rot_error = 0
                        if child_rot_error > 90:
                            child_rot_error = 180 - child_rot_error
                        if child_rot_error > 50:
                            child_rot_error = 50
                        print(child_rot_error, child_dist_error)
                        rot_error_list[i] = child_rot_error
                        dist_error_list[i] = child_dist_error


        for i in range(n_parts):
            optimize_rot[i].append(rot_error_list[i])
            optimize_dist[i].append(dist_error_list[i])

        for i in range(n_parts):
            rot_diff_all_bf_part = rot_diff_all_bf[i]
            dist_diff_all_bf_part = dist_diff_all_bf[i]
            rot_diff_all_af_part = rot_diff_all_af[i]
            dist_diff_all_af_part = dist_diff_all_af[i]
            optimize_rot_part = optimize_rot[i]
            optimize_dist_part = optimize_dist[i]
            if opts.optimize:
                print(f'part:{i}', 'before icp:', 'rot_error:', sum(rot_diff_all_bf_part) / len(rot_diff_all_bf_part), 'dis_error:',
                      sum(dist_diff_all_bf_part) / len(dist_diff_all_bf_part))
                print(f'part:{i}', 'after icp:', 'rot_error:', sum(rot_diff_all_af_part) / len(rot_diff_all_af_part), 'dis_error:',
                      sum(dist_diff_all_af_part) / len(dist_diff_all_af_part))
                print(f'part:{i}', 'after optimize:', 'rot_error:', sum(optimize_rot_part) / len(optimize_rot_part), 'dis_error:',
                      sum(optimize_dist_part) / len(optimize_dist_part))
            else:
                print(f'part:{i}', 'before icp:', 'rot_error:', sum(rot_diff_all_bf_part) / len(rot_diff_all_bf_part),
                      'dis_error:',
                      sum(dist_diff_all_bf_part) / len(dist_diff_all_bf_part))
                print(f'part:{i}', 'after icp:', 'rot_error:', sum(rot_diff_all_af_part) / len(rot_diff_all_af_part),
                      'dis_error:',
                      sum(dist_diff_all_af_part) / len(dist_diff_all_af_part))