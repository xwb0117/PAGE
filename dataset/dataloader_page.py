# from torchvision.datasets import VOCSegmentation
from dataset.radius_dataset_3d import RMapDataset
from torch.utils import data
import numpy as np

Art_K = np.array([[914., 0., 320.],
                      [0., 914., 320.],
                      [0., 0., 1.]])

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    actual_xyz=xyz
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy,actual_xyz

def rgbd_to_point_cloud(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    #print(zs.min())
    #print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts, vs, us

class RData(RMapDataset):
    def __init__(self, root, dname, set='train', obj_name = 'laptop', part_num = 0, kpt_num=3, kpt_class = 'KP', n_sample_points = 2048, kpts_path = None):
        transform3d = self.transform3d
        #imageNet mean and std
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.dname = dname
        super().__init__(root,
                        dname,
                        set=set,
                        obj_name = obj_name,
                        transform=transform3d,
                        part_num=part_num,
                        kpt_num=kpt_num,
                        kpt_class=kpt_class,
                        n_sample_points=n_sample_points,
                        kpts_path = kpts_path
                        )

    def transform3d(self, gtpose, kpts, kpt_num, cld):
        gtpose = gtpose.copy()
        radial_3d = np.zeros((cld.shape[0], kpt_num))
        for idx, kpt in enumerate(kpts):
            dump, transformed_kpoint = project(np.array([kpt]),Art_K,gtpose)
            transformed_kpoint=transformed_kpoint[0]
            distance_list_3d = ((cld[:,0]-transformed_kpoint[0])**2+(cld[:,1]-transformed_kpoint[1])**2+(cld[:,2]-transformed_kpoint[2])**2)**0.5
            radial_3d[:, idx] = distance_list_3d

        return radial_3d

    def __len__(self):
        return len(self.ids)

def get_loader(opts):
    modes = ['train', 'test']
    train_loader = data.DataLoader(RData(opts.root_dataset,
                                        opts.dname,
                                        set=modes[0],
                                        obj_name = opts.class_name,
                                        part_num = opts.part_num,
                                        kpt_num = opts.kpt_num,
                                        kpt_class=opts.kpt_class,
                                        n_sample_points = opts.n_sample_points,
                                        kpts_path = opts.kpts_path),
                                        batch_size=int(opts.batch_size),
                                        shuffle=True,
                                        num_workers=10)
    val_loader = data.DataLoader(RData(opts.root_dataset,
                                        opts.dname,
                                       set=modes[1],
                                       obj_name = opts.class_name,
                                       part_num = opts.part_num,
                                       kpt_num = opts.kpt_num,
                                       kpt_class=opts.kpt_class,
                                       n_sample_points = opts.n_sample_points,
                                       kpts_path=opts.kpts_path),
                                       batch_size=int(opts.batch_size),
                                       shuffle=False,
                                       num_workers=10)
    return train_loader, val_loader
