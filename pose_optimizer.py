import open3d as o3d
import torch.utils.data
from scipy.spatial.transform import Rotation as R

cate_list = ['laptop', 'eyeglasses', 'dishwasher', 'drawer', 'scissors']

class PoseEstimator(torch.nn.Module):
    def __init__(self, num_parts, init_base_r, init_base_t, init_joint_state, device, joint_type='revolute', reg_weight=0.0):
        super(PoseEstimator, self).__init__()
        self.num_parts = num_parts
        self.num_joints = num_parts - 1
        self.device = device
        self.joint_type = joint_type
        assert joint_type in ('revolute', 'prismatic')
        self.reg_weight = reg_weight

        x, y, z, w = R.from_matrix(init_base_r.cpu().numpy()).as_quat()
        self.base_r_quat = torch.nn.Parameter(torch.tensor(
            [w, x, y, z], device=device, dtype=torch.float), requires_grad=True)  # q=a+bi+ci+di
        self.base_t = torch.nn.Parameter(init_base_t.clone().detach().to(device), requires_grad=True)
        self.joint_state = torch.nn.Parameter(init_joint_state.clone().detach().to(device), requires_grad=True)

    def chamfer_distance(self, x, y):
        try:
            x = x.to(torch.float64)
            y = y.to(torch.float64)
            dist_matrix = torch.cdist(x, y)

            min_dist_x_to_y, _ = torch.min(dist_matrix, dim=1)
            Dxy = torch.mean(min_dist_x_to_y, dim=0)

            min_dist_y_to_x, _ = torch.min(dist_matrix, dim=0)
            Dyy = torch.mean(min_dist_y_to_x, dim=0)

            chamfer_dist = torch.mean(Dxy + Dyy)

            return chamfer_dist

        except:
            print(x.shape, y.shape)
            tensor = torch.tensor(1.0, dtype=torch.float64, device='cuda:0', requires_grad=True)
            return tensor


    def forward(self, camera_pts, cad_pts, xyz, rpy, part_weight, mode='base'):
        assert mode in ('base', 'joint_single', 'all')

        cad_pts = [torch.cat([pts, torch.ones(pts.shape[0], 1, device=pts.device)], dim=-1) for pts in cad_pts]
        camera_pts = [torch.cat([pts, torch.ones(pts.shape[0], 1, device=pts.device)], dim=-1) for pts in camera_pts]

        base_r_quat = self.base_r_quat / torch.norm(self.base_r_quat)
        a, b, c, d = base_r_quat[0], base_r_quat[1], base_r_quat[2], base_r_quat[3]  # q=a+bi+ci+di
        base_rot_matrix = torch.stack([1 - 2 * c * c - 2 * d * d, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                                       2 * b * c + 2 * a * d, 1 - 2 * b * b - 2 * d * d, 2 * c * d - 2 * a * b,
                                       2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d,
                                       1 - 2 * b * b - 2 * c * c]).reshape(3, 3)
        base_transform = torch.cat([torch.cat([base_rot_matrix, self.base_t], dim=1),
                                    torch.tensor([[0., 0., 0., 1.]], device=self.device)], dim=0)

        camera_cad = base_transform.matmul(cad_pts[0].T).T
        camera_cad = camera_cad.detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(camera_cad[:, :3])

        cad_base = base_transform.matmul(cad_pts[0].T).T
        camera_base = camera_pts[0]
        base_objective = self.chamfer_distance(cad_base, camera_base)
        all_objective = part_weight[0] * base_objective

        new_joint_anchor_list = []
        new_joint_axis_list = []
        relative_transform_list = []
        camera_cad_child_list = []
        child_objective_list = []
        for joint_idx in range(self.num_joints):
            part_idx = joint_idx + 1
            # TODO: support kinematic tree depth > 2
            joint_loc, joint_axis = xyz[joint_idx], rpy[joint_idx]  # bs=1
            homo_joint_anchor = torch.cat([joint_loc, torch.ones(1, device=self.device)]).unsqueeze(1)
            new_joint_anchor = homo_joint_anchor
            new_joint_axis = joint_axis
            # new_joint_anchor = base_transform.matmul(homo_joint_anchor)[:3, 0]
            # new_joint_axis = base_rot_matrix.matmul(joint_axis.float())
            a, b, c = new_joint_anchor[0], new_joint_anchor[1], new_joint_anchor[2]
            u, v, w = new_joint_axis[0], new_joint_axis[1], new_joint_axis[2]
            if self.joint_type == 'revolute':
                cos = torch.cos(self.joint_state[joint_idx])
                sin = torch.sin(self.joint_state[joint_idx])
                relative_transform = torch.cat([torch.stack([u*u+(v*v+w*w)*cos, u*v*(1-cos)-w*sin, u*w*(1-cos)+v*sin,
                                                   (a*(v*v+w*w)-u*(b*v+c*w))*(1-cos)+(b*w-c*v)*sin,
                                                   u*v*(1-cos)+w*sin, v*v+(u*u+w*w)*cos, v*w*(1-cos)-u*sin,
                                                   (b*(u*u+w*w)-v*(a*u+c*w))*(1-cos)+(c*u-a*w)*sin,
                                                   u*w*(1-cos)-v*sin, v*w*(1-cos)+u*sin, w*w+(u*u+v*v)*cos,
                                                   (c*(u*u+v*v)-w*(a*u+b*v))*(1-cos)+(a*v-b*u)*sin]).reshape(3, 4),
                                                   torch.tensor([[0., 0., 0., 1.]], device=self.device)], dim=0)
            elif self.joint_type == 'prismatic':
                relative_transform = torch.cat([torch.cat([torch.eye(3, device=self.device),
                                                             (new_joint_axis*self.joint_state[joint_idx]).unsqueeze(1)], dim=1),
                                                torch.tensor([[0., 0., 0., 1.]], device=self.device)], dim=0).double()
            relative_transform_list.append(relative_transform.detach())

            camera_cad1 = relative_transform.matmul(base_transform).matmul(cad_pts[part_idx].T).T
            camera_cad1 = camera_cad1.detach().cpu().numpy()
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(camera_cad1[:, :3])
            pcd1.paint_uniform_color([1, 0, 0])
            camera_cad_child_list.append(pcd1)

            cad_child = relative_transform.matmul(base_transform).matmul(cad_pts[part_idx].T).T
            camera_child = camera_pts[part_idx]
            child_objective = self.chamfer_distance(cad_child, camera_child)
            child_objective_list.append(child_objective)
            all_objective += part_weight[part_idx] * child_objective

            new_joint_anchor_list.append(new_joint_anchor.detach())
            new_joint_axis_list.append(new_joint_axis.detach())
        all_objective /= self.num_parts
        new_joint_params_all = (torch.stack(new_joint_anchor_list, dim=0), torch.stack(new_joint_axis_list, dim=0))
        relative_transform_all = torch.stack(relative_transform_list, dim=0)
        return all_objective, base_objective, child_objective_list, base_transform.detach(), relative_transform_all, \
               new_joint_params_all, pcd, camera_cad_child_list


def optimize_pose(estimator, camera_pts, cad_pts, xyz, rpy, part_weight, rank=0, use_initial=False):
    estimator.base_r_quat.requires_grad_(True)
    estimator.base_t.requires_grad_(True)
    estimator.joint_state.requires_grad_(True)
    if use_initial:
        pass
    else:
        optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-2)
        last_loss = 0.
        for iter in range(3000):
            loss, _, _, _, _, _, pcd, camera_cad_child_list = estimator(camera_pts, cad_pts, xyz, rpy, part_weight, mode='all')
            if iter % 50 == 0:
                # camera_pts_combined = torch.cat(camera_pts, dim=0)
                # xyz_camera = camera_pts_combined.view(-1, 3).cpu().numpy()
                # pcd_camera = o3d.geometry.PointCloud()
                # pcd_camera.points = o3d.utility.Vector3dVector(xyz_camera)
                # pcd.paint_uniform_color([1, 0, 0])
                # pcd_camera.paint_uniform_color([0, 0, 1])
                # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
                # all_pcds = [pcd] + camera_cad_child_list + [pcd_camera]
                # o3d.visualization.draw_geometries(all_pcds, width=640, height=640)
                # if rank == 0:
                #     print('base_r + base_t + joint state + beta: iter {}, loss={:05f}'.format(iter, loss.item()))
                if abs(last_loss - loss.item()) < 0.05*1e-3:
                    break
                last_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    loss, loss_base, loss_child_list, base_transform, relative_transform_all, new_joint_params_all, pcd, pcd1 = estimator(camera_pts, cad_pts, xyz, rpy, part_weight)
    joint_state = estimator.joint_state.detach()
    return loss, loss_base, loss_child_list, base_transform, relative_transform_all, new_joint_params_all, joint_state
