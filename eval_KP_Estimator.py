import os
import argparse
import tqdm
import shutil
import numpy as np
import torch
from models import KPNet
import open3d as o3d

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser(description='KPNet Model Test')
    parser.add_argument('--keypointsNo', type=int, default=3, metavar='keypointsNo',
                        help='No, of Keypoints')

    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')

    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of test batch)')

    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train ')

    parser.add_argument('--optim', type=str, default='Adam',choices=['Adam', 'SGD'],
                        help='optimizer for training')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--cuda', type=bool, default=True,
                        help='enables GPU training')

    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points forwarded into the network')

    parser.add_argument('--min_vis_points', type=int, default=2000,
                        help='threshold for minimum segment points')

    parser.add_argument('--data_root', type=str, default='data/lm',
                        help='dataset root dir')

    parser.add_argument('--ckpt_root', type=str, default='',
                        help='root dir for ckpt')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dgcnn dropout rate')

    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='dgcnn dimension of embeddings')

    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='dgcnn random seed (default: 1)')

    parser.add_argument('--lambda_term', type=float, default=1,
                        help='gradient penalty lambda term')

    parser.add_argument('--gamma', type=float, default=-0.5,
                        help='gamma for dispersion loss')

    parser.add_argument('--vote_type', type=int, default=0, choices=[0,1,2],
                        help='vote type to train on. radii:0, vector:1, offset:2.')
    parser.add_argument('--category', type=str, default='laptop',
                        help='category of articulated objects')
    parser.add_argument('--nparts', type=int, default=2,
                        help='parts nums of articulated objects')
    parser.add_argument('--part_num', type=int, default=0,
                        help='parts num of articulated objects')
    parser.add_argument('--dname', type=str, default='Art',
                        help='dataset name')
    parser.add_argument('--scale', type=int, default=1,
                        help='scale factor')

    vote_types = {
            0:'radii',
            1:'vector',
            2:'offset'}
    args = parser.parse_args()
    dataset_name = args.data_root.split('/')[-1]
    log_dir = os.path.join('logs',dataset_name,vote_types[args.vote_type], f'{args.category}_{args.part_num}_{args.keypointsNo}')
    print(log_dir)
    # create log root
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    if args.cuda and torch.cuda.is_available():
        print("Using GPU!")
        device = 'cuda:0'
        torch.cuda.manual_seed(args.seed)
    else:
        device = 'cpu'
        torch.manual_seed(args.seed)

    model = KPNet(args, device).to(device)

    print(".....Saving keypoints to file", os.path.join(log_dir, 'keypoints.npy'), ".....")

    # load best model
    checkpoint = torch.load(os.path.join(log_dir, 'ckpt.pth.tar'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    pts_list = []
    colors_list = []

    if args.dname == 'Art':
        cad_models_path = f'/mnt/7797b2ec-a944-4795-abb2-f472a7fc833e/Data/ArtImage/{args.category}/urdf'
    elif args.dname == 'ReArt':
        args.category1 = args.category[0].upper() + args.category[1:]
        cad_models_path = f'/ReArt-48/{args.category1}'
    for filename in os.listdir(cad_models_path):
        if not os.path.isdir(os.path.join(cad_models_path, filename)):
            continue
        if args.dname == 'Art':
            model_path = os.path.join(cad_models_path, filename, 'part_point_sample_rest', f'{args.part_num}_scale.xyz')
        elif args.dname == 'ReArt':
            model_path = os.path.join(cad_models_path, filename, 'meshes', f'{args.part_num}.txt')
        pts = np.loadtxt(model_path)
        # print(colors.shape)
        # cad_models.append(cad_model)
        idx = np.random.choice(np.arange(pts.shape[0]), args.num_points, replace=True)
        pts = pts[idx]
        for i in range(3):
            # print(i)
            pts[:, i] -= np.mean(pts[:, i])
            # pts[:, i] /= coor_dims[args.part_num][i]
        pts_list.append(pts)
    pts_list = np.array(pts_list)
    pts_list = torch.from_numpy(pts_list).float().cuda(device)
    pts_list = torch.permute(pts_list, (0, 2, 1))
    print(pts_list.shape)
    est_kpts = model(pts_list)
    est_kpts = est_kpts.cpu().detach().numpy()

    key = est_kpts[1]/5
    cld = pts_list[1].permute(1,0).detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cld)
    pcd_key = o3d.geometry.PointCloud()
    pcd_key.points = o3d.utility.Vector3dVector(key)
    o3d.visualization.draw_geometries([pcd, pcd_key])

    np.save(os.path.join(log_dir, f'{args.category}_{args.part_num}_keypoints.npy'), est_kpts)